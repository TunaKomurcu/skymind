"""Görev planı şeması.

Model çıktısı şu üç tipten biridir (``type`` ayırıcı alan):

* ``plan``          – doğrulanmış, sıralı aksiyon listesi
* ``clarification`` – komut eksik/belirsiz; uydurmak yerine soru sorulur
* ``refusal``       – komut kapsam dışı ya da güvensiz

Tüm uzunluklar metre, hız m/s, süre saniye cinsindendir; birim dönüşümü planlayıcının işidir.
Sınırlar bilerek dar tutulmuştur: sınır dışı bir değer "uydurma" sayılır.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

MIN_ALT_M, MAX_ALT_M = 10.0, 5000.0
MIN_RADIUS_M, MAX_RADIUS_M = 50.0, 50_000.0
MIN_SPEED_MS, MAX_SPEED_MS = 5.0, 100.0
MAX_STEPS = 20


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Direction(StrEnum):
    N = "N"
    NE = "NE"
    E = "E"
    SE = "SE"
    S = "S"
    SW = "SW"
    W = "W"
    NW = "NW"


# --- Konum ------------------------------------------------------------------------------------


class Coordinates(_Strict):
    kind: Literal["coords"] = "coords"
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class NamedLocation(_Strict):
    """Operatörün adlandırdığı yer (ör. "Alfa noktası", "baraj")."""

    kind: Literal["named"] = "named"
    name: str = Field(min_length=1, max_length=80)


class RelativeLocation(_Strict):
    """Üsse (home) ya da mevcut konuma göre yön (+ opsiyonel mesafe).

    "kuzeydeki bölge" -> yalnızca yön; "kuzeyde 2 km" -> yön + mesafe.
    """

    kind: Literal["relative"] = "relative"
    direction: Direction
    distance_m: float | None = Field(default=None, gt=0, le=MAX_RADIUS_M)
    origin: Literal["home", "current"] = "home"


Location = Annotated[Coordinates | NamedLocation | RelativeLocation, Field(discriminator="kind")]

Altitude = Annotated[float, Field(ge=MIN_ALT_M, le=MAX_ALT_M)]


# --- Aksiyonlar -------------------------------------------------------------------------------


class Takeoff(_Strict):
    action: Literal["takeoff"] = "takeoff"
    altitude_m: Altitude


class Goto(_Strict):
    action: Literal["goto"] = "goto"
    target: Location
    altitude_m: Altitude | None = None
    speed_ms: float | None = Field(default=None, ge=MIN_SPEED_MS, le=MAX_SPEED_MS)


class Scan(_Strict):
    """Dairesel bölgeyi tara."""

    action: Literal["scan"] = "scan"
    center: Location
    radius_m: float = Field(ge=MIN_RADIUS_M, le=MAX_RADIUS_M)
    altitude_m: Altitude


class Patrol(_Strict):
    action: Literal["patrol"] = "patrol"
    waypoints: list[Location] = Field(min_length=2, max_length=10)
    altitude_m: Altitude
    loops: int = Field(default=1, ge=1, le=20)


class Loiter(_Strict):
    action: Literal["loiter"] = "loiter"
    center: Location
    radius_m: float = Field(default=100.0, ge=MIN_RADIUS_M, le=MAX_RADIUS_M)
    altitude_m: Altitude
    duration_s: int | None = Field(default=None, ge=10, le=6 * 3600)


class SetSpeed(_Strict):
    action: Literal["set_speed"] = "set_speed"
    speed_ms: float = Field(ge=MIN_SPEED_MS, le=MAX_SPEED_MS)


class ReturnToLaunch(_Strict):
    action: Literal["return_to_launch"] = "return_to_launch"


class Land(_Strict):
    action: Literal["land"] = "land"
    at: Location | None = None


Action = Annotated[
    Takeoff | Goto | Scan | Patrol | Loiter | SetSpeed | ReturnToLaunch | Land,
    Field(discriminator="action"),
]

ACTION_NAMES: tuple[str, ...] = (
    "takeoff",
    "goto",
    "scan",
    "patrol",
    "loiter",
    "set_speed",
    "return_to_launch",
    "land",
)


# --- Üst düzey çıktılar -----------------------------------------------------------------------


class MissionPlan(_Strict):
    type: Literal["plan"] = "plan"
    steps: list[Action] = Field(min_length=1, max_length=MAX_STEPS)


class Clarification(_Strict):
    type: Literal["clarification"] = "clarification"
    question: str = Field(min_length=3, max_length=300)
    missing: list[str] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def _missing_not_blank(self) -> Clarification:
        if any(not m.strip() for m in self.missing):
            raise ValueError("missing alanları boş olamaz")
        return self


class Refusal(_Strict):
    type: Literal["refusal"] = "refusal"
    reason: str = Field(min_length=3, max_length=300)


PlannerOutput = Annotated[MissionPlan | Clarification | Refusal, Field(discriminator="type")]

_adapter: TypeAdapter[MissionPlan | Clarification | Refusal] = TypeAdapter(PlannerOutput)


def parse_output(data: str | bytes | dict[str, object]) -> MissionPlan | Clarification | Refusal:
    """JSON metni ya da sözlüğü doğrulayıp tipli çıktıya çevirir (hata: ``ValidationError``)."""
    if isinstance(data, (str, bytes)):
        return _adapter.validate_json(data)
    return _adapter.validate_python(data)


def json_schema() -> dict[str, object]:
    """Constrained decoding ve dokümantasyon için JSON Schema."""
    return _adapter.json_schema()
