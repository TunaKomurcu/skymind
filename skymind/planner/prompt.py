"""Planlayıcı için istem ve sohbet biçimi (baseline, SFT ve DPO'da aynı kullanılır)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = """Sen bir İHA görev planlayıcısısın.
Operatörün Türkçe komutunu YALNIZCA JSON olarak yanıtla.

Üç çıktı tipi vardır:
{"type":"plan","steps":[<aksiyon>,...]}
{"type":"clarification","question":"<Türkçe soru>","missing":["<alan>",...]}
{"type":"refusal","reason":"<Türkçe gerekçe>"}

Aksiyonlar (action):
takeoff(altitude_m) | goto(target, altitude_m?, speed_ms?) | scan(center, radius_m, altitude_m) |
patrol(waypoints[2-10], altitude_m, loops?) | loiter(center, radius_m?, altitude_m, duration_s?) |
set_speed(speed_ms) | return_to_launch | land(at?)

Konum (location):
{"kind":"relative","direction":"N|NE|E|SE|S|SW|W|NW","distance_m"?} |
{"kind":"named","name":"<yer adı>"} | {"kind":"coords","lat":..,"lon":..}

Birimler: metre, m/s, saniye (km ve dakikayı çevir). Sınırlar: irtifa 10-5000 m, yarıçap 50-50000 m,
hız 5-100 m/s.
Kurallar:
- Zorunlu bir bilgi komutta YOKSA uydurma; clarification ile sor ve eksik alanları "missing"e yaz.
- Silah kullanımı, güvensiz uçuş, kapsam dışı istek ya da sınır dışı değer için refusal döndür.
- Başka hiçbir metin yazma."""


SYSTEM_SHORT = (
    "Operatörün Türkçe İHA komutunu JSON görev planına çevir. Zorunlu bilgi eksikse "
    "clarification, kapsam dışı/güvensiz/sınır dışı ise refusal döndür. Yalnızca JSON yaz."
)


def to_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def messages(
    command: str, answer: dict[str, Any] | None = None, system: str = SYSTEM_PROMPT
) -> list[dict[str, str]]:
    """Sohbet mesajları; ``answer`` verilirse asistan turu de eklenir (eğitim için)."""
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": command},
    ]
    if answer is not None:
        msgs.append({"role": "assistant", "content": to_json(answer)})
    return msgs


def load_split(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return rows[:limit] if limit else rows


def sft_example(row: dict[str, Any], system: str = SYSTEM_SHORT) -> dict[str, Any]:
    """Prompt/completion biçimi: kayıp yalnızca asistan yanıtında hesaplanır."""
    full = messages(row["command"], row["gold"], system)
    return {"prompt": full[:-1], "completion": full[-1:]}


def dpo_example(row: dict[str, Any], system: str = SYSTEM_SHORT) -> dict[str, Any] | None:
    """Tercih çifti: gold (chosen) vs uydurma plan (rejected). Plan satırlarında çift yok."""
    if "rejected" not in row:
        return None
    prompt = messages(row["command"], system=system)
    return {
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": to_json(row["gold"])}],
        "rejected": [{"role": "assistant", "content": to_json(row["rejected"])}],
    }
