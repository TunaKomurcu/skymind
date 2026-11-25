"""METAR ayrıştırıcı ve Türkçe özet (kural tabanlı, klasik NLP).

Kapsam: ICAO biçimli METAR/SPECI (rüzgâr, görüş, CAVOK, hadise, bulut, sıcaklık/çiğ noktası, QNH,
trend). Tanınmayan gruplar ``unparsed`` listesinde döner, sessizce atılmaz. RMK sonrası yok sayılır.
Eşikler (``Limits``) prototip değerleridir, gerçek bir operasyon kuralı değildir.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

KT_TO_MS = 0.514444

WEATHER_TR = {
    "DZ": "çise",
    "RA": "yağmur",
    "SN": "kar",
    "SG": "kar taneleri",
    "PL": "buz paleti",
    "GR": "dolu",
    "GS": "küçük dolu",
    "FG": "sis",
    "BR": "hafif sis",
    "HZ": "pus",
    "FU": "duman",
    "DU": "toz",
    "SA": "kum",
    "VA": "volkanik kül",
    "SQ": "bora",
    "FC": "hortum",
    "SS": "kum fırtınası",
    "DS": "toz fırtınası",
    "UP": "bilinmeyen yağış",
}
DESCRIPTOR_TR = {
    "MI": "sığ",
    "BC": "parçalı",
    "PR": "kısmi",
    "DR": "alçak savrulan",
    "BL": "yüksek savrulan",
    "SH": "sağanak",
    "TS": "gök gürültülü",
    "FZ": "donan",
}
CLOUD_TR = {
    "FEW": "az bulutlu",
    "SCT": "parçalı bulutlu",
    "BKN": "çok bulutlu",
    "OVC": "kapalı",
}
TREND_TR = {
    "NOSIG": "önemli değişiklik beklenmiyor",
    "BECMG": "değişim bekleniyor (BECMG)",
    "TEMPO": "geçici değişim bekleniyor (TEMPO)",
}
CARDINALS = ["kuzey", "kuzeydoğu", "doğu", "güneydoğu", "güney", "güneybatı", "batı", "kuzeybatı"]

_PHENOM = "|".join(WEATHER_TR)
_DESC = "|".join(DESCRIPTOR_TR)
RE_STATION = re.compile(r"^[A-Z]{4}$")
RE_TIME = re.compile(r"^(\d{2})(\d{2})(\d{2})Z$")
RE_WIND = re.compile(r"^(\d{3}|VRB)(\d{2,3})(?:G(\d{2,3}))?(KT|MPS)$")
RE_WIND_VAR = re.compile(r"^(\d{3})V(\d{3})$")
RE_VIS = re.compile(r"^(\d{4})$")
RE_RVR = re.compile(r"^R\d{2}[LCR]?/[PM]?\d{4}(?:V[PM]?\d{4})?[UDN]?$")
RE_WX = re.compile(rf"^([-+]|VC)?((?:{_DESC})?)((?:{_PHENOM})+)$|^([-+]|VC)?((?:{_DESC}))$")
RE_CLOUD = re.compile(r"^(FEW|SCT|BKN|OVC)(\d{3})(CB|TCU)?$")
RE_VV = re.compile(r"^VV(\d{3}|///)$")
RE_TEMP = re.compile(r"^(M?\d{2})/(M?\d{2})?$")
RE_QNH = re.compile(r"^(Q)(\d{4})$|^(A)(\d{4})$")
SKY_CLEAR = {"SKC", "NSC", "NCD", "CLR"}


@dataclass(frozen=True)
class Limits:
    """Prototip uçuş uygunluk eşikleri (SHGM kuralı değildir)."""

    max_wind_ms: float = 10.0
    min_visibility_m: int = 5000
    min_ceiling_ft: int = 500


@dataclass(frozen=True)
class Cloud:
    amount: str  # FEW/SCT/BKN/OVC/VV
    height_ft: int | None
    kind: str | None = None  # CB/TCU


@dataclass
class Metar:
    raw: str
    station: str
    day: int
    hour: int
    minute: int
    report_type: str = "METAR"
    auto: bool = False
    corrected: bool = False
    wind_dir: int | None = None  # None yalnızca VRB'de; sakin rüzgâr (00000KT) 0 verir
    wind_variable: bool = False
    wind_speed_kt: float | None = None
    gust_kt: float | None = None
    visibility_m: int | None = None
    cavok: bool = False
    weather: list[str] = field(default_factory=list)  # "-RA", "+TSRA", "FG" ...
    clouds: list[Cloud] = field(default_factory=list)
    sky_clear: bool = False
    temp_c: int | None = None
    dew_c: int | None = None
    qnh_hpa: int | None = None
    trend: str | None = None
    trend_detail: list[str] = field(default_factory=list)  # trend sonrası gruplar (ham)
    unparsed: list[str] = field(default_factory=list)


def _temp(tok: str) -> int:
    return -int(tok[1:]) if tok.startswith("M") else int(tok)


def parse_metar(text: str) -> Metar:
    toks = text.replace("=", " ").split()
    if "RMK" in toks:
        toks = toks[: toks.index("RMK")]
    report_type = "METAR"
    if toks and toks[0] in ("METAR", "SPECI"):
        report_type = toks.pop(0)
    corrected = False
    if toks and toks[0] == "COR":
        corrected = True
        toks.pop(0)
    if len(toks) < 2 or not RE_STATION.match(toks[0]):
        raise ValueError("istasyon kodu bulunamadı")
    m = RE_TIME.match(toks[1])
    if not m or not (1 <= int(m[1]) <= 31 and int(m[2]) < 24 and int(m[3]) < 60):
        raise ValueError("geçerli gün/saat grubu bulunamadı")
    out = Metar(
        raw=text.strip(),
        station=toks[0],
        day=int(m[1]),
        hour=int(m[2]),
        minute=int(m[3]),
        report_type=report_type,
        corrected=corrected,
    )
    rest = toks[2:]
    for i, tok in enumerate(rest):
        if tok in TREND_TR:  # trend sonrası gruplar tahmindir; ana rasat alanlarını ezmesin
            out.trend, out.trend_detail = tok, rest[i + 1 :]
            break
        _apply(out, tok)
    return out


def _apply(out: Metar, tok: str) -> None:
    if tok == "AUTO":
        out.auto = True
    elif tok == "CAVOK":
        out.cavok = True
        out.visibility_m = 9999
    elif m := RE_WIND.match(tok):
        speed = float(m[2]) * (1 if m[4] == "KT" else 1 / KT_TO_MS)
        gust = None if m[3] is None else float(m[3]) * (1 if m[4] == "KT" else 1 / KT_TO_MS)
        out.wind_speed_kt = round(speed, 1)
        out.gust_kt = None if gust is None else round(gust, 1)
        out.wind_variable = m[1] == "VRB"
        out.wind_dir = None if m[1] == "VRB" else int(m[1])
    elif RE_WIND_VAR.match(tok) or RE_RVR.match(tok):
        pass  # rüzgâr yön değişimi ve RVR: özette kullanılmıyor, ama tanınan geçerli gruplardır
    elif m := RE_VIS.match(tok):
        out.visibility_m = int(m[1])
    elif m := RE_CLOUD.match(tok):
        out.clouds.append(Cloud(m[1], int(m[2]) * 100, m[3]))
    elif m := RE_VV.match(tok):
        out.clouds.append(Cloud("VV", None if m[1] == "///" else int(m[1]) * 100))
    elif tok in SKY_CLEAR:
        out.sky_clear = True
    elif m := RE_TEMP.match(tok):
        out.temp_c = _temp(m[1])
        out.dew_c = None if m[2] is None else _temp(m[2])
    elif m := RE_QNH.match(tok):
        out.qnh_hpa = int(m[2]) if m[1] else round(int(m[4]) / 100 * 33.8639)
    elif RE_WX.match(tok):
        out.weather.append(tok)
    else:
        out.unparsed.append(tok)


# --- Türkçe özet ---------------------------------------------------------------------------------


def _cardinal(deg: int) -> str:
    return CARDINALS[round(deg / 45) % 8]


def _weather_tr(code: str) -> str:
    m = RE_WX.match(code)
    assert m is not None
    intensity = m[1] or m[4] or ""
    desc = m[2] or m[5] or ""
    phen = m[3] or ""
    words = [DESCRIPTOR_TR[desc]] if desc else []
    words += [WEATHER_TR[phen[i : i + 2]] for i in range(0, len(phen), 2)]
    text = " ".join(words)
    prefix = {"-": "hafif ", "+": "kuvvetli ", "VC": "çevrede "}.get(intensity, "")
    return prefix + text


def fmt_visibility(m: int) -> str:
    if m >= 9999:
        return "10 km veya üzeri"
    if m >= 1000:
        return f"{m / 1000:g} km"
    return f"{m} m"


def wind_ms(kt: float) -> int:
    return round(kt * KT_TO_MS)


def summarize_tr(m: Metar) -> str:
    """Kaynakta olan her sayıyı doğrudan taşıyan kısa Türkçe özet."""
    parts = [f"{m.station} {m.day}. gün {m.hour:02d}:{m.minute:02d} UTC rasadı"]
    if m.wind_speed_kt is not None:
        if m.wind_speed_kt == 0:
            wind = "rüzgâr sakin"
        else:
            src = (
                "değişken yönlü"
                if m.wind_variable
                else f"{_cardinal(m.wind_dir or 0)} ({m.wind_dir}°)"
            )
            wind = f"rüzgâr {src}, {m.wind_speed_kt:g} kt (≈{wind_ms(m.wind_speed_kt)} m/s)"
            if m.gust_kt:
                wind += f", hamle {m.gust_kt:g} kt (≈{wind_ms(m.gust_kt)} m/s)"
        parts.append(wind)
    if m.cavok:
        parts.append("CAVOK (görüş 10 km veya üzeri, önemli bulut ve hadise yok)")
    elif m.visibility_m is not None:
        parts.append(f"görüş {fmt_visibility(m.visibility_m)}")
    for w in m.weather:
        parts.append(_weather_tr(w))
    if not m.cavok:
        if m.clouds:
            parts.append(
                "bulut: "
                + ", ".join(
                    (
                        CLOUD_TR.get(c.amount, "dikey görüş")
                        + (f" {c.height_ft} ft" if c.height_ft else "")
                    )
                    + (f" ({c.kind})" if c.kind else "")
                    for c in m.clouds
                )
            )
        elif m.sky_clear:
            parts.append("önemli bulut yok")
    if m.temp_c is not None:
        t = f"sıcaklık {m.temp_c}°C"
        if m.dew_c is not None:
            t += f", çiğ noktası {m.dew_c}°C"
        parts.append(t)
    if m.qnh_hpa is not None:
        parts.append(f"QNH {m.qnh_hpa} hPa")
    if m.trend:
        detail = f": {' '.join(m.trend_detail)}" if m.trend_detail else ""
        parts.append(TREND_TR[m.trend] + detail)
    return "; ".join(parts) + "."


def advisory_tr(m: Metar, limits: Limits = Limits()) -> list[str]:
    """Prototip eşiklere göre uyarılar; boş liste: eşik aşımı yok (uçuş izni anlamına GELMEZ)."""
    out: list[str] = []
    top = max(x for x in (m.wind_speed_kt, m.gust_kt) if x is not None) if m.wind_speed_kt else 0
    if wind_ms(top) > limits.max_wind_ms:
        out.append(f"rüzgâr/hamle {wind_ms(top)} m/s: {limits.max_wind_ms:g} m/s sınırının üstünde")
    if m.visibility_m is not None and m.visibility_m < limits.min_visibility_m:
        out.append(f"görüş {fmt_visibility(m.visibility_m)}: {limits.min_visibility_m} m altında")
    ceilings = [c.height_ft for c in m.clouds if c.amount in ("BKN", "OVC", "VV") and c.height_ft]
    if ceilings and min(ceilings) < limits.min_ceiling_ft:
        out.append(f"bulut tavanı {min(ceilings)} ft: {limits.min_ceiling_ft} ft altında")
    for w in m.weather:
        if "TS" in w or w.lstrip("+-").startswith(("FZ", "GR", "GS", "FC", "SS", "DS")):
            out.append(f"tehlikeli hadise: {_weather_tr(w)}")
    return out
