"""Özet sadakati: özetteki her sayı kaynaktan türetilebilir mi? (uydurma sayı denetimi)

Bir özetteki sayılar kaynak metindeki sayılar ya da bunlardan hesaplanan değerler (kt -> m/s,
m -> km, tarih/saat bileşenleri) kümesinde değilse "desteklenmeyen" sayılır. Karşılaştırma işaretten
bağımsızdır ("0800-1600" gibi tireli aralıklar için). Bu denetim anlamsal doğruluğu değil yalnızca
sayısal uydurmayı yakalar: yanlış yönü, yanlış hadiseyi ya da eksik bilgiyi görmez.
Kural tabanlı özetleyici için yapı gereği 0 çıkar; asıl amaç LLM özetlerini aynı ölçütle ölçmektir.
"""

from __future__ import annotations

import re

from skymind.aviation.metar import KT_TO_MS, Metar
from skymind.aviation.notam import Notam

_DATE = re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{4})")
_TIME = re.compile(r"(\d{1,2}):(\d{2})")
_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def numbers_in(text: str) -> list[float]:
    """Metindeki sayılar (işaretsiz). Tarih ve saatler bileşenlerine ayrılır."""
    text = _DATE.sub(r"\1 \2 \3", text)
    text = _TIME.sub(r"\1 \2", text)
    return [float(m.replace(",", ".")) for m in _NUM.findall(text)]


def source_numbers_metar(m: Metar) -> set[float]:
    allowed = set(numbers_in(m.raw))
    allowed |= {float(m.day), float(m.hour), float(m.minute)}
    for kt in (m.wind_speed_kt, m.gust_kt):
        if kt is not None:
            allowed |= {kt, float(round(kt * KT_TO_MS))}
    for v in (m.wind_dir, m.temp_c, m.dew_c, m.qnh_hpa):
        if v is not None:
            allowed.add(abs(float(v)))
    if m.visibility_m is not None:
        allowed |= {float(m.visibility_m), m.visibility_m / 1000}
        if m.visibility_m >= 9999:
            allowed.add(10.0)
    allowed |= {float(c.height_ft) for c in m.clouds if c.height_ft}
    return allowed


def source_numbers_notam(n: Notam) -> set[float]:
    allowed = set(numbers_in(n.raw)) | set(numbers_in(n.text))
    allowed |= {float(n.number), float(n.year)}
    for d in (n.valid_from, n.valid_to):
        if d:
            allowed |= {float(d.day), float(d.month), float(d.year), float(d.hour), float(d.minute)}
    if n.q:
        allowed |= {float(n.q.lower_fl), float(n.q.upper_fl)}
        if n.q.radius_nm:
            allowed.add(float(n.q.radius_nm))
    for s in (n.schedule, n.lower, n.upper):
        if s:
            allowed |= set(numbers_in(s))
    return allowed


def unsupported_numbers(summary: str, allowed: set[float]) -> list[float]:
    """Özetteki, kaynak kümesinde bulunmayan sayılar (boş liste: hepsi desteklenmiş)."""
    return [x for x in numbers_in(summary) if not any(abs(x - a) < 1e-6 for a in allowed)]
