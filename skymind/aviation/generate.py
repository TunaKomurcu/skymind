"""Sentetik METAR/NOTAM üretici: önce yapı örneklenir, metin sonra yazılır (etiket hatası olamaz).

Ayrıştırıcıyı ve özetleyicileri ölçmek içindir. Üretici ve ayrıştırıcı aynı kişi tarafından yazıldı;
bu yüzden "ayrıştırma doğruluğu" biçim kapsamını gösterir, gerçek dünya genellemesini DEĞİL.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from skymind.aviation.metar import Cloud, Metar
from skymind.aviation.notam import Notam, QLine

STATIONS = ["LTBA", "LTFM", "LTAC", "LTAI", "LTBJ", "LTFJ", "LTCG", "LTAF", "LTCC", "LTAU", "LTBS"]
WX = ["-RA", "RA", "+RA", "-SN", "SN", "DZ", "FG", "BR", "HZ", "TSRA", "-SHRA", "+SHRA", "TS", "GR"]
VIS = [9999, 8000, 6000, 4000, 2500, 1200, 800, 400]
TRENDS = [None, None, "NOSIG", "BECMG", "TEMPO"]


def _wx_ok(codes: list[str]) -> list[str]:
    return list(dict.fromkeys(codes))


def random_metar(rng: random.Random) -> Metar:
    cavok = rng.random() < 0.3
    calm = rng.random() < 0.08
    variable = (not calm) and rng.random() < 0.1
    speed = 0 if calm else rng.randint(1, 35)
    gust = speed + rng.randint(8, 18) if (speed >= 8 and rng.random() < 0.3) else None
    clouds: list[Cloud] = []
    if not cavok and rng.random() < 0.9:
        h = rng.choice([2, 5, 8, 15, 30, 50])
        for _ in range(rng.randint(1, 3)):
            kind = "CB" if rng.random() < 0.06 else None
            clouds.append(Cloud(rng.choice(["FEW", "SCT", "BKN", "OVC"]), h * 100, kind))
            h += rng.choice([5, 10, 20, 30])
    temp = rng.randint(-10, 38)
    return Metar(
        raw="",
        station=rng.choice(STATIONS),
        day=rng.randint(1, 28),
        hour=rng.randint(0, 23),
        minute=rng.choice([0, 20, 50]),
        report_type=rng.choice(["METAR", "METAR", "SPECI"]),
        auto=rng.random() < 0.1,
        wind_dir=None if variable else (0 if calm else rng.randrange(10, 361, 10)),
        wind_variable=variable,
        wind_speed_kt=float(speed),
        gust_kt=None if gust is None else float(gust),
        visibility_m=9999 if cavok else rng.choice(VIS),
        cavok=cavok,
        weather=[] if cavok else _wx_ok([rng.choice(WX) for _ in range(rng.choice([0, 0, 1, 2]))]),
        clouds=clouds,
        temp_c=temp,
        dew_c=temp - rng.randint(0, 12),
        qnh_hpa=rng.randint(985, 1035),
        trend=rng.choice(TRENDS),
    )


def _t(v: int) -> str:
    return f"M{abs(v):02d}" if v < 0 else f"{v:02d}"


def metar_text(m: Metar) -> str:
    wd = "VRB" if m.wind_variable else f"{(m.wind_dir or 0):03d}"
    wind = f"{wd}{int(m.wind_speed_kt or 0):02d}"
    if m.gust_kt:
        wind += f"G{int(m.gust_kt):02d}"
    toks = [m.report_type, m.station, f"{m.day:02d}{m.hour:02d}{m.minute:02d}Z"]
    if m.auto:
        toks.append("AUTO")
    toks.append(wind + "KT")
    toks.append("CAVOK" if m.cavok else f"{m.visibility_m:04d}")
    toks += m.weather
    toks += [f"{c.amount}{(c.height_ft or 0) // 100:03d}{c.kind or ''}" for c in m.clouds]
    toks.append(f"{_t(m.temp_c or 0)}/{_t(m.dew_c or 0)}")
    toks.append(f"Q{m.qnh_hpa:04d}")
    if m.trend:
        toks.append(m.trend)
    return " ".join(toks)


# --- NOTAM ---------------------------------------------------------------------------------------

NOTAM_CASES = [
    ("MR", "LC", "RWY {rwy} CLSD DUE TO MAINT"),
    ("MX", "LC", "TWY {twy} CLSD"),
    ("LR", "AS", "RWY {rwy} EDGE LGT U/S"),
    ("NV", "AS", "VOR {twy}VOR U/S"),
    ("OB", "CH", "CRANE ERECTED 2NM FROM ARP, HGT 250FT AGL"),
    ("RT", "CA", "RESTRICTED AREA LTR{n} ACT. UAS OPR PROHIBITED"),
    ("WU", "LW", "UAS OPR WI {nm}NM RADIUS {coord}. MAX 400FT AGL"),
    ("CO", "CH", "TWR FREQ CHANGED TO 118.{f}"),
]


def random_notam(rng: random.Random) -> tuple[Notam, str]:
    subj, cond, tpl = rng.choice(NOTAM_CASES)
    start = datetime(2025, rng.randint(1, 12), rng.randint(1, 27), rng.randint(0, 23), 0)
    end = start + timedelta(hours=rng.choice([2, 6, 24, 72, 240]))
    lat, lon = rng.randint(3600, 4200), rng.randint(2600, 4400)
    coord = f"{lat:04d}N{lon:05d}E"
    runway = f"{rng.randint(1, 18):02d}/{(rng.randint(1, 18) + 18):02d}"
    text = tpl.format(
        rwy=runway,
        twy=rng.choice(["A", "B2", "C", "D1"]),
        n=rng.randint(10, 99),
        nm=rng.randint(1, 10),
        coord=f"{lat:04d}N {lon:05d}E",
        f=rng.choice(["500", "100", "350"]),
    )
    n = Notam(
        raw="",
        series=rng.choice("ABCD"),
        number=rng.randint(1, 9999),
        year=25,
        action=rng.choice(["NOTAMN", "NOTAMN", "NOTAMN", "NOTAMR"]),
        locations=[rng.choice(STATIONS)],
        valid_from=start,
        valid_to=end,
        estimated_end=rng.random() < 0.2,
        text=text,
        schedule=rng.choice([None, None, "DAILY 0800-1600"]),
        lower="SFC",
        upper=rng.choice(["UNL", "400FT AGL", "FL100"]),
    )
    n.q = QLine("LTAA", f"Q{subj}{cond}", subj, cond, "IV", "NBO", "A", 0, 999, coord, 5)
    if n.action == "NOTAMR":
        n.replaces = f"{n.series}{max(1, n.number - 1):04d}/25"
    return n, notam_text(n)


def _dtok(d: datetime) -> str:
    return d.strftime("%y%m%d%H%M")


def notam_text(n: Notam) -> str:
    assert n.q and n.valid_from and n.valid_to
    head = f"{n.series}{n.number:04d}/{n.year:02d} {n.action}" + (
        f" {n.replaces}" if n.replaces else ""
    )
    q = n.q
    lines = [
        head,
        f"Q) {q.fir}/{q.code}/{q.traffic}/{q.purpose}/{q.scope}/{q.lower_fl:03d}/{q.upper_fl:03d}/"
        f"{q.center}{q.radius_nm:03d}",
        f"A) {n.locations[0]} B) {_dtok(n.valid_from)} C) {_dtok(n.valid_to)}"
        + ("EST" if n.estimated_end else ""),
    ]
    if n.schedule:
        lines.append(f"D) {n.schedule}")
    lines.append(f"E) {n.text}")
    lines.append(f"F) {n.lower} G) {n.upper}")
    return "\n".join(lines)
