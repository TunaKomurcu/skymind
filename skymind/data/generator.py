"""Şablon/gramer tabanlı sentetik veri üreteci.

Önce plan (altın çıktı) örneklenir, sonra Türkçe cümle yazılır; etiket hatası olamaz.
Üç kategori:

* ``plan``          – tüm gerekli bilgi cümlede var
* ``clarification`` – bir/birkaç gerekli alan cümleden çıkarılmış. Çıkarılan alanların
                      uydurulmuş değerlerle dolu hali ``rejected`` (DPO için kötü cevap) olur.
* ``refusal``       – silah, güvensiz, kapsam dışı ya da sınır dışı değer
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Callable
from typing import Any

from skymind.schemas import parse_output
from skymind.schemas.plan import MAX_ALT_M, MAX_RADIUS_M, MAX_SPEED_MS

Step = dict[str, Any]
Frag = tuple[str, Step]  # (cümle parçası, tam adım)

ALTITUDES = [100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000]
RADII = [100, 200, 500, 1000, 1500, 2000, 3000, 5000, 10000]
DISTANCES = [500, 1000, 2000, 3000, 5000]
SPEEDS_MS = [5, 10, 15, 20, 25, 30]  # km/sa karşılığı: x3.6

# yön: (kod, ad, yönelme -e, bulunma -de, sıfat -deki)
DIRS = {
    "N": ("kuzey", "kuzeye", "kuzeyde", "kuzeydeki"),
    "NE": ("kuzeydoğu", "kuzeydoğuya", "kuzeydoğuda", "kuzeydoğudaki"),
    "E": ("doğu", "doğuya", "doğuda", "doğudaki"),
    "SE": ("güneydoğu", "güneydoğuya", "güneydoğuda", "güneydoğudaki"),
    "S": ("güney", "güneye", "güneyde", "güneydeki"),
    "SW": ("güneybatı", "güneybatıya", "güneybatıda", "güneybatıdaki"),
    "W": ("batı", "batıya", "batıda", "batıdaki"),
    "NW": ("kuzeybatı", "kuzeybatıya", "kuzeybatıda", "kuzeybatıdaki"),
}

# adlandırılmış yer: ad -> yönelme hali
PLACES = {
    "Alfa noktası": "Alfa noktasına",
    "Bravo noktası": "Bravo noktasına",
    "Charlie noktası": "Charlie noktasına",
    "baraj": "baraja",
    "köprü": "köprüye",
    "liman": "limana",
    "radar tepesi": "radar tepesine",
    "depo": "depoya",
    "pist": "piste",
    "sınır karakolu": "sınır karakoluna",
    "orman sahası": "orman sahasına",
}

PREFIX = [
    "",
    "",
    "",
    "lütfen ",
    "hemen ",
    "şimdi ",
    "tamam, ",
    "Kartal, ",
    "Şahin-1, ",
    "operatör: ",
]
SUFFIX = ["", "", "", "", " lütfen", " acil", " teyit et", " bildir"]

# --- yardımcılar ------------------------------------------------------------------------------


def tr_upper_first(s: str) -> str:
    if not s:
        return s
    first = {"i": "İ", "ı": "I"}.get(s[0], s[0].upper())
    return first + s[1:]


def num_text(x: float) -> str:
    return f"{x:g}".replace(".", ",")


def dist_text(m: int, rng: random.Random) -> str:
    if m >= 1000:
        km = num_text(m / 1000)
        return rng.choice([f"{km} km", f"{km} kilometre", f"{m} m"])
    return rng.choice([f"{m} m", f"{m} metre"])


def alt_phrase(m: int, rng: random.Random) -> str:
    return rng.choice(["{} irtifada", "{} yükseklikte", "{} yükseklikten"]).format(
        dist_text(m, rng)
    )


def pick_loc(rng: random.Random, allow_coords: bool = False) -> tuple[dict[str, Any], str]:
    """(konum sözlüğü, tür) – tür: relative/relative_d/named/coords."""
    r = rng.random()
    if allow_coords and r < 0.12:
        lat, lon = round(rng.uniform(36, 42), 2), round(rng.uniform(26, 44), 2)
        return {"kind": "coords", "lat": lat, "lon": lon}, "coords"
    if r < 0.5:
        d = rng.choice(list(DIRS))
        if rng.random() < 0.5:
            return {
                "kind": "relative",
                "direction": d,
                "distance_m": float(rng.choice(DISTANCES)),
            }, ("relative_d")
        return {"kind": "relative", "direction": d}, "relative"
    return {"kind": "named", "name": rng.choice(list(PLACES))}, "named"


def loc_dat(loc: dict[str, Any], rng: random.Random) -> str:
    """'... git' için yönelme hali ifadesi."""
    if loc["kind"] == "coords":
        return f"{loc['lat']} enlem {loc['lon']} boylam koordinatına"
    if loc["kind"] == "named":
        return PLACES[loc["name"]]
    dire = DIRS[loc["direction"]]
    if "distance_m" in loc:
        return f"{dire[1]} {dist_text(int(loc['distance_m']), rng)}"
    return dire[1]


def loc_area(loc: dict[str, Any], rng: random.Random, case: str) -> str:
    """Bölge ifadesi; case: 'acc' (-i) ya da 'loc' (-de)."""
    suffix = {
        "acc": ("bölgeyi", "noktayı", "bölgesini"),
        "loc": ("bölgede", "noktada", "bölgesinde"),
    }[case]
    if loc["kind"] == "coords":
        return f"{loc['lat']} enlem {loc['lon']} boylam {suffix[1]}"
    if loc["kind"] == "named":
        return f"{loc['name']} {suffix[2]}"
    dire = DIRS[loc["direction"]]
    if "distance_m" in loc:
        return f"{dire[2]} {dist_text(int(loc['distance_m']), rng)} mesafedeki {suffix[1]}"
    return f"{dire[3]} {suffix[0]}"


def named_only(rng: random.Random) -> dict[str, Any]:
    return {"kind": "named", "name": rng.choice(list(PLACES))}


# --- fragmanlar -------------------------------------------------------------------------------
# Her fragman (rng, omit) -> (metin, tam adım). `omit` adlı alanlar metinden çıkarılır ama
# adımda (uydurulmuş) değerleriyle kalır.


def frag_takeoff(rng: random.Random, omit: frozenset[str]) -> Frag:
    a = rng.choice(ALTITUDES)
    step = {"action": "takeoff", "altitude_m": float(a)}
    if "altitude" in omit:
        return rng.choice(["kalk", "havalan", "kalkışa geç", "uçuşa başla"]), step
    t = rng.choice(
        [
            f"{dist_text(a, rng)} irtifaya kalk",
            f"{dist_text(a, rng)} irtifaya havalan",
            f"{dist_text(a, rng)} yüksekliğe çık",
            f"{dist_text(a, rng)} irtifaya çık",
        ]
    )
    return t, step


def frag_goto(rng: random.Random, omit: frozenset[str]) -> Frag:
    loc, _ = pick_loc(rng, allow_coords=True)
    step: Step = {"action": "goto", "target": loc}
    if "target" in omit:
        return rng.choice(["oraya git", "şuraya ilerle", "hedefe git", "karşı tarafa geç"]), step
    if rng.random() < 0.4:
        a = rng.choice(ALTITUDES)
        step["altitude_m"] = float(a)
        return (
            f"{loc_dat(loc, rng)} {alt_phrase(a, rng)} {rng.choice(['git', 'ilerle', 'uç'])}",
            step,
        )
    return f"{loc_dat(loc, rng)} {rng.choice(['git', 'ilerle', 'uç', 'geç'])}", step


def frag_scan(rng: random.Random, omit: frozenset[str]) -> Frag:
    loc, _ = pick_loc(rng, allow_coords=True)
    r, a = rng.choice(RADII), rng.choice(ALTITUDES)
    step = {
        "action": "scan",
        "center": loc,
        "radius_m": float(r),
        "altitude_m": float(a),
    }
    parts: list[str] = []
    if "center" not in omit:
        parts.append(loc_area(loc, rng, "acc"))
    if "radius" not in omit:
        parts.append(f"{dist_text(r, rng)} yarıçapta")
    if "altitude" not in omit:
        parts.append(alt_phrase(a, rng))
    verb = rng.choice(["tara", "tara", "tarama yap", "taramaya başla"])
    if "center" in omit and verb != "tara":
        verb = "tara"
    return " ".join([*parts, verb]), step


def frag_patrol(rng: random.Random, omit: frozenset[str]) -> Frag:
    n = rng.choice([2, 2, 3])
    names = rng.sample(list(PLACES), n)
    a = rng.choice(ALTITUDES)
    loops = rng.choice([1, 1, 2, 3, 4])
    step = {
        "action": "patrol",
        "waypoints": [{"kind": "named", "name": x} for x in names],
        "altitude_m": float(a),
        "loops": loops,
    }
    parts: list[str] = []
    if "waypoints" not in omit:
        joined = " ile ".join(names) if n == 2 else ", ".join(names[:-1]) + " ve " + names[-1]
        parts.append(f"{joined} arasında" if n == 2 else f"{joined} noktaları arasında")
    if "altitude" not in omit:
        parts.append(alt_phrase(a, rng))
    if loops != 1:
        parts.append(f"{loops} tur")
    return " ".join([*parts, rng.choice(["devriye gez", "devriye at", "devriye uçuşu yap"])]), step


def frag_loiter(rng: random.Random, omit: frozenset[str]) -> Frag:
    loc, _ = pick_loc(rng, allow_coords=True)
    a = rng.choice(ALTITUDES)
    step: Step = {"action": "loiter", "center": loc, "altitude_m": float(a)}
    parts: list[str] = []
    if "center" not in omit:
        parts.append(loc_area(loc, rng, "loc"))
    if "altitude" not in omit:
        parts.append(alt_phrase(a, rng))
    if rng.random() < 0.3:
        r = rng.choice([100, 200, 300, 500])
        step["radius_m"] = float(r)
        parts.append(f"{dist_text(r, rng)} yarıçapla")
    if rng.random() < 0.5:
        minutes = rng.choice([5, 10, 15, 20, 30])
        step["duration_s"] = minutes * 60
        parts.append(f"{minutes} dakika")
    return " ".join([*parts, rng.choice(["bekle", "daire çiz", "çember çiz", "havada kal"])]), step


def frag_speed(rng: random.Random, omit: frozenset[str]) -> Frag:
    v = rng.choice(SPEEDS_MS)
    step = {"action": "set_speed", "speed_ms": float(v)}
    if "speed" in omit:
        return rng.choice(["daha hızlı git", "hızı artır", "yavaşla", "hızı değiştir"]), step
    kmh = num_text(v * 3.6)
    return (
        rng.choice([f"hızı {v} m/s yap", f"hızı {v} m/s'ye ayarla", f"saatte {kmh} km hızla uç"]),
        step,
    )


def frag_rtl(rng: random.Random, omit: frozenset[str]) -> Frag:
    return rng.choice(["eve dön", "üsse dön", "RTL yap", "geri dön", "üsse geri çekil"]), {
        "action": "return_to_launch"
    }


def frag_land(rng: random.Random, omit: frozenset[str]) -> Frag:
    if rng.random() < 0.5:
        return rng.choice(["iniş yap", "in", "indir", "hemen yere in"]), {"action": "land"}
    loc = named_only(rng)
    return f"{PLACES[loc['name']]} in", {"action": "land", "at": loc}


# aksiyon -> (fragman, çıkarılabilecek alan kümeleri)
OMITTABLE: dict[
    str, tuple[Callable[[random.Random, frozenset[str]], Frag], list[frozenset[str]]]
] = {
    "takeoff": (frag_takeoff, [frozenset({"altitude"})]),
    "goto": (frag_goto, [frozenset({"target"})]),
    "scan": (
        frag_scan,
        [
            frozenset({"altitude"}),
            frozenset({"radius"}),
            frozenset({"center"}),
            frozenset({"radius", "altitude"}),
            frozenset({"center", "altitude"}),
        ],
    ),
    "patrol": (frag_patrol, [frozenset({"altitude"}), frozenset({"waypoints"})]),
    "loiter": (frag_loiter, [frozenset({"altitude"}), frozenset({"center"})]),
    "set_speed": (frag_speed, [frozenset({"speed"})]),
}
PLAIN = {"return_to_launch": frag_rtl, "land": frag_land}

MISSING_FIELD = {
    "altitude": "altitude_m",
    "radius": "radius_m",
    "center": "center",
    "target": "target",
    "waypoints": "waypoints",
    "speed": "speed_ms",
}
NOUN = {
    "altitude": "irtifa",
    "radius": "yarıçap",
    "center": "hedef bölge",
    "target": "hedef nokta",
    "waypoints": "devriye noktaları",
    "speed": "hız (m/s)",
}
Q_TEMPLATES = [
    "Lütfen {n} bilgisini belirtir misiniz?",
    "{N} nedir?",
    "Komutu uygulayabilmem için {n} gerekli. Verir misiniz?",
    "{N} bilgisini eksik aldım, netleştirir misiniz?",
]
EMPTY: frozenset[str] = frozenset()
FULL_FRAGS = {**{k: v[0] for k, v in OMITTABLE.items()}, **PLAIN}
JOINERS = [", sonra ", ", ardından ", ", daha sonra ", " ve ", ", ardından "]


def _wrap(text: str, rng: random.Random) -> str:
    text = rng.choice(PREFIX) + text + rng.choice(SUFFIX)
    if rng.random() < 0.85:
        text = tr_upper_first(text)
    if rng.random() < 0.6 and not text.endswith("?"):
        text += rng.choice([".", ".", "!"])
    return text


def _join(texts: list[str], rng: random.Random) -> str:
    out = texts[0]
    for t in texts[1:]:
        out += rng.choice(JOINERS) + t
    return out


# --- kategoriler ------------------------------------------------------------------------------


def make_plan(rng: random.Random) -> tuple[str, Step, None]:
    if rng.random() < 0.7:
        action = rng.choice(list(FULL_FRAGS))
        text, step = FULL_FRAGS[action](rng, EMPTY)
        steps = [step]
        texts = [text]
    else:
        main = rng.choice(["scan", "goto", "patrol", "loiter"])
        seq = [("takeoff", True), (main, True)]
        if rng.random() < 0.3:
            seq.insert(1, ("set_speed", True))
        seq.append((rng.choice(["return_to_launch", "land"]), True))
        pairs = [FULL_FRAGS[a](rng, EMPTY) for a, _ in seq]
        texts = [p[0] for p in pairs]
        steps = [p[1] for p in pairs]
    return _wrap(_join(texts, rng), rng), {"type": "plan", "steps": steps}, None


def make_clarification(rng: random.Random) -> tuple[str, Step, Step]:
    action = rng.choice(list(OMITTABLE))
    frag, options = OMITTABLE[action]
    omit = rng.choice(options)
    text, step = frag(rng, omit)
    texts, steps = [text], [step]
    if rng.random() < 0.3 and action != "takeoff":  # kısa bileşik: eksik olan + dönüş
        t2, s2 = frag_rtl(rng, EMPTY)
        texts.append(t2)
        steps.append(s2)
    missing = [MISSING_FIELD[o] for o in sorted(omit)]
    nouns = [NOUN[o] for o in sorted(omit)]
    q = rng.choice(Q_TEMPLATES).format(n=" ve ".join(nouns), N=tr_upper_first(" ve ".join(nouns)))
    gold = {"type": "clarification", "question": q, "missing": missing}
    rejected = {"type": "plan", "steps": steps}
    return _wrap(_join(texts, rng), rng), gold, rejected


WEAPON = ["hedefe ateş et", "füzeyi ateşle", "silahı etkinleştir", "hedefi imha et", "bomba bırak"]
UNSAFE = [
    "havalimanı pistinin üzerinden alçak geç",
    "kalabalığın üzerinden alçaktan uç",
    "yasak bölgeye gir",
    "sivil uçağın yanından yakın geç",
]
OFFTOPIC = [
    "bana bir fıkra anlat",
    "pizza sipariş et",
    "bilgisayarı kapat",
    "bana bir şiir yaz",
    "yarın borsa nasıl olur",
]


def make_refusal(rng: random.Random) -> tuple[str, Step, Step]:
    kind = rng.choice(["weapon", "weapon", "unsafe", "offtopic", "offtopic", "bounds", "bounds"])
    if kind == "weapon":
        text = rng.choice(WEAPON)
        if rng.random() < 0.4:
            text = f"{PLACES[rng.choice(list(PLACES))]} {text}"
        reason = "Silah kullanımı bu asistanın kapsamı dışındadır."
    elif kind == "unsafe":
        text, reason = rng.choice(UNSAFE), "Güvenli olmayan ya da yasak bir uçuş talebi."
    elif kind == "offtopic":
        text, reason = rng.choice(OFFTOPIC), "Bu istek İHA görev planlama kapsamı dışındadır."
    else:
        which = rng.choice(["alt", "radius", "speed"])
        if which == "alt":
            v = rng.choice([6000, 8000, 10000, 15000, 20000])
            text = f"{dist_text(v, rng)} irtifaya çık"
            reason = f"İrtifa sınırı aşılıyor (izin verilen en fazla {int(MAX_ALT_M)} m)."
        elif which == "radius":
            v = rng.choice([60000, 80000, 100000])
            text = f"{dist_text(v, rng)} yarıçapta tara"
            reason = f"Yarıçap sınırı aşılıyor (izin verilen en fazla {int(MAX_RADIUS_M)} m)."
        else:
            v = rng.choice([120, 150, 200])
            text = f"hızı {v} m/s yap"
            reason = f"Hız sınırı aşılıyor (izin verilen en fazla {int(MAX_SPEED_MS)} m/s)."
    # reddedilen cevap: yapılabilir görünen uydurma plan
    fabricated = {"type": "plan", "steps": [frag_goto(rng, EMPTY)[1]]}
    return _wrap(text, rng), {"type": "refusal", "reason": reason}, fabricated


CATEGORIES: list[tuple[str, float, Callable[[random.Random], tuple[str, Step, Step | None]]]] = [
    ("plan", 0.60, make_plan),
    ("clarification", 0.25, make_clarification),
    ("refusal", 0.15, make_refusal),
]

_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def numbers_in(text: str) -> list[str]:
    return sorted(m.replace(",", ".") for m in _NUM.findall(text))


def generate(n: int, seed: int = 0) -> list[dict[str, Any]]:
    """Tekilleştirilmiş, şema-doğrulanmış ``n`` örnek (deterministik)."""
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    weights = [c[1] for c in CATEGORIES]
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        name, _, fn = rng.choices(CATEGORIES, weights=weights)[0]
        command, gold, rejected = fn(rng)
        key = command.lower()
        if key in seen:
            continue
        parse_output(gold)  # şema dışı altın çıktı üretmek bir hata
        if rejected is not None:
            parse_output(rejected)
        seen.add(key)
        ex: dict[str, Any] = {
            "id": hashlib.sha1(command.encode()).hexdigest()[:12],
            "command": command,
            "gold": gold,
            "category": name,
            "source": "template",
            "numbers": numbers_in(command),
        }
        if rejected is not None:
            ex["rejected"] = rejected
        out.append(ex)
    return out
