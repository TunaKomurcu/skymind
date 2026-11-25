"""Zayıf (weak) NER etiketleri: plandaki altın değerleri komut metnine hizalar.

Etiketler: ALT, RADIUS, DIST, SPEED, DUR, COORD, PLACE, DIR. Elle etiketlenmiş DEĞİLDİR:
etiketler, plan değerlerinin metindeki sayılara/yer adlarına bağlam kurallarıyla hizalanmasından
gelir. Hizalanamayan örnekler (bir altın değer metinde tam olarak bulunamadı) atılır; kapsama
oranı ``coverage`` ile raporlanır. Bu yüzden F1 "zayıf etiketlere karşı" ölçülür.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

LABELS = ["O", "ALT", "RADIUS", "DIST", "SPEED", "DUR", "COORD", "PLACE", "DIR"]
ENTITY_TYPES = LABELS[1:]
BIO = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]

DIR_FORMS = {
    "N": "kuzey",
    "NE": "kuzeydoğu",
    "E": "doğu",
    "SE": "güneydoğu",
    "S": "güney",
    "SW": "güneybatı",
    "W": "batı",
    "NW": "kuzeybatı",
}
_PUNCT = ".,;:!?\"'()"
_NUM = re.compile(r"^\d+(?:[.,]\d+)?$")
_ATTACHED = re.compile(r"^(\d+(?:[.,]\d+)?)(km|m)$", re.I)
_COORD_PAIR = re.compile(r"^\d{2}[.,]\d{2}'\w+$")

Word = tuple[str, int, int]  # metin, başlangıç, bitiş (karakter)


def words(text: str) -> list[Word]:
    out: list[Word] = []
    for m in re.finditer(r"\S+", text):
        w = m.group(0)
        lead = len(w) - len(w.lstrip(_PUNCT))
        core = w.strip(_PUNCT)
        if core:
            out.append((core, m.start() + lead, m.start() + lead + len(core)))
    return out


def _tr_lower(s: str) -> str:
    return s.replace("İ", "i").replace("I", "ı").lower()


def _num(s: str) -> float:
    return float(s.replace(",", "."))


def _unit(w: str) -> tuple[str, float] | None:
    """Birim sözcüğü -> (sınıf, çarpan)."""
    w = _tr_lower(w)
    if w.startswith(("m/s", "metre/s")):
        return "speed", 1.0
    if w.startswith(("kilometre", "km")):
        return "length", 1000.0
    if w.startswith("metre") or w == "m" or w.startswith("m'"):
        return "length", 1.0
    if w.startswith("dakika"):
        return "time", 60.0
    if w.startswith("saniye"):
        return "time", 1.0
    return None


def _length_role(ws: list[Word], i_unit: int, i_num: int) -> str | None:
    nxt = [_tr_lower(w[0]) for w in ws[i_unit + 1 : i_unit + 3]]
    prev = _tr_lower(ws[i_num - 1][0]) if i_num > 0 else ""
    if any(w.startswith("yarıçap") for w in nxt):
        return "RADIUS"
    if any(w.startswith(("irtifa", "yüksek")) for w in nxt):
        return "ALT"
    if any(w.startswith("mesafe") for w in nxt) or any(
        prev.startswith(d) for d in DIR_FORMS.values()
    ):
        return "DIST"
    return None


def find_numeric(text: str) -> list[dict[str, Any]]:
    """Metindeki sayısal bahisler: {span:(i0,i1) kelime aralığı, cls, value, role|None}."""
    ws = words(text)
    out: list[dict[str, Any]] = []
    for i, (w, _s, _e) in enumerate(ws):
        attached = _ATTACHED.match(w)
        if attached:  # "500m", "1,5km"
            mult = 1000.0 if attached[2] == "km" else 1.0
            out.append(
                {
                    "span": (i, i),
                    "cls": "length",
                    "value": _num(attached[1]) * mult,
                    "role": _length_role(ws, i, i),
                }
            )
            continue
        if not _NUM.match(w):
            if _COORD_PAIR.match(w):  # "41.43'e" gibi ekli koordinat
                out.append(
                    {
                        "span": (i, i),
                        "cls": "coord",
                        "value": _num(w.split("'")[0]),
                        "role": "COORD",
                    }
                )
            continue
        nxt = ws[i + 1][0] if i + 1 < len(ws) else ""
        nxt2 = _tr_lower(nxt)
        if nxt2.startswith(("enlem", "boylam")):
            out.append({"span": (i, i), "cls": "coord", "value": _num(w), "role": "COORD"})
            continue
        u = _unit(nxt)
        if u is None:
            if re.match(r"^\d{2}[.,]\d{2}$", w):  # birimsiz iki ondalıklı sayı: koordinat çifti
                out.append({"span": (i, i), "cls": "coord", "value": _num(w), "role": "COORD"})
            continue
        cls, mult = u
        value = _num(w) * mult
        if cls == "length" and i > 0 and _tr_lower(ws[i - 1][0]) == "saatte" and mult == 1000.0:
            out.append(
                {"span": (i - 1, i + 1), "cls": "speed", "value": _num(w) / 3.6, "role": "SPEED"}
            )
            continue
        role = {"speed": "SPEED", "time": "DUR"}.get(cls) or _length_role(ws, i + 1, i)
        out.append({"span": (i, i + 1), "cls": cls, "value": value, "role": role})
    return out


def _walk_locs(step: dict[str, Any]) -> list[dict[str, Any]]:
    locs: list[dict[str, Any]] = []
    for key in ("target", "center", "at"):
        if isinstance(step.get(key), dict):
            locs.append(step[key])
    locs += [w for w in step.get("waypoints", []) if isinstance(w, dict)]
    return locs


def gold_entities(plan: dict[str, Any]) -> dict[str, list[Any]]:
    g: dict[str, list[Any]] = {t: [] for t in ENTITY_TYPES}
    for st in plan.get("steps", []):
        if "altitude_m" in st:
            g["ALT"].append(float(st["altitude_m"]))
        if st.get("action") in ("scan", "loiter") and "radius_m" in st:
            g["RADIUS"].append(float(st["radius_m"]))
        if "speed_ms" in st:
            g["SPEED"].append(float(st["speed_ms"]))
        if "duration_s" in st:
            g["DUR"].append(float(st["duration_s"]))
        for loc in _walk_locs(st):
            if loc["kind"] == "coords":
                g["COORD"] += [float(loc["lat"]), float(loc["lon"])]
            elif loc["kind"] == "named":
                g["PLACE"].append(loc["name"])
            else:
                g["DIR"].append(loc["direction"])
                if "distance_m" in loc:
                    g["DIST"].append(float(loc["distance_m"]))
    return g


def _near(a: float, b: float) -> bool:
    return abs(a - b) < 1e-6 * max(1.0, abs(b)) or abs(a - b) < 0.011


def _assign_unknown(mentions: list[dict[str, Any]], gold: dict[str, list[Any]]) -> None:
    """Bağlamı olmayan uzunluk bahislerine, hâlâ eşleşmemiş altın değerlerden rol ata."""
    left = {t: list(gold[t]) for t in ("ALT", "RADIUS", "DIST")}
    for m in mentions:
        if m["role"] in left:
            for v in left[m["role"]]:
                if _near(m["value"], v):
                    left[m["role"]].remove(v)
                    break
    for m in mentions:
        if m["role"] is None and m["cls"] == "length":
            for t in ("ALT", "RADIUS", "DIST"):
                hit = next((v for v in left[t] if _near(m["value"], v)), None)
                if hit is not None:
                    m["role"] = t
                    left[t].remove(hit)
                    break


def _match_place(
    ws: list[Word], name: str, taken: tuple[tuple[int, int], ...] = ()
) -> tuple[int, int] | None:
    """Yer adının ilk, ``taken`` ile örtüşmeyen geçişi. "X noktası" kısaca "X" diye de geçer."""
    full = [_tr_lower(p) for p in name.split()]
    variants = [full] + ([full[:1]] if full[-1].startswith("nokta") else [])
    for parts in variants:
        for i in range(len(ws) - len(parts) + 1):
            if any(i <= b and i + len(parts) - 1 >= a for a, b in taken):
                continue
            ok = True
            for k, p in enumerate(parts):
                w = _tr_lower(ws[i + k][0])
                stem = p if k < len(parts) - 1 else p[: max(3, len(p) - 1)]
                if not w.startswith(stem):
                    ok = False
                    break
            if ok:
                return i, i + len(parts) - 1
    return None


def _word_spans(
    ws: list[Word], plan: dict[str, Any], mentions: list[dict[str, Any]]
) -> list[tuple[int, int, str]]:
    """Bütün varlıkların (kelime i0, i1, tür) listesi ya da hizalanamazsa ValueError."""
    gold = gold_entities(plan)
    _assign_unknown(mentions, gold)
    spans: list[tuple[int, int, str]] = []
    for t in ("ALT", "RADIUS", "DIST", "SPEED", "DUR", "COORD"):
        mine = [m for m in mentions if m["role"] == t]
        want = list(gold[t])
        if len(mine) != len(want):
            raise ValueError(f"{t}: {len(mine)} bahis, {len(want)} altın değer")
        for m in mine:
            hit = next((v for v in want if _near(m["value"], v)), None)
            if hit is None:
                raise ValueError(f"{t}: {m['value']} altın değerler arasında yok")
            want.remove(hit)
            spans.append((*m["span"], t))
    taken: tuple[tuple[int, int], ...] = ()
    for name in gold["PLACE"]:
        sp = _match_place(ws, name, taken)
        if sp is None:
            raise ValueError(f"PLACE: {name!r} metinde yok")
        taken += (sp,)
        spans.append((*sp, "PLACE"))
    for i, (w, _s, _e) in enumerate(ws):
        lw = _tr_lower(w)
        if any(lw.startswith(f) for f in DIR_FORMS.values()):
            spans.append((i, i, "DIR"))
    n_dir = sum(1 for s in spans if s[2] == "DIR")
    if n_dir != len(gold["DIR"]):
        raise ValueError(f"DIR: {n_dir} bahis, {len(gold['DIR'])} altın")
    spans.sort()
    for a, b in zip(spans, spans[1:], strict=False):
        if b[0] <= a[1]:
            raise ValueError("örtüşen varlıklar")
    return spans


def bio_tags(n_words: int, spans: list[tuple[int, int, str]]) -> list[str]:
    tags = ["O"] * n_words
    for i0, i1, t in spans:
        tags[i0] = f"B-{t}"
        for i in range(i0 + 1, i1 + 1):
            tags[i] = f"I-{t}"
    return tags


def align_example(command: str, plan: dict[str, Any]) -> tuple[list[str], list[str]] | None:
    """(kelimeler, BIO etiketleri) ya da hizalanamazsa None."""
    ws = words(command)
    try:
        spans = _word_spans(ws, plan, find_numeric(command))
    except ValueError:
        return None
    return [w[0] for w in ws], bio_tags(len(ws), spans)


def rule_tagger(command: str) -> list[str]:
    """Altın plan KULLANMAYAN kural tabanlı taban çizgisi (yalnızca bağlam kuralları)."""
    ws = words(command)
    spans: list[tuple[int, int, str]] = []
    for m in find_numeric(command):
        role = m["role"] or ("ALT" if m["cls"] == "length" else None)
        if role:
            spans.append((*m["span"], role))
    for i, (w, _s, _e) in enumerate(ws):
        if any(_tr_lower(w).startswith(f) for f in DIR_FORMS.values()):
            spans.append((i, i, "DIR"))
    from skymind.data.generator import PLACES

    for name in PLACES:
        sp = _match_place(ws, name)
        if sp:
            spans.append((*sp, "PLACE"))
    spans.sort()
    clean: list[tuple[int, int, str]] = []
    for s in spans:
        if not clean or s[0] > clean[-1][1]:
            clean.append(s)
    return bio_tags(len(ws), clean)


def coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    plans = [r for r in rows if r["gold"]["type"] == "plan"]
    ok = [align_example(r["command"], r["gold"]) for r in plans]
    return {
        "plans": len(plans),
        "aligned": sum(x is not None for x in ok),
        "coverage": round(sum(x is not None for x in ok) / max(1, len(plans)), 3),
        "entities": dict(Counter(t[2:] for x in ok if x for t in x[1] if t.startswith("B-"))),
    }


# --- niyet -------------------------------------------------------------------------------------

INTENTS = [
    "scan",
    "patrol",
    "loiter",
    "goto",
    "land",
    "return_to_launch",
    "set_speed",
    "takeoff",
    "clarification",
    "refusal",
]
_PRIORITY = ["scan", "patrol", "loiter", "goto", "land", "return_to_launch", "set_speed", "takeoff"]


def intent_of(gold: dict[str, Any]) -> str:
    """Plan için en baskın görev fiili (öncelik sırasıyla), diğerleri için tür."""
    if gold["type"] != "plan":
        return str(gold["type"])
    acts = {s["action"] for s in gold["steps"]}
    return next(a for a in _PRIORITY if a in acts)
