"""C++ servisi için Python'dan üretilen sabit dosyalar.

* ``cpp/configs/system_short.txt``  – eğitimde kullanılan kısa sistem istemi (tek kaynak: prompt.py)
* ``cpp/tests/data/cases.jsonl``    – şema eşlik vakaları; ``valid`` Pydantic'in kararıdır

    python scripts/export_cpp_fixtures.py
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from skymind.planner.prompt import SYSTEM_SHORT, load_split
from skymind.schemas import parse_output

ROOT = Path(__file__).resolve().parents[1]
OUT_SYSTEM = ROOT / "cpp" / "configs" / "system_short.txt"
OUT_CASES = ROOT / "cpp" / "tests" / "data" / "cases.jsonl"


def is_valid(doc: Any) -> bool:
    try:
        parse_output(doc)
    except ValidationError:
        return False
    return True


def walk(node: Any, path: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    """Tüm (iç içe) yaprak ve nesne yollarını döndürür."""
    out: list[tuple[Any, ...]] = [path]
    if isinstance(node, dict):
        for k, v in node.items():
            out += walk(v, (*path, k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out += walk(v, (*path, i))
    return out


def get(doc: Any, path: tuple[Any, ...]) -> Any:
    for p in path:
        doc = doc[p]
    return doc


def mutate(doc: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    """Rastgele tek bir bozulma uygular."""
    d = copy.deepcopy(doc)
    paths = [p for p in walk(d) if p]
    if not paths:
        return None
    path = rng.choice(paths)
    parent, key = get(d, path[:-1]), path[-1]
    value = parent[key]
    kind = rng.choice(["extra", "drop", "bound", "type", "enum"])
    if kind == "extra" and isinstance(value, dict):
        value["bogus"] = 1
    elif kind == "drop" and isinstance(parent, dict):
        del parent[key]
    elif kind == "bound" and isinstance(value, (int, float)) and not isinstance(value, bool):
        parent[key] = rng.choice([0, -5, 9.99, 5000.5, 60000, 1e9, 3.5])
    elif kind == "type" and isinstance(value, (int, float, str)):
        parent[key] = rng.choice(["abc", [], {}, None])
    elif kind == "enum" and isinstance(value, str):
        parent[key] = value + "_x"
    else:
        return None
    return d


EDGE: list[dict[str, Any]] = [
    {"type": "plan", "steps": [{"action": "takeoff", "altitude_m": 10.0}]},
    {"type": "plan", "steps": [{"action": "takeoff", "altitude_m": 5000.0}]},
    {"type": "plan", "steps": [{"action": "takeoff", "altitude_m": 5000.1}]},
    {"type": "plan", "steps": [{"action": "takeoff", "altitude_m": 9.9}]},
    {"type": "plan", "steps": [{"action": "takeoff", "altitude_m": 100}]},
    {"type": "plan", "steps": []},
    {"type": "plan", "steps": [{"action": "return_to_launch"}] * 20},
    {"type": "plan", "steps": [{"action": "return_to_launch"}] * 21},
    {"type": "plan", "steps": [{"action": "return_to_launch", "x": 1}]},
    {"type": "plan", "steps": [{"action": "land", "at": None}]},
    {"type": "plan", "steps": [{"action": "land"}]},
    {"type": "plan", "steps": [{"action": "fly"}]},
    {"type": "plan", "steps": [{"altitude_m": 100}]},
    {"type": "plan", "steps": [{"action": "set_speed", "speed_ms": 5}]},
    {"type": "plan", "steps": [{"action": "set_speed", "speed_ms": 100.1}]},
    {
        "type": "plan",
        "steps": [
            {
                "action": "goto",
                "target": {"kind": "relative", "direction": "N", "distance_m": None},
            }
        ],
    },
    {
        "type": "plan",
        "steps": [{"action": "goto", "target": {"kind": "relative", "direction": "N"}}],
    },
    {
        "type": "plan",
        "steps": [
            {
                "action": "goto",
                "target": {"kind": "relative", "direction": "N", "distance_m": 0},
            }
        ],
    },
    {
        "type": "plan",
        "steps": [{"action": "goto", "target": {"kind": "relative", "direction": "UP"}}],
    },
    {
        "type": "plan",
        "steps": [{"action": "goto", "target": {"kind": "coords", "lat": 91, "lon": 0}}],
    },
    {
        "type": "plan",
        "steps": [{"action": "goto", "target": {"kind": "coords", "lat": 40.5, "lon": 32.1}}],
    },
    {"type": "plan", "steps": [{"action": "goto", "target": {"kind": "named", "name": ""}}]},
    {
        "type": "plan",
        "steps": [{"action": "goto", "target": {"kind": "named", "name": "ç" * 80}}],
    },
    {
        "type": "plan",
        "steps": [{"action": "goto", "target": {"kind": "named", "name": "ç" * 81}}],
    },
    {
        "type": "plan",
        "steps": [
            {
                "action": "patrol",
                "waypoints": [{"kind": "named", "name": "a"}],
                "altitude_m": 100,
            }
        ],
    },
    {
        "type": "plan",
        "steps": [
            {
                "action": "patrol",
                "waypoints": [{"kind": "named", "name": "a"}, {"kind": "named", "name": "b"}],
                "altitude_m": 100,
                "loops": 3.0,
            }
        ],
    },
    {
        "type": "plan",
        "steps": [
            {
                "action": "patrol",
                "waypoints": [{"kind": "named", "name": "a"}, {"kind": "named", "name": "b"}],
                "altitude_m": 100,
                "loops": 21,
            }
        ],
    },
    {
        "type": "plan",
        "steps": [
            {
                "action": "loiter",
                "center": {"kind": "named", "name": "a"},
                "altitude_m": 100,
                "duration_s": 10.5,
            }
        ],
    },
    {
        "type": "plan",
        "steps": [
            {
                "action": "loiter",
                "center": {"kind": "named", "name": "a"},
                "altitude_m": 100,
                "duration_s": None,
            }
        ],
    },
    {"type": "clarification", "question": "Hangi irtifa?", "missing": ["altitude_m"]},
    {"type": "clarification", "question": "ab", "missing": ["altitude_m"]},
    {"type": "clarification", "question": "Hangi irtifa?", "missing": []},
    {"type": "clarification", "question": "Hangi irtifa?", "missing": ["  "]},
    {"type": "clarification", "question": "Hangi irtifa?", "missing": ["a"] * 6},
    {"type": "clarification", "question": "x" * 301, "missing": ["a"]},
    {"type": "clarification", "question": "Hangi irtifa?", "missing": ["a"], "extra": 1},
    {"type": "refusal", "reason": "Kapsam dışı"},
    {"type": "refusal", "reason": "no"},
    {"type": "refusal"},
    {"type": "unknown"},
    {},
]


def main() -> None:
    OUT_SYSTEM.parent.mkdir(parents=True, exist_ok=True)
    OUT_SYSTEM.write_text(SYSTEM_SHORT + "\n", encoding="utf-8", newline="\n")

    rows = load_split(ROOT / "data" / "processed" / "test.jsonl")
    rng = random.Random(0)
    golds = [r["gold"] for r in rows[:: max(1, len(rows) // 150)]][:150]

    docs: list[Any] = list(golds) + EDGE
    for g in golds:
        for _ in range(3):
            bad = mutate(g, rng) if g["type"] == "plan" else mutate(g, rng)
            if bad is not None:
                docs.append(bad)

    seen: set[str] = set()
    cases: list[dict[str, Any]] = []
    for doc in docs:
        key = json.dumps(doc, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        cases.append({"doc": doc, "valid": is_valid(doc)})

    OUT_CASES.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CASES.open("w", encoding="utf-8", newline="\n") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    n_valid = sum(c["valid"] for c in cases)
    print(f"{len(cases)} vaka ({n_valid} geçerli, {len(cases) - n_valid} geçersiz) -> {OUT_CASES}")


if __name__ == "__main__":
    main()
