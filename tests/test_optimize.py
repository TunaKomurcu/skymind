import json
from pathlib import Path

from skymind.optimize.eval_gguf import percentile, pick_rows


def test_percentile_bounds() -> None:
    xs = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert percentile(xs, 0.0) == 1.0
    assert percentile(xs, 1.0) == 5.0
    assert percentile(xs, 0.5) == 3.0


def test_pick_rows_is_equally_spaced_and_deterministic(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    p.write_text(
        "\n".join(json.dumps({"id": i, "command": f"c{i}", "gold": {}}) for i in range(100)),
        encoding="utf-8",
    )
    a = [r["id"] for r in pick_rows(p, 10)]
    assert a == list(range(0, 100, 10))
    assert a == [r["id"] for r in pick_rows(p, 10)]
    assert len(pick_rows(p, 0)) == 100
