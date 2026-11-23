from pathlib import Path

import httpx
import pytest

from skymind.llm import BudgetExceeded, OpenRouterClient

MODELS = {
    "data": [
        {"id": "paid/model", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
        {"id": "free/model:free", "pricing": {"prompt": "0", "completion": "0"}},
    ]
}


def make(tmp_path: Path, budget: float, cost: float = 0.01, statuses: list[int] | None = None):  # type: ignore[no-untyped-def]
    calls: list[str] = []
    seq = list(statuses or [200])

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json=MODELS)
        calls.append(req.url.path)
        status = seq.pop(0) if len(seq) > 1 else seq[0]
        if status != 200:
            return httpx.Response(status, json={})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "merhaba"}}], "usage": {"cost": cost}},
        )

    client = OpenRouterClient(
        api_key="k",
        budget_usd=budget,
        state_dir=tmp_path,
        transport=httpx.MockTransport(handler),
        sleep=lambda _s: None,
    )
    return client, calls


MSG = [{"role": "user", "content": "selam"}]


def test_cost_recorded_and_cached(tmp_path: Path) -> None:
    c, calls = make(tmp_path, budget=1.0, cost=0.02)
    a = c.chat("paid/model", MSG, max_tokens=100)
    b = c.chat("paid/model", MSG, max_tokens=100)
    assert (a.cached, b.cached) == (False, True)
    assert len(calls) == 1
    assert c.ledger.spent() == pytest.approx(0.02)


def test_request_blocked_when_worst_case_exceeds_budget(tmp_path: Path) -> None:
    c, calls = make(tmp_path, budget=0.0001)
    with pytest.raises(BudgetExceeded):
        c.chat("paid/model", MSG, max_tokens=4000)  # 4000*2e-6 = 0.008 > 0.0001
    assert calls == []


def test_unknown_model_is_blocked_not_free(tmp_path: Path) -> None:
    c, calls = make(tmp_path, budget=4.0)
    with pytest.raises(BudgetExceeded):
        c.chat("bilinmeyen/model", MSG)
    assert calls == []


def test_free_model_costs_nothing_even_with_tiny_budget(tmp_path: Path) -> None:
    c, _ = make(tmp_path, budget=0.0, cost=0.0)
    assert c.chat("free/model:free", MSG, max_tokens=4000).text == "merhaba"


def test_spend_persists_across_clients(tmp_path: Path) -> None:
    c1, _ = make(tmp_path, budget=1.0, cost=0.4)
    c1.chat("paid/model", MSG, max_tokens=10)
    c2, _ = make(tmp_path, budget=1.0)
    assert c2.remaining() == pytest.approx(0.6)


def test_retries_on_429(tmp_path: Path) -> None:
    c, calls = make(tmp_path, budget=1.0, statuses=[429, 429, 200])
    assert c.chat("paid/model", MSG, max_tokens=10).text == "merhaba"
    assert len(calls) == 3
