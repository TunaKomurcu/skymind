"""Python baseline servisi: sahte motorla davranış ve yük testi aracı."""

import asyncio
import socket
import threading
import time
from collections.abc import Iterator

import pytest
import uvicorn
from fastapi.testclient import TestClient

from skymind.serve.app import create_app
from skymind.serve.engines import GenerationResult
from skymind.serve.loadtest import run_level
from skymind.serve.service import PlannerService, validate_text

PLAN = '{"type":"plan","steps":[{"action":"takeoff","altitude_m":100}]}'


class FakeEngine:
    name = "fake"

    def __init__(self, text: str = PLAN, delay: float = 0.0) -> None:
        self.text = text
        self.delay = delay
        self.calls = 0

    def generate(self, system: str, command: str) -> GenerationResult:
        self.calls += 1
        time.sleep(self.delay)
        return GenerationResult(text=self.text, prompt_tokens=10, new_tokens=7)


def client(engine: FakeEngine) -> TestClient:
    return TestClient(create_app(PlannerService(engine, "sys")))


def test_validate_text_accepts_and_rejects() -> None:
    plan, kind, err = validate_text(PLAN)
    assert kind == "plan" and err is None and plan is not None
    assert validate_text("kod değil")[2] == "geçerli JSON değil"
    _, kind, err = validate_text('{"type":"plan","steps":[{"action":"takeoff","altitude_m":9999}]}')
    assert kind is None and err and "altitude_m" in err


def test_plan_ok_response_shape() -> None:
    r = client(FakeEngine()).post("/v1/plan", json={"command": "kalk"})
    body = r.json()
    assert r.status_code == 200
    assert body["valid"] is True and body["type"] == "plan" and body["plan"]["type"] == "plan"
    assert body["validation_error"] is None and body["engine"] == "fake"
    assert body["usage"]["new_tokens"] == 7 and body["latency_ms"] >= 0


def test_invalid_model_output_is_flagged_not_passed_as_plan() -> None:
    bad = '{"type":"plan","steps":[{"action":"loiter","altitude_m":100}]}'
    body = client(FakeEngine(bad)).post("/v1/plan", json={"command": "dön"}).json()
    assert body["valid"] is False and body["plan"] is None and body["type"] is None
    assert body["raw"] == bad and "center" in body["validation_error"]


@pytest.mark.parametrize(
    "payload", [{}, {"command": 5}, {"command": ""}, {"command": "a" * 2001}, [1]]
)
def test_bad_requests_return_400_and_skip_engine(payload: object) -> None:
    engine = FakeEngine()
    r = client(engine).post("/v1/plan", json=payload)
    assert r.status_code == 400 and "error" in r.json()
    assert engine.calls == 0


def test_non_json_body_returns_400() -> None:
    r = client(FakeEngine()).post("/v1/plan", content=b"{bozuk")
    assert r.status_code == 400


def test_engine_exception_returns_500() -> None:
    class Boom(FakeEngine):
        def generate(self, system: str, command: str) -> GenerationResult:
            raise RuntimeError("patladı")

    r = client(Boom()).post("/v1/plan", json={"command": "x"})
    assert r.status_code == 500 and "patladı" in r.json()["error"]


def test_healthz_and_metrics_names_match_cpp_service() -> None:
    c = client(FakeEngine())
    assert c.get("/healthz").json() == {"status": "ok"}
    c.post("/v1/plan", json={"command": "kalk"})
    c.post("/v1/plan", json={})
    text = c.get("/metrics").text
    for needle in (
        'skymind_requests_total{status="ok"} 1.0',
        'skymind_requests_total{status="bad_request"} 1.0',
        'skymind_output_total{type="plan"} 1.0',
        "skymind_prompt_tokens_total 10.0",
        "skymind_generated_tokens_total 7.0",
        "skymind_requests_in_flight 0.0",
        'skymind_request_duration_seconds_bucket{le="0.05"}',
        "skymind_request_duration_seconds_count 1.0",
        'skymind_model_info{engine="fake"} 1.0',
    ):
        assert needle in text, needle


@pytest.fixture
def live_url() -> Iterator[str]:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    app = create_app(PlannerService(FakeEngine(delay=0.02), "sys"))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    t.join(timeout=5)


def test_loadtest_run_level_against_live_service(live_url: str) -> None:
    res = asyncio.run(run_level(live_url, ["a", "b", "c"], concurrency=4, total=20, timeout=10))
    assert res["ok"] == 20 and res["errors"] == 0 and res["valid_rate"] == 1.0
    assert res["types"] == {"plan": 20}
    lat = res["latency_ms"]
    assert 0 < lat["p50"] <= lat["p95"] <= lat["p99"] <= lat["max"]
    # tek bağlam sıralıdır: 4 eşzamanlı istekte işçi sayısı kadar kuyruk bekleme olmalı
    assert lat["max"] >= 2 * 20  # en az iki istek süresi kadar bekleyen var


def test_demo_planner_section_reads_the_real_service_response_shape(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json as _json

    import httpx

    from skymind import demo

    svc = PlannerService(FakeEngine(), "sys")

    def handler(request: httpx.Request) -> httpx.Response:
        status, body = svc.plan(_json.loads(request.content))
        return httpx.Response(status, json=body)

    bad = demo.show_planner("http://x", 5.0, transport=httpx.MockTransport(handler))
    out = capsys.readouterr().out
    assert bad == 0 and "[plan]" in out and "takeoff" in out and "7 token" in out
    # geçersiz model çıktısı demoda saklanmaz
    svc2 = PlannerService(FakeEngine("saçma"), "sys")

    def handler2(request: httpx.Request) -> httpx.Response:
        status, body = svc2.plan(_json.loads(request.content))
        return httpx.Response(status, json=body)

    assert demo.show_planner("http://x", 5.0, transport=httpx.MockTransport(handler2)) == len(
        demo.COMMANDS
    )
