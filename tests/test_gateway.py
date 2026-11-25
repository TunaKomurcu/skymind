"""A/B ağ geçidi: atama, istatistik ve vekil davranışı (sahte arka uçlarla, ağsız)."""

import httpx
import pytest
from fastapi.testclient import TestClient

from skymind.serve.gateway import (
    Variant,
    assign,
    create_app,
    parse_variant,
    two_proportion_p,
)

A = Variant("a", "http://a", 1)
B = Variant("b", "http://b", 1)


def test_parse_variant() -> None:
    assert parse_variant("sft=http://127.0.0.1:8091@50") == Variant(
        "sft", "http://127.0.0.1:8091", 50
    )
    assert parse_variant("x=http://h:1/") == Variant("x", "http://h:1", 1)
    for bad in ("x", "=http://h", "x=", "x=http://h@0", "x=http://h@abc"):
        with pytest.raises(ValueError):
            parse_variant(bad)


def test_assign_is_deterministic_and_follows_weights() -> None:
    vs = [Variant("a", "u", 80), Variant("b", "u", 20)]
    first = [assign(f"komut {i}", vs).name for i in range(4000)]
    assert first == [assign(f"komut {i}", vs).name for i in range(4000)]
    share_a = first.count("a") / len(first)
    assert 0.77 < share_a < 0.83


def test_two_proportion_p_known_values() -> None:
    assert two_proportion_p(50, 100, 50, 100) == pytest.approx(1.0)
    assert two_proportion_p(0, 0, 5, 10) is None
    assert two_proportion_p(10, 10, 10, 10) == 1.0  # varyans sıfır
    # 90/100 vs 70/100: z = 3.48 civarı -> p ~ 0.0005
    assert two_proportion_p(90, 100, 70, 100) == pytest.approx(0.0005, abs=2e-4)


def backend(valid: bool, status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(status, json={"error": "kötü"})
        t = "plan" if valid else None
        body = {"valid": valid, "type": t, "latency_ms": 1, "plan": None, "raw": "{}"}
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


def routed_client(by_host: dict[str, httpx.MockTransport]) -> TestClient:
    def route(request: httpx.Request) -> httpx.Response:
        return by_host[request.url.host].handle_request(request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(route))
    return TestClient(create_app([A, B], client=http))


def test_proxy_adds_variant_and_summary_compares() -> None:
    c = routed_client({"a": backend(True), "b": backend(False)})
    seen: dict[str, int] = {}
    for i in range(400):
        r = c.post("/v1/plan", json={"command": f"komut {i}"})
        assert r.status_code == 200
        assert r.json()["variant"] == r.headers["x-variant"]
        seen[r.headers["x-variant"]] = seen.get(r.headers["x-variant"], 0) + 1
    assert set(seen) == {"a", "b"} and min(seen.values()) > 120
    s = c.get("/ab/summary").json()
    assert s["variants"]["a"]["valid_rate"] == 1.0
    assert s["variants"]["b"]["valid_rate"] == 0.0
    assert s["variants"]["b"]["types"] == {"invalid": seen["b"]}
    cmp = s["comparisons"][0]
    assert cmp["control"] == "a" and cmp["treatment"] == "b"
    assert cmp["significant_at_0.05"] is True


def test_same_command_sticks_to_same_variant_and_key_header_overrides() -> None:
    c = routed_client({"a": backend(True), "b": backend(True)})
    picks = {c.post("/v1/plan", json={"command": "aynı"}).headers["x-variant"] for _ in range(10)}
    assert len(picks) == 1
    k = {
        c.post("/v1/plan", json={"command": f"x{i}"}, headers={"X-Request-Key": "sabit"}).headers[
            "x-variant"
        ]
        for i in range(10)
    }
    assert len(k) == 1


def test_upstream_failure_returns_502_and_is_counted() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("kapalı")

    c = routed_client({"a": httpx.MockTransport(boom), "b": httpx.MockTransport(boom)})
    r = c.post("/v1/plan", json={"command": "x"})
    assert r.status_code == 502 and "variant" in r.json()
    s = c.get("/ab/summary").json()
    assert sum(v["errors"] for v in s["variants"].values()) == 1
    assert 'status="upstream_error"' in c.get("/metrics").text


def test_backend_400_passes_through_without_polluting_quality_stats() -> None:
    c = routed_client({"a": backend(True, 400), "b": backend(True, 400)})
    r = c.post("/v1/plan", json={"command": "x"})
    assert r.status_code == 400
    s = c.get("/ab/summary").json()
    assert all(v["requests"] == 0 for v in s["variants"].values())


def test_metrics_and_health() -> None:
    c = routed_client({"a": backend(True), "b": backend(True)})
    c.post("/v1/plan", json={"command": "x"})
    text = c.get("/metrics").text
    assert "skymind_gateway_requests_total" in text and "skymind_gateway_output_total" in text
    assert "skymind_gateway_request_duration_seconds_bucket" in text
    assert c.get("/healthz").json()["variants"] == ["a", "b"]


def test_create_app_rejects_bad_config() -> None:
    with pytest.raises(ValueError):
        create_app([])
    with pytest.raises(ValueError):
        create_app([A, Variant("a", "http://x", 1)])
