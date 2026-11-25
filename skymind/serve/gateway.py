"""A/B ağ geçidi: istekleri ağırlıklara göre model varyantlarına dağıtır, varyant bazında ölçer.

    python -m skymind.serve.gateway --port 8000 \
        --variant sft=http://127.0.0.1:8091@50 --variant dpo=http://127.0.0.1:8092@50

Atama deterministiktir: aynı komut (ya da ``X-Request-Key`` başlığı) her zaman aynı varyanta gider;
böylece yeniden denemeler tutarlı kalır ve iki grup rastgele ama tekrarlanabilir bölünür.
Uçlar: POST /v1/plan (vekil), GET /ab/summary (karşılaştırma + anlamlılık), GET /metrics, /healthz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from skymind.eval.sampling import percentile

BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
KEEP_LATENCIES = 5000


@dataclass(frozen=True)
class Variant:
    name: str
    url: str
    weight: float


def parse_variant(spec: str) -> Variant:
    """``ad=http://host:port@ağırlık`` (ağırlık verilmezse 1)."""
    name, sep, rest = spec.partition("=")
    if not sep or not name or not rest:
        raise ValueError(f"varyant biçimi ad=url[@ağırlık]: {spec!r}")
    url, _, w = rest.rpartition("@") if "@" in rest.rsplit("/", 1)[-1] else (rest, "", "1")
    weight = float(w)
    if weight <= 0 or not math.isfinite(weight):
        raise ValueError(f"ağırlık pozitif olmalı: {spec!r}")
    return Variant(name, url.rstrip("/"), weight)


def assign(key: str, variants: list[Variant]) -> Variant:
    """Anahtarın SHA-256'sından [0,1) noktası; ağırlıklı aralıklara düşen varyant."""
    point = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") / 2**64
    total = sum(v.weight for v in variants)
    acc = 0.0
    for v in variants:
        acc += v.weight / total
        if point < acc:
            return v
    return variants[-1]


def two_proportion_p(x1: int, n1: int, x2: int, n2: int) -> float | None:
    """İki oran farkı için iki yönlü z-testi p değeri (havuzlanmış varyans)."""
    if n1 == 0 or n2 == 0:
        return None
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (x1 / n1 - x2 / n2) / se
    return math.erfc(abs(z) / math.sqrt(2))


class Stats:
    def __init__(self, names: list[str]) -> None:
        self.n = dict.fromkeys(names, 0)
        self.valid = dict.fromkeys(names, 0)
        self.errors = dict.fromkeys(names, 0)
        self.lat: dict[str, deque[float]] = {k: deque(maxlen=KEEP_LATENCIES) for k in names}
        self.types: dict[str, dict[str, int]] = {k: {} for k in names}

    def record(self, name: str, ms: float, valid: bool, otype: str) -> None:
        self.n[name] += 1
        self.valid[name] += valid
        self.lat[name].append(ms)
        self.types[name][otype] = self.types[name].get(otype, 0) + 1

    def summary(self, names: list[str]) -> dict[str, Any]:
        out: dict[str, Any] = {"variants": {}, "comparisons": []}
        for k in names:
            lat = list(self.lat[k])
            n = self.n[k]
            out["variants"][k] = {
                "requests": n,
                "errors": self.errors[k],
                "valid_rate": round(self.valid[k] / n, 4) if n else None,
                "latency_ms": {
                    "p50": round(percentile(lat, 0.5), 1) if lat else None,
                    "p95": round(percentile(lat, 0.95), 1) if lat else None,
                },
                "types": self.types[k],
            }
        control = names[0]
        for k in names[1:]:
            p = two_proportion_p(self.valid[control], self.n[control], self.valid[k], self.n[k])
            out["comparisons"].append(
                {
                    "control": control,
                    "treatment": k,
                    "metric": "valid_rate",
                    "p_value": None if p is None else round(p, 4),
                    "significant_at_0.05": None if p is None else p < 0.05,
                }
            )
        return out


def create_app(
    variants: list[Variant], client: httpx.AsyncClient | None = None, timeout: float = 120.0
) -> FastAPI:
    if not variants:
        raise ValueError("en az bir varyant gerekli")
    names = [v.name for v in variants]
    if len(set(names)) != len(names):
        raise ValueError("varyant adları benzersiz olmalı")
    app = FastAPI(title="SkyMind A/B gateway")
    stats = Stats(names)
    registry = CollectorRegistry()
    req_total = Counter(
        "skymind_gateway_requests", "Ağ geçidi istekleri", ["variant", "status"], registry=registry
    )
    out_total = Counter(
        "skymind_gateway_output", "Varyant çıktı türleri", ["variant", "type"], registry=registry
    )
    dur = Histogram(
        "skymind_gateway_request_duration_seconds",
        "Ağ geçidinde ölçülen uçtan uca süre",
        ["variant"],
        buckets=BUCKETS,
        registry=registry,
    )
    http = client or httpx.AsyncClient(timeout=timeout)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "variants": names}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type="text/plain; version=0.0.4")

    @app.get("/ab/summary")
    def summary() -> dict[str, Any]:
        return stats.summary(names)

    @app.post("/v1/plan")
    async def plan(request: Request) -> JSONResponse:
        raw = await request.body()
        key = request.headers.get("x-request-key")
        if key is None:
            try:
                cmd = json.loads(raw).get("command")
                key = cmd if isinstance(cmd, str) else raw.decode("utf-8", "replace")
            except (ValueError, AttributeError):
                key = raw.decode("utf-8", "replace")
        v = assign(key, variants)
        t0 = time.perf_counter()
        try:
            r = await http.post(
                f"{v.url}/v1/plan", content=raw, headers={"content-type": "application/json"}
            )
            body: Any = r.json()
        except (httpx.HTTPError, ValueError) as e:
            stats.errors[v.name] += 1
            req_total.labels(variant=v.name, status="upstream_error").inc()
            return JSONResponse(
                {"error": f"{v.name} yanıt vermedi: {type(e).__name__}", "variant": v.name},
                status_code=502,
                headers={"X-Variant": v.name},
            )
        secs = time.perf_counter() - t0
        if r.status_code == 200:
            otype = body.get("type") or "invalid"
            stats.record(v.name, secs * 1000, bool(body.get("valid")), otype)
            out_total.labels(variant=v.name, type=otype).inc()
            dur.labels(variant=v.name).observe(secs)
            body["variant"] = v.name
            req_total.labels(variant=v.name, status="ok").inc()
        else:
            stats.errors[v.name] += 1
            req_total.labels(variant=v.name, status=f"http_{r.status_code}").inc()
        return JSONResponse(body, status_code=r.status_code, headers={"X-Variant": v.name})

    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", action="append", required=True, help="ad=url[@ağırlık]")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    app = create_app([parse_variant(s) for s in a.variant])
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
