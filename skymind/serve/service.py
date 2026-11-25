"""Motordan bağımsız plan servisi mantığı: istek doğrulama, çıkarım, şema doğrulaması, metrikler.

C++ ``PlannerService`` ile aynı davranış ve aynı yanıt biçimi (karşılaştırma adil olsun diye).
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import ValidationError

from skymind.schemas import parse_output
from skymind.serve.engines import Engine

MAX_COMMAND_BYTES = 2000
BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)


def validate_text(text: str) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """(plan, tür, hata). Pydantic şeması tek doğruluk kaynağıdır."""
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        return None, None, "geçerli JSON değil"
    try:
        out = parse_output(doc)
    except ValidationError as e:
        first = e.errors()[0]
        loc = ".".join(str(p) for p in first["loc"])
        return None, None, f"{loc}: {first['msg']}"
    return doc, out.type, None


class PlannerService:
    def __init__(self, engine: Engine, system: str, registry: CollectorRegistry | None = None):
        self.engine = engine
        self.system = system
        self._lock = threading.Lock()  # tek model bağlamı: istekler sırayla işlenir (C++ ile aynı)
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.requests = Counter("skymind_requests", "Plan istekleri", ["status"], registry=r)
        self.output = Counter("skymind_output", "Üretilen çıktı türleri", ["type"], registry=r)
        self.prompt_tokens = Counter("skymind_prompt_tokens", "İşlenen istem token", registry=r)
        self.generated_tokens = Counter("skymind_generated_tokens", "Üretilen token", registry=r)
        self.in_flight = Gauge("skymind_requests_in_flight", "İşlenen istek", registry=r)
        self.duration = Histogram(
            "skymind_request_duration_seconds", "Uçtan uca süre", buckets=BUCKETS, registry=r
        )
        info = Gauge("skymind_model_info", "Yüklü model", ["engine"], registry=r)
        info.labels(engine=engine.name).set(1)

    def plan(self, body: Any) -> tuple[int, dict[str, Any]]:
        t0 = time.perf_counter()
        command = body.get("command") if isinstance(body, dict) else None
        if not isinstance(command, str):
            self.requests.labels(status="bad_request").inc()
            return 400, {"error": 'govde {"command": "..."} bicimde JSON olmali'}
        if not command or len(command.encode("utf-8")) > MAX_COMMAND_BYTES:
            self.requests.labels(status="bad_request").inc()
            return 400, {"error": "command bos olamaz ve en fazla 2000 bayt olabilir"}

        self.in_flight.inc()
        try:
            with self._lock:
                g = self.engine.generate(self.system, command)
            secs = time.perf_counter() - t0
            plan, otype, err = validate_text(g.text)
            self.requests.labels(status="ok").inc()
            self.output.labels(type=otype or "invalid").inc()
            self.prompt_tokens.inc(g.prompt_tokens)
            self.generated_tokens.inc(g.new_tokens)
            self.duration.observe(secs)
            return 200, {
                "raw": g.text,
                "valid": err is None,
                "type": otype,
                "validation_error": err,
                "plan": plan,
                "usage": {
                    "prompt_tokens": g.prompt_tokens,
                    "new_tokens": g.new_tokens,
                    "prefill_ms": g.prefill_ms,
                    "decode_ms": g.decode_ms,
                    "truncated": g.truncated,
                },
                "latency_ms": secs * 1000,
                "engine": self.engine.name,
            }
        except Exception as e:  # noqa: BLE001 - istemciye 500 olarak iletilir
            self.requests.labels(status="error").inc()
            return 500, {"error": f"cikarim hatasi: {e}"}
        finally:
            self.in_flight.dec()

    def metrics(self) -> bytes:
        return generate_latest(self.registry)
