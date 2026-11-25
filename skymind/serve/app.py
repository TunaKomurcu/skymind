"""FastAPI uygulaması.

    python -m skymind.serve.app --model outputs/merged-sft --port 8092

Uçlar: POST /v1/plan {"command": "..."}   GET /healthz   GET /metrics
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from skymind.planner.prompt import SYSTEM_SHORT
from skymind.serve.service import PlannerService


def create_app(service: PlannerService) -> FastAPI:
    app = FastAPI(title="SkyMind (Python baseline)")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(service.metrics(), media_type="text/plain; version=0.0.4")

    @app.post("/v1/plan")
    async def plan(request: Request) -> JSONResponse:
        body: Any
        try:
            body = await request.json()
        except ValueError:
            body = None
        # çıkarım bloklayıcıdır: olay döngüsünü tıkamamak için iş parçacığı havuzunda çalıştır
        status, payload = await run_in_threadpool(service.plan, body)
        return JSONResponse(payload, status_code=status)

    return app


def main() -> None:
    import uvicorn

    from skymind.serve.engines import HFEngine

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="birleştirilmiş HF modeli dizini")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--system", type=Path, default=None, help="varsayılan: eğitimdeki kısa istem")
    a = ap.parse_args()

    system = a.system.read_text(encoding="utf-8").rstrip("\n") if a.system else SYSTEM_SHORT
    engine = HFEngine(a.model, max_new_tokens=a.max_new_tokens, device=a.device)
    app = create_app(PlannerService(engine, system))
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
