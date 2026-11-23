"""MLflow deney takibi için ince yardımcı.

Yerelde SQLite deposu kullanır (sunucu gerekmez). ``log_run`` her faz tarafından aynı şekilde
çağrılır; böylece tüm deneyler tek tabloda karşılaştırılabilir.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path

import mlflow

DEFAULT_URI = "sqlite:///outputs/mlflow.db"


def _tracking_uri() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_URI)
    if uri.startswith("sqlite:///"):
        Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    return uri


def log_run(
    experiment: str,
    run_name: str,
    params: Mapping[str, object],
    metrics: Mapping[str, float],
    artifacts: list[Path] | None = None,
    tags: Mapping[str, str] | None = None,
) -> str:
    """Tek bir koşuyu kaydeder ve run_id döndürür. NaN metrikler atlanır."""
    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics({k: v for k, v in metrics.items() if not math.isnan(v)})
        if tags:
            mlflow.set_tags(dict(tags))
        for path in artifacts or []:
            mlflow.log_artifact(str(path))
        return str(run.info.run_id)
