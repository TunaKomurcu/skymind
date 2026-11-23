from pathlib import Path

import mlflow
import pytest

from skymind.mlops import log_run


def test_log_run_records_params_and_skips_nan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    run_id = log_run(
        experiment="test",
        run_name="r1",
        params={"lr": 2e-4},
        metrics={"json_valid": 0.9, "fabrication_rate": float("nan")},
    )
    run = mlflow.get_run(run_id)
    assert run.data.params["lr"] == "0.0002"
    assert run.data.metrics["json_valid"] == 0.9
    assert "fabrication_rate" not in run.data.metrics
