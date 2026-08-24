from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from stockwatch.detection import model_store
from stockwatch.detection.feature_schema import FEATURE_COLUMNS
from stockwatch.detection.ml_detector import MLAnomalyDetector
from stockwatch.inference_service.app import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _isolated_models_dir(tmp_path, monkeypatch):
    """Never touch the real project models/ directory from tests."""
    monkeypatch.setattr(model_store, "MODELS_DIR", tmp_path / "models")


def _synthetic_matrix(n: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({column: rng.normal(size=n).tolist() for column in FEATURE_COLUMNS})


def _synthetic_rows(n: int, seed: int = 0) -> list[dict]:
    matrix = _synthetic_matrix(n, seed=seed)
    now = datetime.now(UTC)
    return [
        {
            "ticker": "AAPL",
            "window_end": (now - timedelta(minutes=n - i)).isoformat(),
            "features": {column: row[column] for column in FEATURE_COLUMNS},
        }
        for i, row in enumerate(matrix.iter_rows(named=True))
    ]


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_current_with_no_saved_model() -> None:
    response = client.get("/model/current")
    assert response.status_code == 200
    body = response.json()
    assert body["trained_at"] is None
    assert body["is_stale"] is False


def test_model_current_reflects_saved_model() -> None:
    detector = MLAnomalyDetector(random_state=1, contamination=0.1)
    detector.fit(_synthetic_matrix(30))
    model_store.save_model(detector, n_rows=30)

    response = client.get("/model/current")
    body = response.json()
    assert body["trained_at"] is not None
    assert body["n_rows"] == 30
    assert body["contamination"] == 0.1


def test_score_returns_one_row_per_request_row() -> None:
    response = client.post(
        "/score", json={"rows": _synthetic_rows(30), "top_k_features": 3}
    )
    assert response.status_code == 200
    assert len(response.json()["rows"]) == 30


def test_score_only_computes_shap_for_flagged_anomalies() -> None:
    # contamination=0.5 against a saved model guarantees some (but not all)
    # rows get flagged, so both branches of the SHAP-only-for-anomalies
    # logic are actually exercised, deterministically.
    detector = MLAnomalyDetector(random_state=0, contamination=0.5)
    detector.fit(_synthetic_matrix(40))
    model_store.save_model(detector, n_rows=40)

    response = client.post(
        "/score", json={"rows": _synthetic_rows(40, seed=1), "top_k_features": 3}
    )
    rows = response.json()["rows"]
    assert any(row["is_anomaly"] for row in rows)
    assert any(not row["is_anomaly"] for row in rows)
    for row in rows:
        if row["is_anomaly"]:
            assert row["top_features"] is not None
            assert len(row["top_features"]) <= 3
        else:
            assert row["top_features"] is None


def test_score_below_min_rows_with_no_saved_model_returns_422() -> None:
    response = client.post("/score", json={"rows": _synthetic_rows(3)})
    assert response.status_code == 422


def test_score_empty_rows_returns_empty_response() -> None:
    response = client.post("/score", json={"rows": []})
    assert response.status_code == 200
    assert response.json() == {"rows": [], "model_trained_at": None}


def test_score_rejects_a_row_missing_a_feature_column() -> None:
    rows = _synthetic_rows(15)
    del rows[0]["features"][FEATURE_COLUMNS[0]]

    response = client.post("/score", json={"rows": rows})
    assert response.status_code == 422
