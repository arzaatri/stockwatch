"""api/ routes. Mocks the network boundary to the inference microservice
(inference_client.score) the same way llm/graph.py's tests substitute a
fake LLM client - the actual cross-service HTTP call belongs to
inference_service/'s own tests, not here.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from stockwatch.api import inference_client
from stockwatch.api.app import app
from stockwatch.db.models import Watchlist, WindowedPriceStats
from stockwatch.explain.shap_explainer import FeatureAttribution
from stockwatch.llm.schemas import ExplanationOutput, GraphState
from stockwatch.pipeline import explain_anomaly as explain_anomaly_module

client = TestClient(app)


def _seed_price_history(db_session: Session, ticker: str, n: int) -> datetime:
    now = datetime.now(UTC)
    rng = np.random.default_rng(0)
    db_session.add(Watchlist(ticker=ticker, added_at=now, is_active=True))
    target_window_end = None
    for i in range(n):
        window_end = now - timedelta(minutes=n - i)
        target_window_end = window_end
        db_session.add(
            WindowedPriceStats(
                ticker=ticker,
                window_end=window_end,
                avg_price=100.0 + float(rng.normal()),
                total_volume=1000,
                volatility_estimate=0.1,
                price_zscore=float(rng.normal()),
                ingested_at=now,
            )
        )
    db_session.commit()
    assert target_window_end is not None
    return target_window_end


def _fake_score(matrix: pl.DataFrame, top_k_features: int = 5) -> inference_client.ScoreResult:
    """Flags every row as anomalous with a canned SHAP attribution - avoids
    depending on IsolationForest's actual thresholding, or on Postgres row
    order (no ORDER BY on the feature query), to know which row got flagged.
    """
    height = matrix.height
    scored_matrix = matrix.with_columns(
        pl.Series("anomaly_score", [-1.0] * height),
        pl.Series("is_anomaly", [1] * height),
    )
    top_features_by_key = {
        (row["ticker"], row["window_end"]): [
            FeatureAttribution(feature="price_zscore", value=3.0, shap_value=-0.2)
        ]
        for row in matrix.iter_rows(named=True)
    }
    return inference_client.ScoreResult(
        scored_matrix, top_features_by_key, datetime.now(UTC)
    )


class _FakeGraph:
    def invoke(self, state: GraphState) -> dict:
        return {
            "explanation": ExplanationOutput(
                summary="test summary",
                likely_cause_category="unclear",
                confidence="low",
                supporting_evidence=[],
            )
        }


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200


def test_detect_returns_anomalies_from_the_inference_service(
    monkeypatch, db_session: Session
) -> None:
    _seed_price_history(db_session, "AAPL", n=15)
    monkeypatch.setattr(inference_client, "score", _fake_score)

    response = client.get("/detect")

    assert response.status_code == 200
    anomalies = response.json()["anomalies"]
    assert len(anomalies) == 15
    assert all(anomaly["ticker"] == "AAPL" for anomaly in anomalies)


def test_detect_returns_empty_below_min_rows(db_session: Session) -> None:
    _seed_price_history(db_session, "AAPL", n=3)

    response = client.get("/detect")

    assert response.status_code == 200
    assert response.json() == {"anomalies": []}


def test_explain_job_completes_and_can_be_polled(
    monkeypatch, db_session: Session
) -> None:
    target_window_end = _seed_price_history(db_session, "AAPL", n=15)
    monkeypatch.setattr(inference_client, "score", _fake_score)
    monkeypatch.setattr(
        explain_anomaly_module, "get_sector_industry_as_of", lambda ticker, as_of: None
    )
    monkeypatch.setattr(
        explain_anomaly_module, "get_rating_consensus_as_of", lambda ticker, as_of: None
    )
    monkeypatch.setattr(
        explain_anomaly_module,
        "get_recent_news",
        lambda scope, scope_key, count=5, before=None, since=None: [],
    )
    monkeypatch.setattr(
        explain_anomaly_module, "build_default_graph", lambda: _FakeGraph()
    )

    create_response = client.post(
        "/explain",
        json={"ticker": "AAPL", "window_end": target_window_end.isoformat()},
    )
    assert create_response.status_code == 202
    job_id = create_response.json()["job_id"]

    # TestClient runs BackgroundTasks synchronously before returning the
    # create-job response, so the job is already done by the time we poll.
    status_response = client.get(f"/explain/{job_id}")
    assert status_response.status_code == 200
    body = status_response.json()
    assert body["status"] == "done"
    assert body["result"]["explanation"]["summary"] == "test summary"


def test_explain_unknown_job_returns_404() -> None:
    response = client.get("/explain/does-not-exist")
    assert response.status_code == 404
