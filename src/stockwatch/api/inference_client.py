"""Thin HTTP client the orchestration api service uses to reach the
isolation-forest microservice (inference_service/) - the only module in
api/ that knows that service exists. Uses `requests` (already a main
dependency, via ingestion/) rather than adding a second HTTP library just
for this.
"""

from datetime import datetime

import polars as pl
import requests

from stockwatch.config import get_settings
from stockwatch.detection.feature_schema import FEATURE_COLUMNS
from stockwatch.explain.shap_explainer import FeatureAttribution
from stockwatch.inference_service.schemas import (
    FeatureRow,
    ModelStatus,
    ScoreRequest,
    ScoreResponse,
)

REQUEST_TIMEOUT_SECONDS = 30


class ScoreResult:
    """`scored_matrix` mirrors what detection/isolation_forest.py's
    score_anomalies() used to return in-process (feature_matrix +
    anomaly_score/is_anomaly columns); `top_features_by_key` holds SHAP
    attributions for flagged rows only, keyed by (ticker, window_end).
    """

    def __init__(
        self,
        scored_matrix: pl.DataFrame,
        top_features_by_key: dict[tuple[str, datetime], list[FeatureAttribution]],
        model_trained_at: datetime | None,
    ) -> None:
        self.scored_matrix = scored_matrix
        self.top_features_by_key = top_features_by_key
        self.model_trained_at = model_trained_at


def score(feature_matrix: pl.DataFrame, top_k_features: int = 5) -> ScoreResult:
    if feature_matrix.is_empty():
        empty = feature_matrix.with_columns(
            pl.Series("anomaly_score", [], dtype=pl.Float64),
            pl.Series("is_anomaly", [], dtype=pl.Int64),
        )
        return ScoreResult(empty, {}, None)

    request = ScoreRequest(
        rows=[
            FeatureRow(
                ticker=row["ticker"],
                window_end=row["window_end"],
                features={column: row[column] for column in FEATURE_COLUMNS},
            )
            for row in feature_matrix.iter_rows(named=True)
        ],
        top_k_features=top_k_features,
    )
    response = requests.post(
        f"{get_settings().inference_service_url}/score",
        json=request.model_dump(mode="json"),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    parsed = ScoreResponse.model_validate(response.json())

    # Match back to the request's own rows by key, not position - defensive
    # against the inference service ever reordering its response.
    scored_by_key = {(row.ticker, row.window_end): row for row in parsed.rows}
    anomaly_scores = []
    is_anomaly_flags = []
    for row in feature_matrix.iter_rows(named=True):
        scored_row = scored_by_key[(row["ticker"], row["window_end"])]
        anomaly_scores.append(scored_row.anomaly_score)
        is_anomaly_flags.append(int(scored_row.is_anomaly))

    scored_matrix = feature_matrix.with_columns(
        pl.Series("anomaly_score", anomaly_scores),
        pl.Series("is_anomaly", is_anomaly_flags),
    )
    top_features_by_key = {
        key: row.top_features
        for key, row in scored_by_key.items()
        if row.top_features is not None
    }
    return ScoreResult(scored_matrix, top_features_by_key, parsed.model_trained_at)


def get_model_status() -> ModelStatus:
    response = requests.get(
        f"{get_settings().inference_service_url}/model/current",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return ModelStatus.model_validate(response.json())
