"""The isolation-forest microservice's only routes: health, model metadata,
and scoring. This is where detect_and_explain.py's old in-process "load
model -> score -> SHAP" flow now lives - moved here wholesale so it's the
same computation, just reachable over HTTP instead of a direct import.
"""

from typing import Any

import polars as pl
from fastapi import APIRouter, HTTPException

from stockwatch.config import get_settings
from stockwatch.detection.feature_schema import FEATURE_COLUMNS
from stockwatch.detection.isolation_forest import get_anomalies, to_feature_array
from stockwatch.detection.model_store import (
    is_model_stale,
    load_latest_model_payload,
    resolve_ml_detector,
)
from stockwatch.explain.shap_explainer import get_explainer, top_features_for_anomaly
from stockwatch.inference_service.schemas import (
    FeatureRow,
    ModelStatus,
    ScoredRow,
    ScoreRequest,
    ScoreResponse,
)
from stockwatch.logging_utils import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/model/current", response_model=ModelStatus)
def model_current() -> ModelStatus:
    payload = load_latest_model_payload()
    if payload is None:
        return ModelStatus(trained_at=None, is_stale=False)

    max_age_days = get_settings().model_stale_after_days
    return ModelStatus(
        trained_at=payload["trained_at"],
        is_stale=is_model_stale(payload["trained_at"], max_age_days),
        contamination=payload["contamination"],
        n_rows=payload["n_rows"],
        reference_distribution=payload.get("reference_distribution"),
    )


@router.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    if not request.rows:
        return ScoreResponse(rows=[], model_trained_at=None)

    feature_matrix = _rows_to_feature_matrix(request.rows)

    try:
        detector, trained_at = resolve_ml_detector(feature_matrix)
        scored = detector.score(feature_matrix)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    anomalies = get_anomalies(scored)
    top_features_by_key: dict[tuple[str, Any], list] = {}
    if not anomalies.is_empty():
        background = to_feature_array(feature_matrix)
        explainer = get_explainer(detector.model, background)
        anomaly_features = to_feature_array(anomalies)
        for i, row in enumerate(anomalies.iter_rows(named=True)):
            top_features_by_key[(row["ticker"], row["window_end"])] = (
                top_features_for_anomaly(
                    explainer, anomaly_features[i : i + 1], k=request.top_k_features
                )
            )

    logger.info(
        "Scored %d row(s), %d anomal(y/ies)", feature_matrix.height, anomalies.height
    )
    return ScoreResponse(
        rows=[
            ScoredRow(
                ticker=row["ticker"],
                window_end=row["window_end"],
                anomaly_score=row["anomaly_score"],
                is_anomaly=bool(row["is_anomaly"]),
                top_features=top_features_by_key.get((row["ticker"], row["window_end"])),
            )
            for row in scored.iter_rows(named=True)
        ],
        model_trained_at=trained_at,
    )


def _rows_to_feature_matrix(rows: list[FeatureRow]) -> pl.DataFrame:
    try:
        data = {
            "ticker": [row.ticker for row in rows],
            "window_end": [row.window_end for row in rows],
        } | {column: [row.features[column] for row in rows] for column in FEATURE_COLUMNS}
    except KeyError as error:
        raise HTTPException(
            status_code=422, detail=f"Feature row missing required column {error}"
        ) from error
    return pl.DataFrame(data)
