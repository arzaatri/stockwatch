"""GET /monitoring/drift - compares the current feature matrix + scores
against the reference distribution captured when the current model was
trained (monitoring/drift.py, detection/model_store.py)."""

from fastapi import APIRouter, HTTPException, Query

from stockwatch.api import inference_client
from stockwatch.features.build_features import build_feature_matrix
from stockwatch.monitoring.drift import build_drift_report
from stockwatch.monitoring.schemas import DriftReport

router = APIRouter()


@router.get("/monitoring/drift", response_model=DriftReport)
def drift(
    tickers: list[str] | None = Query(
        default=None, description="Restrict to these tickers; omit for the whole watchlist."
    ),
) -> DriftReport:
    status = inference_client.get_model_status()
    if status.reference_distribution is None:
        raise HTTPException(
            status_code=409,
            detail="No trained model with a reference distribution yet - "
            "run `stockwatch train-model`.",
        )

    feature_matrix = build_feature_matrix(tickers)
    if feature_matrix.is_empty():
        raise HTTPException(status_code=422, detail="No feature data available yet.")

    result = inference_client.score(feature_matrix)
    return build_drift_report(
        result.scored_matrix, status.reference_distribution, status.trained_at
    )
