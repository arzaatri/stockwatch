"""GET /detect - the fast path, no LLM call."""

from fastapi import APIRouter, Query

from stockwatch.api.schemas import DetectedAnomaly, DetectResponse
from stockwatch.pipeline.detect_and_explain import detect_anomalies

router = APIRouter()


@router.get("/detect", response_model=DetectResponse)
def detect(
    tickers: list[str] | None = Query(
        default=None, description="Restrict to these tickers; omit for the whole watchlist."
    ),
) -> DetectResponse:
    anomalies = detect_anomalies(tickers=tickers)
    return DetectResponse(
        anomalies=[
            DetectedAnomaly(
                ticker=anomaly["ticker"],
                window_end=anomaly["window_end"],
                anomaly_score=anomaly["anomaly_score"],
                top_features=anomaly["top_features"],
            )
            for anomaly in anomalies
        ]
    )
