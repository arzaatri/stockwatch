"""Request/response models for the orchestration api service's own routes
(as opposed to inference_service/schemas.py, which is the contract with the
isolation-forest microservice)."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

from stockwatch.explain.shap_explainer import FeatureAttribution


class DetectedAnomaly(BaseModel):
    ticker: str
    window_end: datetime
    anomaly_score: float
    top_features: list[FeatureAttribution]


class DetectResponse(BaseModel):
    anomalies: list[DetectedAnomaly]


class ExplainRequest(BaseModel):
    ticker: str
    window_end: datetime
    news_count: int = 5


class ExplainJobCreated(BaseModel):
    job_id: str
    status: Literal["pending"]


class ExplainJobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "running", "done", "error"]
    ticker: str
    window_end: datetime
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime
    completed_at: datetime | None = None
