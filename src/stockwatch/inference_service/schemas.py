"""HTTP contract for the inference microservice. `FeatureRow.features` is a
dict keyed by detection/feature_schema.py's FEATURE_COLUMNS rather than a
plain list, so the JSON payload is self-documenting and robust to either
side reordering columns.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from stockwatch.explain.shap_explainer import FeatureAttribution


class FeatureRow(BaseModel):
    ticker: str
    window_end: datetime
    features: dict[str, float]


class ScoreRequest(BaseModel):
    rows: list[FeatureRow]
    top_k_features: int = 5


class ScoredRow(BaseModel):
    ticker: str
    window_end: datetime
    anomaly_score: float
    is_anomaly: bool
    top_features: list[FeatureAttribution] | None = None


class ScoreResponse(BaseModel):
    rows: list[ScoredRow]
    model_trained_at: datetime | None


class ModelStatus(BaseModel):
    trained_at: datetime | None
    is_stale: bool
    contamination: float | str | None = None
    n_rows: int | None = None
    reference_distribution: dict[str, dict] | None = Field(
        default=None,
        description="Per-feature + score reference histograms captured at "
        "training time, used by the orchestration API's drift monitoring.",
    )
