"""Pydantic types for drift reporting (monitoring/drift.py) - kept separate
from api/schemas.py since drift is a domain concept usable outside the API
too (e.g. the dashboard builds a DriftReport directly, not via HTTP)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

Severity = Literal["none", "moderate", "significant"]


class FeatureDrift(BaseModel):
    feature: str
    psi: float
    severity: Severity


class DriftReport(BaseModel):
    model_trained_at: datetime | None
    feature_drift: list[FeatureDrift]
    score_drift: FeatureDrift
    observed_anomaly_rate: float
    n_rows_evaluated: int
