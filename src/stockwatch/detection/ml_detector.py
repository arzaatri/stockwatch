"""AnomalyDetector backed by scikit-learn's IsolationForest. Thin adapter
over isolation_forest.py's tested functions - this class owns no detection
math itself, just the fit/score lifecycle and giving callers (SHAP) access
to the underlying fitted model.
"""

import polars as pl
from sklearn.ensemble import IsolationForest

from stockwatch.detection.base import AnomalyDetector
from stockwatch.detection.isolation_forest import fit_isolation_forest, score_anomalies


class MLAnomalyDetector(AnomalyDetector):
    def __init__(
        self, random_state: int = 0, contamination: float | str = "auto"
    ) -> None:
        self.random_state = random_state
        self.contamination = contamination
        self.model: IsolationForest | None = None

    def fit(self, feature_matrix: pl.DataFrame) -> None:
        self.model = fit_isolation_forest(
            feature_matrix,
            random_state=self.random_state,
            contamination=self.contamination,
        )

    def score(self, feature_matrix: pl.DataFrame) -> pl.DataFrame:
        if self.model is None:
            raise RuntimeError("MLAnomalyDetector.score() called before fit()")
        return score_anomalies(self.model, feature_matrix)
