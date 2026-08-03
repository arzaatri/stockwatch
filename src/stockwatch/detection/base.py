"""AnomalyDetector: the common contract both detection strategies implement,
so callers (pipeline/detect_and_explain.py, and eventually a dashboard) can
swap MLAnomalyDetector <-> SimpleAnomalyDetector without caring which one
produced a given result.
"""

from abc import ABC, abstractmethod

import polars as pl


class AnomalyDetector(ABC):
    """Convention shared by every strategy: `anomaly_score` is lower (more
    negative) for more anomalous rows, and `is_anomaly` is 1/0 - this keeps
    `detect()`'s sort/filter logic (and any downstream dashboard code) the
    same regardless of which strategy produced the score.
    """

    @abstractmethod
    def fit(self, feature_matrix: pl.DataFrame) -> None:
        """Learns whatever "normal" looks like from `feature_matrix`."""

    @abstractmethod
    def score(self, feature_matrix: pl.DataFrame) -> pl.DataFrame:
        """Returns `feature_matrix` with `anomaly_score`/`is_anomaly` columns added."""

    def detect(self, feature_matrix: pl.DataFrame) -> pl.DataFrame:
        """fit -> score -> flagged rows only, most anomalous first. The one
        entrypoint most callers need; `fit`/`score` stay separate methods so
        callers that need the fitted state directly (e.g. SHAP against
        MLAnomalyDetector.model) can still get at it.
        """
        self.fit(feature_matrix)
        scored = self.score(feature_matrix)
        return scored.filter(pl.col("is_anomaly") == 1).sort("anomaly_score")
