"""AnomalyDetector backed by a plain per-feature standard-deviation rule: a
row is anomalous if any feature is more than `threshold` standard deviations
from that feature's mean across the batch. No training/model needed - this
is the "simple strategy" scaffold, a baseline to compare MLAnomalyDetector
against and to unblock dashboard work before a real model is trained.
"""

import numpy as np
import polars as pl

from stockwatch.detection.base import AnomalyDetector
from stockwatch.detection.isolation_forest import to_feature_array

DEFAULT_THRESHOLD = 3.0  # "3-sigma" rule of thumb


class SimpleAnomalyDetector(AnomalyDetector):
    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, feature_matrix: pl.DataFrame) -> None:
        array = to_feature_array(feature_matrix)
        self._mean = array.mean(axis=0)
        self._std = array.std(axis=0)

    def score(self, feature_matrix: pl.DataFrame) -> pl.DataFrame:
        if self._mean is None or self._std is None:
            raise RuntimeError("SimpleAnomalyDetector.score() called before fit()")

        array = to_feature_array(feature_matrix)
        safe_std = np.where(
            self._std == 0, 1.0, self._std
        )  # constant feature -> zscore 0
        max_abs_zscore = np.abs((array - self._mean) / safe_std).max(axis=1)

        return feature_matrix.with_columns(
            pl.Series("anomaly_score", -max_abs_zscore),  # lower = more anomalous
            pl.Series("is_anomaly", (max_abs_zscore > self.threshold).astype(int)),
        )
