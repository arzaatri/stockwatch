import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import IsolationForest

from stockwatch.detection.ml_detector import MLAnomalyDetector
from stockwatch.features.build_features import FEATURE_COLUMNS


def _synthetic_matrix(n: int = 60, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    data = {column: rng.normal(size=n).tolist() for column in FEATURE_COLUMNS}
    data["price_zscore"][0] = 50.0  # inject an obvious outlier
    return pl.DataFrame(data)


def test_score_before_fit_raises() -> None:
    detector = MLAnomalyDetector()

    with pytest.raises(RuntimeError):
        detector.score(_synthetic_matrix(n=5))


def test_fit_sets_the_underlying_model() -> None:
    detector = MLAnomalyDetector()

    detector.fit(_synthetic_matrix())

    assert isinstance(detector.model, IsolationForest)


def test_detect_flags_the_injected_outlier_most_anomalous_first() -> None:
    detector = MLAnomalyDetector()

    anomalies = detector.detect(_synthetic_matrix())

    assert anomalies.height >= 1
    assert 50.0 in anomalies["price_zscore"].to_list()
    scores = anomalies["anomaly_score"].to_list()
    assert scores == sorted(scores)
