import numpy as np
import polars as pl
import pytest

from stockwatch.detection.simple_detector import SimpleAnomalyDetector
from stockwatch.features.build_features import FEATURE_COLUMNS


def _synthetic_matrix(n: int = 60, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    data = {column: rng.normal(size=n).tolist() for column in FEATURE_COLUMNS}
    data["price_zscore"][0] = 50.0  # inject an obvious outlier
    return pl.DataFrame(data)


def test_score_before_fit_raises() -> None:
    detector = SimpleAnomalyDetector()

    with pytest.raises(RuntimeError):
        detector.score(_synthetic_matrix(n=5))


def test_detect_flags_the_injected_outlier_most_anomalous_first() -> None:
    detector = SimpleAnomalyDetector()

    anomalies = detector.detect(_synthetic_matrix())

    assert anomalies.height >= 1
    assert 50.0 in anomalies["price_zscore"].to_list()
    scores = anomalies["anomaly_score"].to_list()
    assert scores == sorted(scores)


def test_threshold_controls_sensitivity() -> None:
    matrix = _synthetic_matrix()

    lenient = SimpleAnomalyDetector(threshold=100.0).detect(matrix)
    strict = SimpleAnomalyDetector(threshold=0.0).detect(matrix)

    assert lenient.height == 0
    assert strict.height == matrix.height


def test_detect_does_not_crash_on_constant_features() -> None:
    matrix = pl.DataFrame({column: [1.0] * 10 for column in FEATURE_COLUMNS})
    detector = SimpleAnomalyDetector()

    anomalies = detector.detect(matrix)

    assert anomalies.height == 0
