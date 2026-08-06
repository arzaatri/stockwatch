import numpy as np
import polars as pl
import pytest

from stockwatch.detection.isolation_forest import (
    MIN_ROWS_TO_FIT,
    fit_isolation_forest,
    get_anomalies,
    score_anomalies,
    to_feature_array,
)
from stockwatch.features.build_features import FEATURE_COLUMNS


def _synthetic_matrix(n: int = 60, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    data = {column: rng.normal(size=n).tolist() for column in FEATURE_COLUMNS}
    data["price_zscore"][0] = 50.0  # inject an obvious outlier
    return pl.DataFrame(data)


def test_to_feature_array_matches_column_order() -> None:
    matrix = _synthetic_matrix(n=5)

    array = to_feature_array(matrix)

    assert array.shape == (5, len(FEATURE_COLUMNS))


def test_fit_and_score_flags_the_injected_outlier() -> None:
    matrix = _synthetic_matrix()

    model = fit_isolation_forest(matrix)
    scored = score_anomalies(model, matrix)

    assert "anomaly_score" in scored.columns
    assert "is_anomaly" in scored.columns
    anomalies = get_anomalies(scored)
    assert anomalies.height >= 1
    assert 50.0 in anomalies["price_zscore"].to_list()


def test_get_anomalies_sorts_most_anomalous_first() -> None:
    matrix = _synthetic_matrix()
    scored = score_anomalies(fit_isolation_forest(matrix), matrix)

    anomalies = get_anomalies(scored)

    scores = anomalies["anomaly_score"].to_list()
    assert scores == sorted(scores)


def test_fit_raises_below_min_rows() -> None:
    matrix = _synthetic_matrix(n=MIN_ROWS_TO_FIT - 1)

    with pytest.raises(ValueError, match="at least"):
        fit_isolation_forest(matrix)


def test_fit_accepts_exactly_min_rows() -> None:
    matrix = _synthetic_matrix(n=MIN_ROWS_TO_FIT)

    fit_isolation_forest(matrix)  # should not raise


def test_contamination_is_configurable() -> None:
    matrix = _synthetic_matrix()

    model = fit_isolation_forest(matrix, contamination=0.5)

    assert model.contamination == 0.5


def test_to_feature_array_raises_a_clear_error_on_nan() -> None:
    matrix = _synthetic_matrix(n=5)
    matrix = matrix.with_columns(pl.lit(None).cast(pl.Float64).alias("price_zscore"))

    with pytest.raises(ValueError, match="NaN"):
        to_feature_array(matrix)
