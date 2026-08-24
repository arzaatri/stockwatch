"""Pure PSI math (monitoring/drift.py) - no DB, no model, just numpy/polars."""

import numpy as np
import polars as pl
import pytest

from stockwatch.detection.feature_schema import FEATURE_COLUMNS
from stockwatch.monitoring import drift


def _synthetic_matrix(n: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {column: rng.normal(size=n).tolist() for column in FEATURE_COLUMNS}
    )


def test_compute_psi_is_zero_for_identical_distributions() -> None:
    proportions = [0.2, 0.2, 0.2, 0.2, 0.2]
    assert drift.compute_psi(proportions, proportions) == pytest.approx(0.0, abs=1e-6)


def test_compute_psi_is_large_for_a_big_shift() -> None:
    reference = [0.2, 0.2, 0.2, 0.2, 0.2]
    current = [0.9, 0.025, 0.025, 0.025, 0.025]

    psi = drift.compute_psi(reference, current)

    assert psi > 0.25  # "significant" by the standard rule of thumb


@pytest.mark.parametrize(
    ("psi", "expected"),
    [(0.01, "none"), (0.15, "moderate"), (0.5, "significant")],
)
def test_classify_psi(psi: float, expected: str) -> None:
    assert drift.classify_psi(psi) == expected


def test_bin_proportions_sums_to_one() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=200)
    edges = drift._quantile_bin_edges(values, n_bins=5)

    proportions = drift.bin_proportions(values, edges)

    assert sum(proportions) == pytest.approx(1.0)
    assert len(proportions) == len(edges) - 1


def test_bin_proportions_handles_empty_values() -> None:
    edges = [-np.inf, 0.0, np.inf]
    assert drift.bin_proportions(np.array([]), edges) == [0.0, 0.0]


def test_build_reference_distribution_covers_every_feature_and_the_score() -> None:
    matrix = _synthetic_matrix(60)
    scores = np.random.default_rng(1).normal(size=60)

    reference = drift.build_reference_distribution(matrix, scores)

    assert set(reference) == {*FEATURE_COLUMNS, drift.SCORE_DISTRIBUTION_KEY}
    for entry in reference.values():
        assert sum(entry["bin_proportions"]) == pytest.approx(1.0)


def test_build_drift_report_flags_a_shifted_feature_as_significant() -> None:
    reference_matrix = _synthetic_matrix(200, seed=0)
    reference_scores = np.random.default_rng(1).normal(size=200)
    reference_distribution = drift.build_reference_distribution(
        reference_matrix, reference_scores
    )

    # Everything scored "normal" except price_zscore, wildly shifted from
    # what the reference distribution saw.
    current = _synthetic_matrix(100, seed=2)
    current = current.with_columns(pl.Series("price_zscore", [50.0] * 100))
    scored_matrix = current.with_columns(
        pl.Series("anomaly_score", np.random.default_rng(3).normal(size=100)),
        pl.Series("is_anomaly", [0] * 90 + [1] * 10),
    )

    report = drift.build_drift_report(scored_matrix, reference_distribution, None)

    price_zscore_drift = next(
        fd for fd in report.feature_drift if fd.feature == "price_zscore"
    )
    assert price_zscore_drift.severity == "significant"
    assert report.observed_anomaly_rate == pytest.approx(0.10)
    assert report.n_rows_evaluated == 100


def test_build_drift_report_handles_an_empty_matrix() -> None:
    reference_matrix = _synthetic_matrix(60)
    reference_scores = np.random.default_rng(1).normal(size=60)
    reference_distribution = drift.build_reference_distribution(
        reference_matrix, reference_scores
    )
    empty = reference_matrix.clear().with_columns(
        pl.Series("anomaly_score", [], dtype=pl.Float64),
        pl.Series("is_anomaly", [], dtype=pl.Int64),
    )

    report = drift.build_drift_report(empty, reference_distribution, None)

    assert report.n_rows_evaluated == 0
    assert report.observed_anomaly_rate == 0.0
