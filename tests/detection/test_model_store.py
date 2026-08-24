from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from stockwatch.detection import model_store
from stockwatch.detection.ml_detector import MLAnomalyDetector
from stockwatch.features.build_features import FEATURE_COLUMNS
from stockwatch.monitoring.drift import SCORE_DISTRIBUTION_KEY


def _synthetic_matrix(n: int = 60, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {column: rng.normal(size=n).tolist() for column in FEATURE_COLUMNS}
    )


@pytest.fixture(autouse=True)
def _isolated_models_dir(tmp_path, monkeypatch):
    """Never touch the real project models/ directory from tests - a stray
    real model on disk (e.g. from running train_model.sh manually) would
    otherwise make these tests order/environment dependent.
    """
    monkeypatch.setattr(model_store, "MODELS_DIR", tmp_path / "models")


def test_latest_model_path_is_none_when_nothing_saved() -> None:
    assert model_store.latest_model_path() is None


def test_save_raises_if_detector_not_fit() -> None:
    detector = MLAnomalyDetector()

    with pytest.raises(RuntimeError):
        model_store.save_model(detector, _synthetic_matrix())


def test_save_and_load_round_trips_the_model() -> None:
    detector = MLAnomalyDetector(random_state=42, contamination=0.1)
    matrix = _synthetic_matrix()
    detector.fit(matrix)

    path = model_store.save_model(detector, matrix)

    assert path.exists()
    loaded = model_store.load_latest_detector()
    assert loaded is not None
    loaded_detector, trained_at = loaded
    assert loaded_detector.random_state == 42
    assert loaded_detector.contamination == 0.1
    assert loaded_detector.model is not None
    assert (datetime.now(UTC) - trained_at).total_seconds() < 5


def test_save_computes_a_reference_distribution_for_every_feature_and_the_score() -> None:
    detector = MLAnomalyDetector(random_state=0)
    matrix = _synthetic_matrix(n=60)
    detector.fit(matrix)

    model_store.save_model(detector, matrix)

    payload = model_store.load_latest_model_payload()
    assert payload is not None
    reference_distribution = payload["reference_distribution"]
    for column in [*FEATURE_COLUMNS, SCORE_DISTRIBUTION_KEY]:
        assert column in reference_distribution
        bin_proportions = reference_distribution[column]["bin_proportions"]
        assert len(reference_distribution[column]["bin_edges"]) == len(bin_proportions) + 1
        assert sum(bin_proportions) == pytest.approx(1.0)


def test_latest_model_path_picks_the_most_recently_saved() -> None:
    detector = MLAnomalyDetector()
    matrix = _synthetic_matrix()
    detector.fit(matrix)

    first_path = model_store.save_model(detector, matrix)
    second_path = model_store.save_model(detector, matrix)

    assert first_path != second_path
    assert model_store.latest_model_path() == second_path


@pytest.mark.parametrize(
    ("age_days", "max_age_days", "expected"),
    [(10, 7, True), (1, 7, False)],
)
def test_is_model_stale(age_days: int, max_age_days: int, expected: bool) -> None:
    trained_at = datetime.now(UTC) - timedelta(days=age_days)

    assert model_store.is_model_stale(trained_at, max_age_days) is expected


def test_resolve_ml_detector_falls_back_to_an_ad_hoc_fit_when_nothing_saved() -> None:
    matrix = _synthetic_matrix()

    detector, trained_at = model_store.resolve_ml_detector(matrix)

    assert trained_at is None
    assert detector.model is not None


def test_resolve_ml_detector_prefers_the_saved_model() -> None:
    saved_detector = MLAnomalyDetector(random_state=99)
    matrix = _synthetic_matrix()
    saved_detector.fit(matrix)
    model_store.save_model(saved_detector, matrix)

    detector, trained_at = model_store.resolve_ml_detector(_synthetic_matrix())

    assert trained_at is not None
    assert detector.random_state == 99
