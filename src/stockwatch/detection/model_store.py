"""Persists trained IsolationForest models to models/ (via train_model.sh),
so the app reuses one model trained once instead of refitting from scratch on
every detect/explain call. Also the single source of truth for "how long ago
was the model last trained" (surfaced as a staleness warning by callers).
"""

from datetime import UTC, datetime
from pathlib import Path

import joblib
import polars as pl

from stockwatch.detection.isolation_forest import to_feature_array
from stockwatch.detection.ml_detector import MLAnomalyDetector
from stockwatch.logging_utils import get_logger
from stockwatch.monitoring.drift import build_reference_distribution

logger = get_logger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models"
_FILENAME_PREFIX = "isolation_forest_"
_FILENAME_SUFFIX = ".joblib"
_FILENAME_TIME_FORMAT = "%Y%m%dT%H%M%S%fZ"  # microsecond precision - sorts
# lexicographically = chronologically, and stays unique even for training
# runs that happen to fall within the same second (e.g. in tests).


def save_model(detector: MLAnomalyDetector, feature_matrix: pl.DataFrame) -> Path:
    """Persists `detector.model` plus enough metadata (contamination,
    random_state, when, how many rows it was trained on, and a reference
    distribution for drift monitoring - see monitoring/drift.py) to reload
    it as an equivalent MLAnomalyDetector later via `load_latest_detector`.
    `feature_matrix` must be the exact batch `detector` was just fit on -
    that's what makes the reference distribution meaningful.
    """
    if detector.model is None:
        raise RuntimeError("Cannot save an unfit MLAnomalyDetector - call .fit() first")

    trained_at = datetime.now(UTC)
    scores = detector.model.decision_function(to_feature_array(feature_matrix))
    MODELS_DIR.mkdir(exist_ok=True)
    path = (
        MODELS_DIR
        / f"{_FILENAME_PREFIX}{trained_at.strftime(_FILENAME_TIME_FORMAT)}{_FILENAME_SUFFIX}"
    )
    joblib.dump(
        {
            "model": detector.model,
            "random_state": detector.random_state,
            "contamination": detector.contamination,
            "trained_at": trained_at,
            "n_rows": feature_matrix.height,
            "reference_distribution": build_reference_distribution(feature_matrix, scores),
        },
        path,
    )
    logger.info("Saved trained model to %s (%d rows)", path, feature_matrix.height)
    return path


def latest_model_path() -> Path | None:
    if not MODELS_DIR.exists():
        return None
    candidates = sorted(MODELS_DIR.glob(f"{_FILENAME_PREFIX}*{_FILENAME_SUFFIX}"))
    return candidates[-1] if candidates else None


def load_latest_model_payload() -> dict | None:
    """The raw joblib payload `save_model` wrote (model, random_state,
    contamination, trained_at, n_rows, reference_distribution) - the one
    place anything that needs more than just a ready-to-score detector (e.g.
    the inference service's /model/current, drift reporting) should read
    from. None if train_model.sh has never been run.
    """
    path = latest_model_path()
    if path is None:
        return None
    return joblib.load(path)


def load_latest_detector() -> tuple[MLAnomalyDetector, datetime] | None:
    """The most recently trained+saved model, ready to `.score()` (already
    "fit" - no need to call `.fit()` again). None if train_model.sh has never
    been run.
    """
    payload = load_latest_model_payload()
    if payload is None:
        return None

    detector = MLAnomalyDetector(
        random_state=payload["random_state"], contamination=payload["contamination"]
    )
    detector.model = payload["model"]
    return detector, payload["trained_at"]


def is_model_stale(trained_at: datetime, max_age_days: int) -> bool:
    age_days = (datetime.now(UTC) - trained_at).total_seconds() / 86400
    return age_days > max_age_days


def resolve_ml_detector(
    feature_matrix: pl.DataFrame,
) -> tuple[MLAnomalyDetector, datetime | None]:
    """Prefers the most recently trained+saved model (see train_model.sh);
    falls back to an ad-hoc in-memory fit (not persisted) if none exists yet,
    so the app still works before train_model.sh has ever been run. Returns
    (detector, trained_at) - trained_at is None for an ad-hoc fit, since
    "just fit this second" is never stale.
    """
    loaded = load_latest_detector()
    if loaded is not None:
        return loaded

    logger.warning(
        "No trained model found - fitting an ad-hoc model for this call. "
        "Run train_model.sh to persist one instead."
    )
    detector = MLAnomalyDetector()
    detector.fit(feature_matrix)
    return detector, None
