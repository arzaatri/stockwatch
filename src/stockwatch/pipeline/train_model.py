"""Trains an IsolationForest on the full current feature matrix and persists
it (detection/model_store.py) - the batch counterpart to detect_and_explain.py,
which then just loads and scores against the saved model instead of refitting
every call. Run via `train_model.sh` / `stockwatch train-model`.
"""

from pathlib import Path

from stockwatch.detection.isolation_forest import MIN_ROWS_TO_FIT
from stockwatch.detection.ml_detector import MLAnomalyDetector
from stockwatch.detection.model_store import save_model
from stockwatch.features.build_features import build_feature_matrix
from stockwatch.logging_utils import get_logger

logger = get_logger(__name__)


def train_and_save_model(
    contamination: float | str = "auto", random_state: int = 0
) -> Path:
    feature_matrix = build_feature_matrix()
    if feature_matrix.height < MIN_ROWS_TO_FIT:
        raise ValueError(
            f"Need at least {MIN_ROWS_TO_FIT} rows to train, got {feature_matrix.height} "
            "- run `stockwatch backfill` first."
        )

    logger.info(
        "Training IsolationForest on %d rows (contamination=%s, random_state=%d)",
        feature_matrix.height,
        contamination,
        random_state,
    )
    detector = MLAnomalyDetector(random_state=random_state, contamination=contamination)
    detector.fit(feature_matrix)
    return save_model(detector, n_rows=feature_matrix.height)
