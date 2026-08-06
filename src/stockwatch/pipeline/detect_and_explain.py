"""Orchestrates the "explain" half of a run: build features -> detect
anomalies (against the most recently trained model, see train_model.sh) ->
SHAP -> point-in-time LLM explanation (pipeline/explain_anomaly.py), for each
currently flagged anomaly. `pipeline/poll_loop.py` handles the "ingest" half.
"""

from typing import Any

from stockwatch.config import get_settings
from stockwatch.detection.isolation_forest import (
    MIN_ROWS_TO_FIT,
    get_anomalies,
    to_feature_array,
)
from stockwatch.detection.model_store import is_model_stale, resolve_ml_detector
from stockwatch.explain.shap_explainer import get_explainer
from stockwatch.features.build_features import build_feature_matrix
from stockwatch.logging_utils import get_logger
from stockwatch.pipeline.explain_anomaly import explain_anomaly

logger = get_logger(__name__)


def detect_and_explain_anomalies(
    top_k_features: int = 5, news_count: int = 5
) -> list[dict[str, Any]]:
    """Returns one {"context": AnomalyContext, "explanation": ExplanationOutput}
    per detected anomaly (empty list if there isn't enough data yet to fit a
    model, or no anomalies were flagged).
    """
    feature_matrix = build_feature_matrix()
    if feature_matrix.height < MIN_ROWS_TO_FIT:
        logger.info(
            "Only %d feature row(s), need %d to fit - skipping detection",
            feature_matrix.height,
            MIN_ROWS_TO_FIT,
        )
        return []

    detector, trained_at = resolve_ml_detector(feature_matrix)
    if trained_at is not None:
        max_age_days = get_settings().model_stale_after_days
        if is_model_stale(trained_at, max_age_days):
            logger.warning(
                "Model trained at %s is stale (> %d days old) - run "
                "train_model.sh to refresh it",
                trained_at,
                max_age_days,
            )

    scored = detector.score(feature_matrix)
    anomalies = get_anomalies(scored)
    logger.info(
        "Detected %d anomal(y/ies) out of %d rows",
        anomalies.height,
        feature_matrix.height,
    )
    if anomalies.is_empty():
        return []

    background = to_feature_array(feature_matrix)
    anomaly_features = to_feature_array(anomalies)
    explainer = get_explainer(detector.model, background)

    return [
        explain_anomaly(
            ticker=row["ticker"],
            window_end=row["window_end"],
            anomaly_score=row["anomaly_score"],
            explainer=explainer,
            feature_row=anomaly_features[i : i + 1],
            top_k_features=top_k_features,
            news_count=news_count,
        )
        for i, row in enumerate(anomalies.iter_rows(named=True))
    ]
