"""Orchestrates the "explain" half of a run: build features -> detect
anomalies (MLAnomalyDetector) -> SHAP -> point-in-time LLM explanation
(pipeline/explain_anomaly.py), for each currently flagged anomaly.
`pipeline/poll_loop.py` handles the "ingest" half.
"""

from typing import Any

from stockwatch.detection.isolation_forest import to_feature_array
from stockwatch.detection.ml_detector import MLAnomalyDetector
from stockwatch.explain.shap_explainer import get_explainer
from stockwatch.features.build_features import build_feature_matrix
from stockwatch.logging_utils import get_logger
from stockwatch.pipeline.explain_anomaly import explain_anomaly

logger = get_logger(__name__)

MIN_ROWS_TO_FIT = 10


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

    detector = MLAnomalyDetector()
    anomalies = detector.detect(feature_matrix)
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
