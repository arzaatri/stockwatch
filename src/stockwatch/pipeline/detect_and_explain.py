"""Orchestrates the "explain" half of a run: build features -> call the
inference microservice to score + get SHAP top features -> a point-in-time
LLM explanation (pipeline/explain_anomaly.py), for each currently flagged
anomaly. `pipeline/poll_loop.py` handles the "ingest" half.
"""

from typing import Any

import polars as pl

from stockwatch.api import inference_client
from stockwatch.detection.isolation_forest import MIN_ROWS_TO_FIT
from stockwatch.features.build_features import build_feature_matrix
from stockwatch.logging_utils import get_logger
from stockwatch.pipeline.explain_anomaly import explain_anomaly

logger = get_logger(__name__)


def detect_anomalies(
    tickers: list[str] | None = None, top_k_features: int = 5
) -> list[dict[str, Any]]:
    """Scores the current feature matrix (all active tickers, or just
    `tickers` if given) via the inference service. Returns one {"ticker",
    "window_end", "anomaly_score", "top_features"} per flagged anomaly
    (empty if there isn't enough data yet to score meaningfully, or nothing
    was flagged). No LLM call - see detect_and_explain_anomalies() for that.
    """
    feature_matrix = build_feature_matrix(tickers)
    if feature_matrix.height < MIN_ROWS_TO_FIT:
        logger.info(
            "Only %d feature row(s), need %d to detect - skipping",
            feature_matrix.height,
            MIN_ROWS_TO_FIT,
        )
        return []

    result = inference_client.score(feature_matrix, top_k_features=top_k_features)
    anomalies = result.scored_matrix.filter(pl.col("is_anomaly") == 1).sort(
        "anomaly_score"
    )
    logger.info(
        "Detected %d anomal(y/ies) out of %d rows",
        anomalies.height,
        feature_matrix.height,
    )
    return [
        {
            "ticker": row["ticker"],
            "window_end": row["window_end"],
            "anomaly_score": row["anomaly_score"],
            "top_features": result.top_features_by_key[
                (row["ticker"], row["window_end"])
            ],
        }
        for row in anomalies.iter_rows(named=True)
    ]


def detect_and_explain_anomalies(
    top_k_features: int = 5, news_count: int = 5
) -> list[dict[str, Any]]:
    """Returns one {"context": AnomalyContext, "explanation": ExplanationOutput}
    per detected anomaly (empty list if there isn't enough data yet to fit a
    model, or no anomalies were flagged).
    """
    return [
        explain_anomaly(
            ticker=anomaly["ticker"],
            window_end=anomaly["window_end"],
            anomaly_score=anomaly["anomaly_score"],
            top_features=anomaly["top_features"],
            news_count=news_count,
        )
        for anomaly in detect_anomalies(top_k_features=top_k_features)
    ]
