"""Fit/score an IsolationForest over the feature matrix from features/build_features.py."""

import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest

from stockwatch.features.build_features import FEATURE_COLUMNS


def to_feature_array(feature_matrix: pl.DataFrame) -> np.ndarray:
    return feature_matrix.select(FEATURE_COLUMNS).to_numpy()


def fit_isolation_forest(
    feature_matrix: pl.DataFrame, random_state: int = 0
) -> IsolationForest:
    model = IsolationForest(random_state=random_state)
    model.fit(to_feature_array(feature_matrix))
    return model


def score_anomalies(
    model: IsolationForest, feature_matrix: pl.DataFrame
) -> pl.DataFrame:
    """Adds `anomaly_score` (lower = more anomalous) and `is_anomaly` (1 = flagged)."""
    features = to_feature_array(feature_matrix)
    scores = model.decision_function(features)
    predictions = model.predict(features)  # -1 = anomaly, 1 = normal
    return feature_matrix.with_columns(
        pl.Series("anomaly_score", scores),
        pl.Series("is_anomaly", (predictions == -1).astype(int)),
    )


def get_anomalies(scored_feature_matrix: pl.DataFrame) -> pl.DataFrame:
    """Flagged rows only, most anomalous first."""
    return scored_feature_matrix.filter(pl.col("is_anomaly") == 1).sort("anomaly_score")
