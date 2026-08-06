"""Fit/score an IsolationForest over the feature matrix from features/build_features.py.

Usage follows scikit-learn's standard outlier-detection pattern: fit and
score the *same* batch of data (this is unsupervised outlier detection, not
supervised train/test - there's no "leakage" concern the way there would be
for a predictive model). IsolationForest also needs no feature scaling: it
splits on a random threshold within each feature's own observed range at
each tree node, so it's naturally invariant to per-feature scale - the real
scale hazard for this project was raw cross-ticker price/volume levels
(fixed by using price_zscore/volume_zscore instead, see build_features.py).
"""

import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest

from stockwatch.features.build_features import FEATURE_COLUMNS

# Below this many rows, IsolationForest's splits are too few to mean anything -
# fail clearly rather than silently return meaningless scores.
MIN_ROWS_TO_FIT = 10


def to_feature_array(feature_matrix: pl.DataFrame) -> np.ndarray:
    array = feature_matrix.select(FEATURE_COLUMNS).to_numpy()
    if np.isnan(array).any():
        nan_columns = [
            column
            for column, has_nan in zip(
                FEATURE_COLUMNS, np.isnan(array).any(axis=0), strict=True
            )
            if has_nan
        ]
        raise ValueError(
            f"Feature matrix has NaN in column(s) {nan_columns} - "
            "build_feature_matrix() should have filled these; check its fill_null calls."
        )
    return array


def fit_isolation_forest(
    feature_matrix: pl.DataFrame,
    random_state: int = 0,
    contamination: float | str = "auto",
) -> IsolationForest:
    """`contamination` is the expected proportion of anomalies in the data -
    "auto" (sklearn's default heuristic) unless the caller wants to tune
    detection sensitivity explicitly, e.g. contamination=0.05 for "flag
    roughly the most unusual 5%".
    """
    if feature_matrix.height < MIN_ROWS_TO_FIT:
        raise ValueError(
            f"Need at least {MIN_ROWS_TO_FIT} rows to fit IsolationForest "
            f"meaningfully, got {feature_matrix.height}"
        )
    model = IsolationForest(random_state=random_state, contamination=contamination)
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
