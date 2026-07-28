import numpy as np
import polars as pl

from stockwatch.detection.isolation_forest import fit_isolation_forest, to_feature_array
from stockwatch.explain.shap_explainer import (
    FeatureAttribution,
    get_explainer,
    top_features_for_anomaly,
)
from stockwatch.features.build_features import FEATURE_COLUMNS


def _synthetic_matrix(n: int = 60, seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    data = {column: rng.normal(size=n).tolist() for column in FEATURE_COLUMNS}
    data["price_zscore"][0] = 50.0
    return pl.DataFrame(data)


def test_top_features_for_anomaly_returns_k_attributions() -> None:
    matrix = _synthetic_matrix()
    model = fit_isolation_forest(matrix)
    features_array = to_feature_array(matrix)
    explainer = get_explainer(model, features_array)

    attributions = top_features_for_anomaly(explainer, features_array[:1], k=3)

    assert len(attributions) == 3
    assert all(isinstance(a, FeatureAttribution) for a in attributions)
    assert all(a.feature in FEATURE_COLUMNS for a in attributions)


def test_top_features_are_sorted_by_absolute_shap_value() -> None:
    matrix = _synthetic_matrix()
    model = fit_isolation_forest(matrix)
    features_array = to_feature_array(matrix)
    explainer = get_explainer(model, features_array)

    attributions = top_features_for_anomaly(explainer, features_array[:1], k=5)

    magnitudes = [abs(a.shap_value) for a in attributions]
    assert magnitudes == sorted(magnitudes, reverse=True)
