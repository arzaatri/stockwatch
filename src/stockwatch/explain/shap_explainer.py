"""SHAP-based explanation of individual anomalies. `TreeExplainer` is the
primary path (IsolationForest is tree-based); if it fails to construct for any
reason, fall back to `KernelExplainer` against a background sample. Either
way, callers only ever see `top_features_for_anomaly`.
"""

from typing import Any

import numpy as np
import shap
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest

from stockwatch.features.build_features import FEATURE_COLUMNS


class FeatureAttribution(BaseModel):
    feature: str
    value: float
    shap_value: float


def get_explainer(model: IsolationForest, background: np.ndarray) -> Any:
    try:
        return shap.TreeExplainer(model)
    except Exception:
        sample_size = min(100, len(background))
        return shap.KernelExplainer(
            model.decision_function, shap.sample(background, sample_size)
        )


def top_features_for_anomaly(
    explainer: Any, row: np.ndarray, k: int = 5
) -> list[FeatureAttribution]:
    """`row` is a single-row 2D array (1, n_features) matching FEATURE_COLUMNS order."""
    shap_values = np.asarray(explainer.shap_values(row)).reshape(-1)
    row_values = row.reshape(-1)
    top_indices = np.argsort(-np.abs(shap_values))[:k]
    return [
        FeatureAttribution(
            feature=FEATURE_COLUMNS[i],
            value=float(row_values[i]),
            shap_value=float(shap_values[i]),
        )
        for i in top_indices
    ]
