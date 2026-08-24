"""Population Stability Index (PSI) drift monitoring. Reference distributions
are quantile-binned histograms of the training feature matrix + anomaly-score
distribution, captured once at training time (detection/model_store.py's
save_model() calls build_reference_distribution() and persists the result
alongside the model artifact) - drift is always measured against "what the
model was trained on", not an arbitrary earlier window. build_drift_report()
then buckets a live scored feature matrix into those same bins and compares.

PSI severity thresholds (< 0.1 none, 0.1-0.25 moderate, > 0.25 significant)
are the standard industry rule of thumb for population stability monitoring,
not something tuned for this project's data.
"""

from datetime import datetime

import numpy as np
import polars as pl

from stockwatch.config import get_settings
from stockwatch.detection.feature_schema import FEATURE_COLUMNS
from stockwatch.monitoring.schemas import DriftReport, FeatureDrift, Severity

# Not a real feature column - reference_distribution's key for the trained
# model's anomaly_score distribution (output/prediction drift, alongside the
# per-feature input drift).
SCORE_DISTRIBUTION_KEY = "__anomaly_score__"

_EPSILON = 1e-4  # avoids log(0)/divide-by-zero when a bin's proportion is 0


def _quantile_bin_edges(values: np.ndarray, n_bins: int) -> list[float]:
    """Equal-frequency bin edges over `values`. `np.unique` collapses
    degenerate/duplicate edges that come from heavily-skewed or near-binary
    features (e.g. this project's 0/1 flag features) - such a feature just
    ends up with fewer, still-meaningful bins rather than zero-width ones.

    Deliberately finite (not -inf/+inf) at the outer edges: bin_proportions()
    only ever digitizes against the *interior* edges (edges[1:-1]), so a
    value beyond the training range still lands in the first/last bin either
    way - and finite edges survive a JSON round trip (FastAPI's response
    encoding turns float('inf') into `null`), while -inf/+inf wouldn't.
    """
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    return edges.tolist()


def bin_proportions(values: np.ndarray, bin_edges: list[float]) -> list[float]:
    """Buckets `values` into `bin_edges` (as produced by _quantile_bin_edges)
    and returns each bin's proportion of the total - the shape compute_psi
    compares on both sides (reference vs. current).
    """
    edges = np.asarray(bin_edges)
    n_bins = len(edges) - 1
    if len(values) == 0:
        return [0.0] * n_bins
    bin_indices = np.digitize(values, edges[1:-1])
    counts = np.bincount(bin_indices, minlength=n_bins)
    return (counts / len(values)).tolist()


def compute_psi(reference_proportions: list[float], current_proportions: list[float]) -> float:
    """Population Stability Index: sum((cur - ref) * ln(cur / ref)) over
    bins, epsilon-smoothed so an empty bin on either side doesn't blow up
    log(0)/divide-by-zero.
    """
    ref = np.asarray(reference_proportions) + _EPSILON
    cur = np.asarray(current_proportions) + _EPSILON
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def classify_psi(psi: float) -> Severity:
    settings = get_settings()
    if psi >= settings.drift_psi_significant_threshold:
        return "significant"
    if psi >= settings.drift_psi_moderate_threshold:
        return "moderate"
    return "none"


def build_reference_distribution(
    feature_matrix: pl.DataFrame, scores: np.ndarray
) -> dict[str, dict]:
    """Captured once at training time - see module docstring. `scores` is
    the trained model's own decision_function() output on `feature_matrix`
    (the training batch), i.e. what "normal" anomaly scores looked like when
    the model was fit.
    """
    n_bins = get_settings().drift_reference_bins
    distribution: dict[str, dict] = {}
    for column in [*FEATURE_COLUMNS, SCORE_DISTRIBUTION_KEY]:
        values = (
            scores if column == SCORE_DISTRIBUTION_KEY else feature_matrix[column].to_numpy()
        )
        edges = _quantile_bin_edges(values, n_bins)
        distribution[column] = {
            "bin_edges": edges,
            "bin_proportions": bin_proportions(values, edges),
        }
    return distribution


def _feature_drift(name: str, values: np.ndarray, reference: dict) -> FeatureDrift:
    current_proportions = bin_proportions(values, reference["bin_edges"])
    psi = compute_psi(reference["bin_proportions"], current_proportions)
    return FeatureDrift(feature=name, psi=psi, severity=classify_psi(psi))


def build_drift_report(
    scored_matrix: pl.DataFrame,
    reference_distribution: dict[str, dict],
    model_trained_at: datetime | None,
) -> DriftReport:
    """`scored_matrix` is inference_client.score()'s output (feature columns
    + anomaly_score + is_anomaly) for the current/live data; `reference_distribution`
    comes from the trained model's own metadata (inference_service's
    /model/current, via api/inference_client.get_model_status()).
    """
    feature_drift = [
        _feature_drift(column, scored_matrix[column].to_numpy(), reference_distribution[column])
        for column in FEATURE_COLUMNS
    ]
    score_drift = _feature_drift(
        "anomaly_score",
        scored_matrix["anomaly_score"].to_numpy(),
        reference_distribution[SCORE_DISTRIBUTION_KEY],
    )
    observed_anomaly_rate = (
        float(scored_matrix["is_anomaly"].mean()) if scored_matrix.height else 0.0
    )
    return DriftReport(
        model_trained_at=model_trained_at,
        feature_drift=feature_drift,
        score_drift=score_drift,
        observed_anomaly_rate=observed_anomaly_rate,
        n_rows_evaluated=scored_matrix.height,
    )
