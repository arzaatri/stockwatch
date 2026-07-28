"""Orchestrates the "explain" half of a run: build features -> fit/score
IsolationForest -> SHAP -> LangGraph LLM explanation, for each currently
flagged anomaly. `pipeline/poll_loop.py` handles the "ingest" half.
"""

from typing import Any

from stockwatch.detection.isolation_forest import (
    fit_isolation_forest,
    get_anomalies,
    score_anomalies,
    to_feature_array,
)
from stockwatch.explain.shap_explainer import get_explainer, top_features_for_anomaly
from stockwatch.features.build_features import build_feature_matrix
from stockwatch.ingestion.news_source import NewsSource, YFinanceNewsSource
from stockwatch.ingestion.yfinance_client import get_rating_consensus
from stockwatch.llm.graph import build_default_graph
from stockwatch.llm.schemas import AnomalyContext, GraphState

MIN_ROWS_TO_FIT = 10


def detect_and_explain_anomalies(
    top_k_features: int = 5, news_count: int = 5, news_source: NewsSource | None = None
) -> list[dict[str, Any]]:
    """Returns one {"context": AnomalyContext, "explanation": ExplanationOutput}
    per detected anomaly (empty list if there isn't enough data yet to fit a
    model, or no anomalies were flagged).
    """
    feature_matrix = build_feature_matrix()
    if feature_matrix.height < MIN_ROWS_TO_FIT:
        return []

    model = fit_isolation_forest(feature_matrix)
    scored = score_anomalies(model, feature_matrix)
    anomalies = get_anomalies(scored)
    if anomalies.is_empty():
        return []

    background = to_feature_array(feature_matrix)
    anomaly_features = to_feature_array(anomalies)
    explainer = get_explainer(model, background)
    graph = build_default_graph()
    source = news_source or YFinanceNewsSource()

    results = []
    for i, row in enumerate(anomalies.iter_rows(named=True)):
        top_features = top_features_for_anomaly(
            explainer, anomaly_features[i : i + 1], k=top_k_features
        )
        context = AnomalyContext(
            ticker=row["ticker"],
            as_of=row["window_end"],
            anomaly_score=row["anomaly_score"],
            top_features=top_features,
            sector=row["sector"],
            industry=row["industry"],
            recent_news=source.get_news(row["ticker"], count=news_count),
            rating=get_rating_consensus(row["ticker"]),
        )
        result_state = graph.invoke(GraphState(context=context))
        results.append({"context": context, "explanation": result_state["explanation"]})

    return results
