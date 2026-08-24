"""Point-in-time anomaly explanation - the one path both "explain what's
anomalous right now" (pipeline/detect_and_explain.py) and "explain this
specific historical point the user clicked" (the dashboard) go through, so
there's exactly one way metadata gets gathered for an explanation, not two
implementations that could drift (one point-in-time-correct, one not).
"""

from datetime import datetime, timedelta
from typing import Any

import polars as pl

from stockwatch.api import inference_client
from stockwatch.explain.shap_explainer import FeatureAttribution
from stockwatch.features.build_features import build_feature_matrix
from stockwatch.ingestion.classification import get_sector_industry_as_of
from stockwatch.ingestion.news import get_recent_news
from stockwatch.ingestion.ratings import get_rating_consensus_as_of
from stockwatch.ingestion.yfinance_client import NewsItem
from stockwatch.llm.graph import build_default_graph
from stockwatch.llm.schemas import AnomalyContext, GraphState
from stockwatch.logging_utils import get_logger

logger = get_logger(__name__)

NEWS_LOOKBACK = timedelta(hours=24)


def explain_anomaly(
    ticker: str,
    window_end: datetime,
    anomaly_score: float,
    top_features: list[FeatureAttribution],
    news_count: int = 5,
) -> dict[str, Any]:
    """Builds an AnomalyContext using metadata as of `window_end` (not "now")
    and invokes the LLM graph. `top_features` comes from the inference
    service (api/inference_client.py) - this function never touches the
    model itself, callers own getting a SHAP attribution first.
    """
    logger.info("Explaining %s @ %s (score=%.4f)", ticker, window_end, anomaly_score)
    sector_industry = get_sector_industry_as_of(ticker, window_end)
    rating = get_rating_consensus_as_of(ticker, window_end)

    context = AnomalyContext(
        ticker=ticker,
        as_of=window_end,
        anomaly_score=anomaly_score,
        top_features=top_features,
        sector=sector_industry.sector if sector_industry else None,
        industry=sector_industry.industry if sector_industry else None,
        recent_news=_gather_recent_news(
            ticker,
            sector_industry.sector if sector_industry else None,
            sector_industry.industry if sector_industry else None,
            window_end,
            news_count,
        ),
        rating=rating,
    )
    graph = build_default_graph()
    result_state = graph.invoke(GraphState(context=context))
    logger.info("Explanation ready for %s @ %s", ticker, window_end)
    return {"context": context, "explanation": result_state["explanation"]}


def explain_anomaly_at(
    ticker: str,
    window_end: datetime,
    top_k_features: int = 5,
    news_count: int = 5,
) -> dict[str, Any]:
    """Dashboard entrypoint: explain one specific (ticker, window_end) point
    the user clicked. Scores the whole current matrix via the inference
    service (same call detect_anomalies() makes) and picks out that row -
    it must be one the current model actually flags as anomalous, since
    that's the only case the inference service computes a SHAP attribution
    for (and the only case the dashboard ever offers a click target for).
    """
    feature_matrix = build_feature_matrix()
    result = inference_client.score(feature_matrix, top_k_features=top_k_features)

    matching = result.scored_matrix.filter(
        (pl.col("ticker") == ticker) & (pl.col("window_end") == window_end)
    )
    if matching.is_empty():
        logger.warning("No feature row found for %s at %s", ticker, window_end)
        raise ValueError(f"No feature row found for {ticker} at {window_end}")

    key = (ticker, window_end)
    if key not in result.top_features_by_key:
        logger.warning(
            "%s at %s isn't flagged anomalous by the current model - nothing to explain",
            ticker,
            window_end,
        )
        raise ValueError(f"{ticker} at {window_end} is not currently flagged as an anomaly")

    anomaly_score = matching.row(0, named=True)["anomaly_score"]
    return explain_anomaly(
        ticker=ticker,
        window_end=window_end,
        anomaly_score=anomaly_score,
        top_features=result.top_features_by_key[key],
        news_count=news_count,
    )


def _gather_recent_news(
    ticker: str,
    sector: str | None,
    industry: str | None,
    window_end: datetime,
    count: int,
) -> list[NewsItem]:
    """The 24h before `window_end`, as of `window_end` - not an unbounded
    "most recent N ever", which would leak articles from after the anomaly
    (or long before it) into what's supposed to be a point-in-time snapshot.
    """
    since = window_end - NEWS_LOOKBACK
    news = get_recent_news(
        "company", ticker, count=count, before=window_end, since=since
    )
    if sector:
        news += get_recent_news(
            "sector", sector, count=count, before=window_end, since=since
        )
    if industry:
        news += get_recent_news(
            "industry", industry, count=count, before=window_end, since=since
        )
    return news
