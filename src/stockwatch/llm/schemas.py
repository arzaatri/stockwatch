"""Pydantic types for the anomaly-explanation LangGraph: input context, the
graph's state, and the structured output schema the LLM must produce.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from stockwatch.explain.shap_explainer import FeatureAttribution
from stockwatch.ingestion.yfinance_client import NewsItem, RatingConsensus


class AnomalyContext(BaseModel):
    ticker: str
    as_of: datetime
    anomaly_score: float
    top_features: list[FeatureAttribution]
    sector: str | None = None
    industry: str | None = None
    recent_news: list[NewsItem] = Field(default_factory=list)
    rating: RatingConsensus | None = None


class ExplanationOutput(BaseModel):
    summary: str
    likely_cause_category: Literal[
        "earnings", "rating_change", "split", "news_driven", "sector_wide", "unclear"
    ]
    confidence: Literal["low", "medium", "high"]
    supporting_evidence: list[str]


class GraphState(BaseModel):
    context: AnomalyContext
    prompt: str | None = None
    explanation: ExplanationOutput | None = None
