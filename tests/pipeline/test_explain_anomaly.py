from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from sqlalchemy.orm import Session

from stockwatch.db.models import Watchlist, WindowedPriceStats
from stockwatch.detection import model_store
from stockwatch.explain.shap_explainer import FeatureAttribution
from stockwatch.ingestion.yfinance_client import (
    NewsItem,
    RatingConsensus,
    SectorIndustry,
)
from stockwatch.llm.schemas import ExplanationOutput, GraphState
from stockwatch.pipeline import explain_anomaly as explain_anomaly_module


class _FakeGraph:
    def invoke(self, state: GraphState) -> dict:
        return {
            "explanation": ExplanationOutput(
                summary="test summary",
                likely_cause_category="unclear",
                confidence="low",
                supporting_evidence=[],
            )
        }


def _patch_common(
    monkeypatch,
    sector_industry: SectorIndustry | None = None,
    rating: RatingConsensus | None = None,
    news: list[NewsItem] | None = None,
) -> None:
    monkeypatch.setattr(
        explain_anomaly_module,
        "get_sector_industry_as_of",
        lambda ticker, as_of: sector_industry,
    )
    monkeypatch.setattr(
        explain_anomaly_module,
        "get_rating_consensus_as_of",
        lambda ticker, as_of: rating,
    )
    monkeypatch.setattr(
        explain_anomaly_module,
        "get_recent_news",
        lambda scope, scope_key, count=5, before=None, since=None: list(news or []),
    )
    monkeypatch.setattr(
        explain_anomaly_module, "build_default_graph", lambda: _FakeGraph()
    )
    monkeypatch.setattr(
        explain_anomaly_module,
        "top_features_for_anomaly",
        lambda explainer, row, k=5: [
            FeatureAttribution(feature="price_zscore", value=3.0, shap_value=-0.1)
        ],
    )


def test_explain_anomaly_builds_context_from_as_of_metadata(monkeypatch) -> None:
    window_end = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)
    sector_industry = SectorIndustry(
        ticker="AAPL", sector="Technology", industry="Consumer Electronics"
    )
    rating = RatingConsensus(
        ticker="AAPL", strong_buy=1, buy=2, hold=3, sell=4, strong_sell=5
    )
    news_item = NewsItem(
        scope="company",
        scope_key="AAPL",
        headline="Apple news",
        link="https://example.com/a",
        publisher="Example",
        published_at=window_end,
    )
    _patch_common(
        monkeypatch, sector_industry=sector_industry, rating=rating, news=[news_item]
    )

    result = explain_anomaly_module.explain_anomaly(
        ticker="AAPL",
        window_end=window_end,
        anomaly_score=-0.5,
        explainer=object(),
        feature_row=np.zeros((1, 1)),
    )

    context = result["context"]
    assert context.ticker == "AAPL"
    assert context.as_of == window_end
    assert context.sector == "Technology"
    assert context.industry == "Consumer Electronics"
    assert context.rating is not None
    assert context.rating.buy == 2
    # company + sector + industry each queried, all returning the fake list
    assert len(context.recent_news) == 3
    assert result["explanation"].summary == "test summary"


def test_explain_anomaly_handles_missing_sector_and_rating(monkeypatch) -> None:
    window_end = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)
    _patch_common(monkeypatch)  # no sector/rating/news data as of this point

    result = explain_anomaly_module.explain_anomaly(
        ticker="AAPL",
        window_end=window_end,
        anomaly_score=-0.5,
        explainer=object(),
        feature_row=np.zeros((1, 1)),
    )

    context = result["context"]
    assert context.sector is None
    assert context.industry is None
    assert context.rating is None
    # only the company-scope query runs when sector/industry are unknown
    assert context.recent_news == []


@pytest.fixture(autouse=True)
def _isolated_models_dir(tmp_path, monkeypatch):
    """Never touch the real project models/ directory - a stray real model
    on disk would otherwise make explain_anomaly_at's detector resolution
    order/environment dependent instead of always fitting fresh here.
    """
    monkeypatch.setattr(model_store, "MODELS_DIR", tmp_path / "models")


def test_explain_anomaly_at_resolves_the_correct_historical_row(
    monkeypatch, db_session: Session
) -> None:
    _patch_common(monkeypatch)
    now = datetime.now(UTC)
    rng = np.random.default_rng(0)

    db_session.add(Watchlist(ticker="AAPL", added_at=now, is_active=True))

    rows = []
    target_window_end = None
    for i in range(30):
        window_end = now - timedelta(minutes=30 - i)
        avg_price = 100.0 + float(rng.normal())
        if i == 25:
            target_window_end = window_end
        rows.append(
            WindowedPriceStats(
                ticker="AAPL",
                window_end=window_end,
                avg_price=avg_price,
                total_volume=1000,
                volatility_estimate=0.1,
                price_zscore=float(rng.normal()),
                ingested_at=now,
            )
        )
    db_session.add_all(rows)
    db_session.commit()
    assert target_window_end is not None

    result = explain_anomaly_module.explain_anomaly_at("AAPL", target_window_end)

    assert result["context"].ticker == "AAPL"
    assert result["context"].as_of == target_window_end
