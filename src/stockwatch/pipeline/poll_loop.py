"""Polls slow-changing metadata (sector, index membership, ratings, splits,
earnings, news) at a coarser interval than the real-time price stream, which
runs separately via streaming/quote_producer.py + flink_job.py. No scheduler
framework - a plain while+sleep loop; `run_once` also backs cron-style
invocation via the CLI, so one code path serves both usages.
"""

import time

from sqlalchemy import select

from stockwatch.db.engine import session_scope
from stockwatch.db.models import DimSectorIndustry
from stockwatch.ingestion.classification import ingest_classification
from stockwatch.ingestion.earnings_calendar import ingest_earnings_estimate
from stockwatch.ingestion.index_membership import ingest_index_membership
from stockwatch.ingestion.news import (
    ingest_industry_news,
    ingest_news,
    ingest_sector_news,
)
from stockwatch.ingestion.ratings import ingest_rating_consensus
from stockwatch.ingestion.splits import ingest_splits
from stockwatch.logging_utils import get_logger
from stockwatch.universe.watchlist import get_active_tickers

logger = get_logger(__name__)


def run_once() -> None:
    tickers = get_active_tickers()
    logger.info("Starting metadata poll cycle for %d ticker(s)", len(tickers))
    for ticker in tickers:
        ingest_classification(ticker)
        ingest_index_membership(ticker)
        ingest_rating_consensus(ticker)
        ingest_splits(ticker)
        ingest_earnings_estimate(ticker)
        ingest_news(ticker)

    # One NewsAPI call per distinct sector/industry, not per ticker - keeps
    # usage well under the free tier's daily cap even as the watchlist grows.
    sectors = _distinct_current_sectors()
    industries = _distinct_current_industries()
    logger.info(
        "Polling news for %d sector(s) and %d industr(y/ies)",
        len(sectors),
        len(industries),
    )
    for sector in sectors:
        ingest_sector_news(sector)
    for industry in industries:
        ingest_industry_news(industry)
    logger.info("Metadata poll cycle complete")


def _distinct_current_sectors() -> list[str]:
    with session_scope() as session:
        stmt = (
            select(DimSectorIndustry.sector)
            .where(DimSectorIndustry.is_current.is_(True))
            .distinct()
        )
        return [sector for sector in session.execute(stmt).scalars() if sector]


def _distinct_current_industries() -> list[str]:
    with session_scope() as session:
        stmt = (
            select(DimSectorIndustry.industry)
            .where(DimSectorIndustry.is_current.is_(True))
            .distinct()
        )
        return [industry for industry in session.execute(stmt).scalars() if industry]


def run_forever(interval_seconds: int) -> None:
    while True:
        run_once()
        time.sleep(interval_seconds)
