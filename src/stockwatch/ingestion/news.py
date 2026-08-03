"""News headlines: an append-only CDC log, one row per (scope, scope_key, link).
Company-scope news is sourced through the pluggable NewsSource protocol
(news_source.py, currently yfinance); sector/industry-scope news has no
yfinance equivalent, so it goes through news_api_client.py (NewsAPI.org)
directly. `get_recent_news` is the read side - the LLM explanation step reads
from this log rather than re-fetching live, so explanations only ever reflect
news that's actually been ingested (the CDC log is the source of truth, not
a passthrough cache).
"""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from stockwatch.db.engine import session_scope
from stockwatch.db.models import RawNews
from stockwatch.ingestion import news_api_client
from stockwatch.ingestion.news_source import NewsSource, YFinanceNewsSource
from stockwatch.ingestion.yfinance_client import NewsItem, NewsScope


def ingest_news(
    ticker: str, news_source: NewsSource | None = None, count: int = 10
) -> None:
    source = news_source or YFinanceNewsSource()
    _store_news_items(source.get_news(ticker, count=count))


def ingest_sector_news(sector: str, count: int = 10) -> None:
    _store_news_items(news_api_client.search_news(sector, scope="sector", count=count))


def ingest_industry_news(industry: str, count: int = 10) -> None:
    _store_news_items(
        news_api_client.search_news(industry, scope="industry", count=count)
    )


def get_recent_news(scope: NewsScope, scope_key: str, count: int = 5) -> list[NewsItem]:
    with session_scope() as session:
        rows = (
            session.execute(
                select(RawNews)
                .where(RawNews.scope == scope, RawNews.scope_key == scope_key)
                .order_by(RawNews.published_at.desc())
                .limit(count)
            )
            .scalars()
            .all()
        )
        return [
            NewsItem(
                scope=row.scope,
                scope_key=row.scope_key,
                headline=row.headline,
                link=row.link,
                publisher=row.publisher,
                snippet=row.snippet,
                published_at=row.published_at,
            )
            for row in rows
        ]


def _store_news_items(items: list[NewsItem]) -> None:
    if not items:
        return

    ingested_at = datetime.now(UTC)
    with session_scope() as session:
        for item in items:
            stmt = (
                pg_insert(RawNews)
                .values(
                    scope=item.scope,
                    scope_key=item.scope_key,
                    headline=item.headline,
                    link=item.link,
                    publisher=item.publisher,
                    snippet=item.snippet,
                    published_at=item.published_at,
                    ingested_at=ingested_at,
                )
                .on_conflict_do_nothing(index_elements=["scope", "scope_key", "link"])
            )
            session.execute(stmt)
