"""News headlines: an append-only CDC log, sourced through the pluggable
NewsSource protocol (news_source.py) so NewsAPI.org can be swapped/added later.
"""

from datetime import UTC, datetime

from sqlalchemy.dialects.postgresql import insert as pg_insert

from stockwatch.db.engine import session_scope
from stockwatch.db.models import RawNews
from stockwatch.ingestion.news_source import NewsSource, YFinanceNewsSource


def ingest_news(
    ticker: str, news_source: NewsSource | None = None, count: int = 10
) -> None:
    source = news_source or YFinanceNewsSource()
    items = source.get_news(ticker, count=count)
    if not items:
        return

    ingested_at = datetime.now(UTC)
    with session_scope() as session:
        for item in items:
            stmt = (
                pg_insert(RawNews)
                .values(
                    ticker=item.ticker,
                    headline=item.headline,
                    link=item.link,
                    publisher=item.publisher,
                    published_at=item.published_at,
                    ingested_at=ingested_at,
                )
                .on_conflict_do_nothing(index_elements=["ticker", "link"])
            )
            session.execute(stmt)
