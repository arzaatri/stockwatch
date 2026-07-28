from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from stockwatch.db.models import RawNews
from stockwatch.ingestion import news
from stockwatch.ingestion.yfinance_client import NewsItem


class _FakeNewsSource:
    def __init__(self, items: list[NewsItem]) -> None:
        self._items = items

    def get_news(self, ticker: str, count: int = 10) -> list[NewsItem]:
        return self._items


def test_ingest_news_appends_items(db_session: Session) -> None:
    item = NewsItem(
        ticker="AAPL",
        headline="Apple does a thing",
        link="https://example.com/a",
        publisher="Example",
        published_at=datetime.now(UTC),
    )

    news.ingest_news("AAPL", news_source=_FakeNewsSource([item]))

    rows = (
        db_session.execute(select(RawNews).where(RawNews.ticker == "AAPL"))
        .scalars()
        .all()
    )
    assert len(rows) == 1
    assert rows[0].headline == "Apple does a thing"


def test_ingest_news_dedupes_on_link(db_session: Session) -> None:
    item = NewsItem(
        ticker="AAPL",
        headline="Apple does a thing",
        link="https://example.com/a",
        publisher="Example",
        published_at=datetime.now(UTC),
    )

    news.ingest_news("AAPL", news_source=_FakeNewsSource([item]))
    news.ingest_news("AAPL", news_source=_FakeNewsSource([item]))

    rows = (
        db_session.execute(select(RawNews).where(RawNews.ticker == "AAPL"))
        .scalars()
        .all()
    )
    assert len(rows) == 1
