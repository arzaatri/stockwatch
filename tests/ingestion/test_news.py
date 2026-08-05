from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from stockwatch.db.models import RawNews
from stockwatch.ingestion import news, news_api_client
from stockwatch.ingestion.yfinance_client import NewsItem


class _FakeNewsSource:
    def __init__(self, items: list[NewsItem]) -> None:
        self._items = items

    def get_news(self, ticker: str, count: int = 10) -> list[NewsItem]:
        return self._items


def test_ingest_news_appends_items(db_session: Session) -> None:
    item = NewsItem(
        scope="company",
        scope_key="AAPL",
        headline="Apple does a thing",
        link="https://example.com/a",
        publisher="Example",
        published_at=datetime.now(UTC),
    )

    news.ingest_news("AAPL", news_source=_FakeNewsSource([item]))

    rows = (
        db_session.execute(
            select(RawNews).where(
                RawNews.scope == "company", RawNews.scope_key == "AAPL"
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1
    assert rows[0].headline == "Apple does a thing"


def test_ingest_news_dedupes_on_link(db_session: Session) -> None:
    item = NewsItem(
        scope="company",
        scope_key="AAPL",
        headline="Apple does a thing",
        link="https://example.com/a",
        publisher="Example",
        published_at=datetime.now(UTC),
    )

    news.ingest_news("AAPL", news_source=_FakeNewsSource([item]))
    news.ingest_news("AAPL", news_source=_FakeNewsSource([item]))

    rows = (
        db_session.execute(
            select(RawNews).where(
                RawNews.scope == "company", RawNews.scope_key == "AAPL"
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1


def _keyword_item(scope: str, scope_key: str, link: str) -> NewsItem:
    return NewsItem(
        scope=scope,
        scope_key=scope_key,
        headline=f"{scope_key} headline",
        link=link,
        publisher="Example",
        snippet="a short summary",
        published_at=datetime.now(UTC),
    )


def test_ingest_sector_news_stores_sector_scoped_rows(
    monkeypatch, db_session: Session
) -> None:
    item = _keyword_item("sector", "Technology", "https://example.com/sector-a")
    monkeypatch.setattr(
        news_api_client, "search_news", lambda query, scope, count=10: [item]
    )

    news.ingest_sector_news("Technology")

    rows = (
        db_session.execute(
            select(RawNews).where(
                RawNews.scope == "sector", RawNews.scope_key == "Technology"
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1
    assert rows[0].snippet == "a short summary"


def test_ingest_industry_news_stores_industry_scoped_rows(
    monkeypatch, db_session: Session
) -> None:
    item = _keyword_item(
        "industry", "Consumer Electronics", "https://example.com/industry-a"
    )
    monkeypatch.setattr(
        news_api_client, "search_news", lambda query, scope, count=10: [item]
    )

    news.ingest_industry_news("Consumer Electronics")

    rows = (
        db_session.execute(
            select(RawNews).where(
                RawNews.scope == "industry", RawNews.scope_key == "Consumer Electronics"
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1


def test_ingest_sector_news_dedupes_on_scope_key_and_link(
    monkeypatch, db_session: Session
) -> None:
    item = _keyword_item("sector", "Technology", "https://example.com/sector-a")
    monkeypatch.setattr(
        news_api_client, "search_news", lambda query, scope, count=10: [item]
    )

    news.ingest_sector_news("Technology")
    news.ingest_sector_news("Technology")

    rows = (
        db_session.execute(select(RawNews).where(RawNews.scope == "sector"))
        .scalars()
        .all()
    )
    assert len(rows) == 1


def test_get_recent_news_filters_by_scope_and_orders_newest_first(
    db_session: Session,
) -> None:
    now = datetime.now(UTC)
    db_session.add_all(
        [
            RawNews(
                scope="company",
                scope_key="AAPL",
                headline="Older company news",
                link="https://example.com/company-old",
                publisher="Example",
                published_at=now - timedelta(days=1),
                ingested_at=now,
            ),
            RawNews(
                scope="company",
                scope_key="AAPL",
                headline="Newer company news",
                link="https://example.com/company-new",
                publisher="Example",
                published_at=now,
                ingested_at=now,
            ),
            RawNews(
                scope="sector",
                scope_key="Technology",
                headline="Sector news",
                link="https://example.com/sector",
                publisher="Example",
                published_at=now,
                ingested_at=now,
            ),
        ]
    )
    db_session.commit()

    items = news.get_recent_news("company", "AAPL", count=5)

    assert [item.headline for item in items] == [
        "Newer company news",
        "Older company news",
    ]


def test_get_recent_news_respects_count_limit(db_session: Session) -> None:
    now = datetime.now(UTC)
    db_session.add_all(
        [
            RawNews(
                scope="company",
                scope_key="AAPL",
                headline=f"headline {i}",
                link=f"https://example.com/{i}",
                publisher="Example",
                published_at=now - timedelta(minutes=i),
                ingested_at=now,
            )
            for i in range(5)
        ]
    )
    db_session.commit()

    items = news.get_recent_news("company", "AAPL", count=2)

    assert len(items) == 2


def test_get_recent_news_before_and_since_bound_the_window(db_session: Session) -> None:
    anchor = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)
    db_session.add_all(
        [
            RawNews(
                scope="company",
                scope_key="AAPL",
                headline="Too early",
                link="https://example.com/too-early",
                publisher="Example",
                published_at=anchor - timedelta(hours=25),
                ingested_at=anchor,
            ),
            RawNews(
                scope="company",
                scope_key="AAPL",
                headline="Within window",
                link="https://example.com/within-window",
                publisher="Example",
                published_at=anchor - timedelta(hours=1),
                ingested_at=anchor,
            ),
            RawNews(
                scope="company",
                scope_key="AAPL",
                headline="Too late",
                link="https://example.com/too-late",
                publisher="Example",
                published_at=anchor + timedelta(hours=1),
                ingested_at=anchor,
            ),
        ]
    )
    db_session.commit()

    items = news.get_recent_news(
        "company",
        "AAPL",
        count=10,
        before=anchor,
        since=anchor - timedelta(hours=24),
    )

    assert [item.headline for item in items] == ["Within window"]
