from datetime import UTC, datetime

from sqlalchemy.orm import Session

from stockwatch.db.models import DimSectorIndustry, Watchlist
from stockwatch.pipeline import poll_loop


def _seed_watchlist_and_sectors(db_session: Session) -> None:
    now = datetime.now(UTC)
    tickers = {
        "AAPL": ("Technology", "Consumer Electronics"),
        "MSFT": ("Technology", "Software"),
        "XOM": ("Energy", "Oil & Gas"),
    }
    for ticker, (sector, industry) in tickers.items():
        db_session.add(Watchlist(ticker=ticker, added_at=now, is_active=True))
        db_session.add(
            DimSectorIndustry(
                ticker=ticker,
                sector=sector,
                industry=industry,
                valid_from=now,
                valid_to=None,
                is_current=True,
            )
        )
    db_session.commit()


def test_run_once_ingests_sector_and_industry_news_once_per_distinct_value(
    monkeypatch, db_session: Session
) -> None:
    _seed_watchlist_and_sectors(db_session)

    for name in (
        "ingest_classification",
        "ingest_index_membership",
        "ingest_rating_consensus",
        "ingest_splits",
        "ingest_earnings_estimate",
        "ingest_news",
    ):
        monkeypatch.setattr(poll_loop, name, lambda ticker: None)

    sector_calls: list[str] = []
    industry_calls: list[str] = []
    monkeypatch.setattr(
        poll_loop, "ingest_sector_news", lambda sector: sector_calls.append(sector)
    )
    monkeypatch.setattr(
        poll_loop,
        "ingest_industry_news",
        lambda industry: industry_calls.append(industry),
    )

    poll_loop.run_once()

    assert sorted(sector_calls) == ["Energy", "Technology"]
    assert sorted(industry_calls) == ["Consumer Electronics", "Oil & Gas", "Software"]
