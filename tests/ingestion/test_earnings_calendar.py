from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from stockwatch.db.models import DimEarningsEstimate
from stockwatch.ingestion import earnings_calendar
from stockwatch.ingestion.yfinance_client import EarningsEstimate


def test_ingest_earnings_estimate_writes_a_current_row(
    monkeypatch, db_session: Session
) -> None:
    estimate = EarningsEstimate(
        ticker="AAPL",
        earnings_date=datetime.now(UTC) + timedelta(days=10),
        eps_estimate=1.5,
    )
    monkeypatch.setattr(
        earnings_calendar, "get_earnings_estimate", lambda ticker: estimate
    )

    earnings_calendar.ingest_earnings_estimate("AAPL")

    rows = (
        db_session.execute(
            select(DimEarningsEstimate).where(DimEarningsEstimate.ticker == "AAPL")
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1
    assert rows[0].eps_estimate == 1.5


def test_ingest_earnings_estimate_versions_when_the_date_shifts(
    monkeypatch, db_session: Session
) -> None:
    now = datetime.now(UTC)
    monkeypatch.setattr(
        earnings_calendar,
        "get_earnings_estimate",
        lambda ticker: EarningsEstimate(
            ticker=ticker, earnings_date=now + timedelta(days=10), eps_estimate=1.5
        ),
    )
    earnings_calendar.ingest_earnings_estimate("AAPL")

    monkeypatch.setattr(
        earnings_calendar,
        "get_earnings_estimate",
        lambda ticker: EarningsEstimate(
            ticker=ticker, earnings_date=now + timedelta(days=12), eps_estimate=1.6
        ),
    )
    earnings_calendar.ingest_earnings_estimate("AAPL")

    rows = (
        db_session.execute(
            select(DimEarningsEstimate).where(DimEarningsEstimate.ticker == "AAPL")
        )
        .scalars()
        .all()
    )
    assert len(rows) == 2
    current = [row for row in rows if row.is_current]
    assert len(current) == 1
    assert current[0].eps_estimate == 1.6
