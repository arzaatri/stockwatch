from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from stockwatch.db.models import RawSplit
from stockwatch.ingestion import splits
from stockwatch.ingestion.yfinance_client import SplitEvent


def test_ingest_splits_appends_new_events(monkeypatch, db_session: Session) -> None:
    event = SplitEvent(
        ticker="AAPL", split_date=datetime(2020, 8, 31, tzinfo=UTC), numerator=4.0
    )
    monkeypatch.setattr(splits, "get_splits", lambda ticker: [event])

    splits.ingest_splits("AAPL")

    rows = (
        db_session.execute(select(RawSplit).where(RawSplit.ticker == "AAPL"))
        .scalars()
        .all()
    )
    assert len(rows) == 1
    assert rows[0].numerator == 4.0


def test_ingest_splits_dedupes_on_reingestion(monkeypatch, db_session: Session) -> None:
    event = SplitEvent(
        ticker="AAPL", split_date=datetime(2020, 8, 31, tzinfo=UTC), numerator=4.0
    )
    monkeypatch.setattr(splits, "get_splits", lambda ticker: [event])

    splits.ingest_splits("AAPL")
    splits.ingest_splits("AAPL")  # yfinance returns the full history every call

    rows = (
        db_session.execute(select(RawSplit).where(RawSplit.ticker == "AAPL"))
        .scalars()
        .all()
    )
    assert len(rows) == 1
