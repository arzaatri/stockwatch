from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from stockwatch.db.models import DimRatingConsensus
from stockwatch.ingestion import ratings
from stockwatch.ingestion.yfinance_client import RatingConsensus


def test_ingest_rating_consensus_writes_a_current_row(
    monkeypatch, db_session: Session
) -> None:
    monkeypatch.setattr(
        ratings,
        "get_rating_consensus",
        lambda ticker: RatingConsensus(
            ticker=ticker, strong_buy=5, buy=10, hold=3, sell=1, strong_sell=0
        ),
    )

    ratings.ingest_rating_consensus("AAPL")

    rows = (
        db_session.execute(
            select(DimRatingConsensus).where(DimRatingConsensus.ticker == "AAPL")
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1
    assert rows[0].buy == 10
    assert rows[0].is_current is True


def test_ingest_rating_consensus_is_a_noop_when_missing(
    monkeypatch, db_session: Session
) -> None:
    monkeypatch.setattr(ratings, "get_rating_consensus", lambda ticker: None)

    ratings.ingest_rating_consensus("AAPL")

    rows = (
        db_session.execute(
            select(DimRatingConsensus).where(DimRatingConsensus.ticker == "AAPL")
        )
        .scalars()
        .all()
    )
    assert rows == []


def test_get_rating_consensus_as_of_returns_the_version_current_then(
    db_session: Session,
) -> None:
    t1 = datetime.now(UTC)
    t2 = t1 + timedelta(days=10)
    db_session.add(
        DimRatingConsensus(
            ticker="AAPL",
            strong_buy=1,
            buy=2,
            hold=3,
            sell=4,
            strong_sell=5,
            valid_from=t1,
            valid_to=t2,
            is_current=False,
        )
    )
    db_session.add(
        DimRatingConsensus(
            ticker="AAPL",
            strong_buy=9,
            buy=9,
            hold=0,
            sell=0,
            strong_sell=0,
            valid_from=t2,
            valid_to=None,
            is_current=True,
        )
    )
    db_session.commit()

    result = ratings.get_rating_consensus_as_of("AAPL", t1 + timedelta(days=1))

    assert result is not None
    assert result.buy == 2
