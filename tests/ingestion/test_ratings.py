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
