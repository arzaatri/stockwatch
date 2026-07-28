"""Analyst rating (buy/hold/sell) consensus, tracked as an SCD2 dimension."""

from datetime import UTC, datetime

from stockwatch.db.engine import session_scope
from stockwatch.db.models import DimRatingConsensus
from stockwatch.db.scd2 import scd2_upsert
from stockwatch.ingestion.yfinance_client import get_rating_consensus


def ingest_rating_consensus(ticker: str) -> None:
    rating = get_rating_consensus(ticker)
    if rating is None:
        return

    with session_scope() as session:
        scd2_upsert(
            session,
            DimRatingConsensus,
            natural_key={"ticker": ticker},
            attributes={
                "strong_buy": rating.strong_buy,
                "buy": rating.buy,
                "hold": rating.hold,
                "sell": rating.sell,
                "strong_sell": rating.strong_sell,
            },
            observed_at=datetime.now(UTC),
        )
