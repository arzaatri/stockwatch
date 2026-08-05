"""Persistence for the price CDC log. The actual yfinance polling for prices
lives in streaming/quote_producer.py (it also publishes each quote to Kafka
for the Flink windowing job) - this module only owns writing raw_price_ticks.
"""

from datetime import UTC, datetime

from sqlalchemy import func, select

from stockwatch.db.engine import session_scope
from stockwatch.db.models import RawPriceTick
from stockwatch.ingestion.yfinance_client import Quote


def append_price_tick(quote: Quote) -> None:
    with session_scope() as session:
        session.add(
            RawPriceTick(
                ticker=quote.ticker,
                price=quote.price,
                volume=quote.volume,
                open=quote.open,
                high=quote.high,
                low=quote.low,
                close=quote.close,
                as_of=quote.as_of,
                ingested_at=datetime.now(UTC),
            )
        )


def get_latest_tick_at(ticker: str) -> datetime | None:
    """The `as_of` of the most recent price tick already ingested for
    `ticker` - lets backfill fetch only the gap since last poll instead of a
    fixed lookback every time. None if nothing's been ingested yet.
    """
    with session_scope() as session:
        return session.execute(
            select(func.max(RawPriceTick.as_of)).where(RawPriceTick.ticker == ticker)
        ).scalar_one()
