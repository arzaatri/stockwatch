from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from stockwatch.db.models import RawPriceTick, WindowedPriceStats
from stockwatch.ingestion.yfinance_client import Quote
from stockwatch.pipeline import backfill


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (30, "1m"),  # smaller than any valid interval -> falls back to the finest
        (60, "1m"),
        (600, "5m"),  # the default 10-minute granularity
        (1200, "15m"),
        (3600, "1h"),
        (86400, "1d"),
    ],
)
def test_nearest_yf_interval(seconds: int, expected: str) -> None:
    assert backfill.nearest_yf_interval(seconds) == expected


def _quote(ticker: str, price: float, volume: int, minute: int) -> Quote:
    return Quote(
        ticker=ticker,
        price=price,
        volume=volume,
        open=price,
        high=price,
        low=price,
        close=price,
        as_of=datetime(2026, 1, 5, 14, minute, tzinfo=UTC),
    )


def test_backfill_prices_writes_raw_ticks_and_windowed_stats(
    monkeypatch, db_session: Session
) -> None:
    quotes = [
        _quote("AAPL", 100.0, 10, minute=0),
        _quote("AAPL", 102.0, 20, minute=1),
        _quote("AAPL", 101.0, 15, minute=2),
    ]
    monkeypatch.setattr(
        backfill, "get_price_history", lambda ticker, period, interval: quotes
    )

    tick_count = backfill.backfill_prices(period="7d", tickers=["AAPL"])

    assert tick_count == 3
    raw_rows = (
        db_session.execute(select(RawPriceTick).where(RawPriceTick.ticker == "AAPL"))
        .scalars()
        .all()
    )
    assert len(raw_rows) == 3

    stats_rows = (
        db_session.execute(
            select(WindowedPriceStats)
            .where(WindowedPriceStats.ticker == "AAPL")
            .order_by(WindowedPriceStats.window_end)
        )
        .scalars()
        .all()
    )
    # Each quote lands in its own 60s window here, so one stats row per quote.
    assert len(stats_rows) == 3
    assert stats_rows[0].avg_price == 100.0
    assert stats_rows[0].price_zscore == 0.0
    assert stats_rows[1].avg_price == 102.0


def test_backfill_prices_is_idempotent_on_windowed_stats(
    monkeypatch, db_session: Session
) -> None:
    quotes = [_quote("AAPL", 100.0, 10, minute=0)]
    monkeypatch.setattr(
        backfill, "get_price_history", lambda ticker, period, interval: quotes
    )

    backfill.backfill_prices(period="7d", tickers=["AAPL"])
    backfill.backfill_prices(
        period="7d", tickers=["AAPL"]
    )  # re-running should not duplicate

    stats_rows = (
        db_session.execute(
            select(WindowedPriceStats).where(WindowedPriceStats.ticker == "AAPL")
        )
        .scalars()
        .all()
    )
    assert len(stats_rows) == 1
