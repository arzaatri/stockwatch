"""Historical price backfill, so detection has data to work with without
needing quote_producer.py + flink_job.py running continuously in the
background. Metadata backfill needs no separate module - it's just
poll_loop.run_once(), since yfinance only exposes *current* sector/ratings/
news snapshots, not history (splits/earnings already return their full
history on every call).
"""

from collections import defaultdict
from datetime import UTC, datetime, timedelta

from stockwatch.config import get_settings
from stockwatch.ingestion.prices import append_price_tick, get_latest_tick_at
from stockwatch.ingestion.yfinance_client import Quote, get_price_history
from stockwatch.streaming.rolling_stats import (
    WINDOW_SECONDS,
    TickerRunningStats,
    update_running_stats,
)
from stockwatch.streaming.stats_consumer import persist_stats_record
from stockwatch.universe.watchlist import get_active_tickers

# yfinance only accepts these interval strings, not arbitrary second counts.
_VALID_YF_INTERVALS_SECONDS: dict[int, str] = {
    60: "1m",
    120: "2m",
    300: "5m",
    900: "15m",
    1800: "30m",
    3600: "1h",
    86400: "1d",
}


def nearest_yf_interval(seconds: int) -> str:
    """Largest valid yfinance interval <= `seconds` (falls back to the finest
    interval, "1m", if `seconds` is smaller than any valid one).
    """
    candidates = [s for s in _VALID_YF_INTERVALS_SECONDS if s <= seconds]
    chosen = max(candidates) if candidates else min(_VALID_YF_INTERVALS_SECONDS)
    return _VALID_YF_INTERVALS_SECONDS[chosen]


def backfill_prices(
    max_lookback_days: int = 7, tickers: list[str] | None = None
) -> int:
    """Fetches historical bars at the configured live-polling granularity,
    appends them to raw_price_ticks, then replays the same tumbling-window +
    Welford logic the Flink job runs live to populate windowed_price_stats.

    Per ticker, only fetches the gap since that ticker's last ingested tick,
    capped at `max_lookback_days` - safe (and cheap) to call on every
    startup: a fresh ticker gets the full lookback, a ticker that's already
    up to date fetches almost nothing. Returns the number of raw ticks written.
    """
    tickers = tickers or get_active_tickers()
    interval = nearest_yf_interval(get_settings().price_poll_interval_seconds)
    now = datetime.now(UTC)
    earliest_start = now - timedelta(days=max_lookback_days)

    tick_count = 0
    for ticker in tickers:
        last_tick_at = get_latest_tick_at(ticker)
        start = max(earliest_start, last_tick_at) if last_tick_at else earliest_start
        if start >= now:
            continue  # already caught up

        quotes = get_price_history(ticker, start=start, end=now, interval=interval)
        for quote in quotes:
            append_price_tick(quote)
        _replay_windowed_stats(ticker, quotes)
        tick_count += len(quotes)
    return tick_count


def _replay_windowed_stats(ticker: str, quotes: list[Quote]) -> None:
    windows: dict[int, list[Quote]] = defaultdict(list)
    for quote in quotes:
        windows[int(quote.as_of.timestamp()) // WINDOW_SECONDS].append(quote)

    running = TickerRunningStats()
    for bucket in sorted(windows):
        bucket_quotes = windows[bucket]
        avg_price = sum(q.price for q in bucket_quotes) / len(bucket_quotes)
        total_volume = sum(q.volume or 0 for q in bucket_quotes)
        running, zscore = update_running_stats(running, avg_price)
        persist_stats_record(
            {
                "ticker": ticker,
                "window_end_ms": (bucket + 1) * WINDOW_SECONDS * 1000,
                "avg_price": avg_price,
                "total_volume": total_volume,
                "volatility_estimate": running.volatility,
                "price_zscore": zscore,
            }
        )
