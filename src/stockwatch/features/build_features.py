"""Feature engineering: joins the real-time price-stats stream
(windowed_price_stats, produced by the Flink job) with slow-changing
dimension/CDC context into a numeric feature matrix for anomaly detection.

Only FEATURE_COLUMNS is used as the numeric matrix passed to IsolationForest/
SHAP (detection.py does the polars -> numpy conversion). `sector`/`industry`
are kept alongside as metadata for the LLM explanation step, not fed to the
model - a categorical field with the plan's 20-ticker scale isn't worth the
one-hot-encoding complexity yet.

Deliberately excluded from FEATURE_COLUMNS: raw `avg_price`/`total_volume`.
Both are on absolute scales that vary by orders of magnitude across tickers
(a $5 stock vs. a $500 stock; a thinly-traded name vs. a heavily-traded one),
so a model shared across the whole watchlist would partly learn "which
ticker is this" rather than "is this unusual for its own history." Both
stay as matrix columns (the dashboard charts `avg_price` directly) but are
fed to the model only via their ticker-relative forms - `price_zscore` and
`volume_zscore`, both computed via the same per-ticker Welford running stats
(streaming/rolling_stats.py) that also drive `volatility_estimate`.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from stockwatch.db.engine import session_scope
from stockwatch.db.models import (
    DimEarningsEstimate,
    DimRatingConsensus,
    DimSectorIndustry,
    RawNews,
    RawSplit,
    WindowedPriceStats,
)
from stockwatch.universe.watchlist import get_active_tickers

FEATURE_COLUMNS = [
    "volatility_estimate",
    "price_zscore",
    "volume_zscore",
    "rating_buy_ratio",
    "rating_sell_ratio",
    "split_recent_flag",
    "news_count_recent",
    "earnings_near_flag",
    "earnings_estimate_shifted_flag",
]
METADATA_COLUMNS = ["ticker", "window_end", "sector", "industry"]

NEWS_LOOKBACK_DAYS = 7
SPLIT_LOOKBACK_DAYS = 30
EARNINGS_NEAR_DAYS = 14
ESTIMATE_SHIFT_LOOKBACK_DAYS = 14

_EMPTY_SCHEMA = {
    "ticker": pl.Utf8,
    "window_end": pl.Datetime("us", "UTC"),
    "avg_price": pl.Float64,
    "total_volume": pl.Int64,
} | {column: pl.Float64 for column in FEATURE_COLUMNS}


def build_feature_matrix(tickers: list[str] | None = None) -> pl.DataFrame:
    """One row per (ticker, window_end) from windowed_price_stats, enriched
    with each ticker's *current* slow-changing context. Empty (correctly
    typed) if there are no price-stats rows yet for these tickers.
    """
    tickers = tickers or get_active_tickers()
    if not tickers:
        return pl.DataFrame(schema=_EMPTY_SCHEMA)

    now = datetime.now(UTC)
    with session_scope() as session:
        price_stats = _load_price_stats(session, tickers)
        if price_stats.is_empty():
            return pl.DataFrame(schema=_EMPTY_SCHEMA)

        feature_matrix = (
            price_stats.join(
                _load_sector_industry(session, tickers), on="ticker", how="left"
            )
            .join(_load_ratings(session, tickers), on="ticker", how="left")
            .join(_load_split_flags(session, tickers, now), on="ticker", how="left")
            .join(_load_news_counts(session, tickers, now), on="ticker", how="left")
            .join(_load_earnings_flags(session, tickers, now), on="ticker", how="left")
        )

    return feature_matrix.with_columns(
        # volatility_estimate/price_zscore/volume_zscore are nullable in the DB
        # (Welford stats always produce a real value in practice, but the
        # column itself doesn't guarantee it) - fill defensively so a stray
        # NULL can't reach IsolationForest.fit() and blow up with a NaN error.
        pl.col("volatility_estimate").fill_null(0.0),
        pl.col("price_zscore").fill_null(0.0),
        pl.col("volume_zscore").fill_null(0.0),
        pl.col("rating_buy_ratio").fill_null(0.0),
        pl.col("rating_sell_ratio").fill_null(0.0),
        pl.col("split_recent_flag").fill_null(0),
        pl.col("news_count_recent").fill_null(0),
        pl.col("earnings_near_flag").fill_null(0),
        pl.col("earnings_estimate_shifted_flag").fill_null(0),
    )


def _rows_to_polars(rows: Any, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame([tuple(row) for row in rows], schema=schema, orient="row")


def _load_price_stats(session: Session, tickers: list[str]) -> pl.DataFrame:
    rows = session.execute(
        select(
            WindowedPriceStats.ticker,
            WindowedPriceStats.window_end,
            WindowedPriceStats.avg_price,
            WindowedPriceStats.total_volume,
            WindowedPriceStats.volatility_estimate,
            WindowedPriceStats.price_zscore,
            WindowedPriceStats.volume_zscore,
        ).where(WindowedPriceStats.ticker.in_(tickers))
    ).all()
    return _rows_to_polars(
        rows,
        {
            "ticker": pl.Utf8,
            "window_end": pl.Datetime("us", "UTC"),
            "avg_price": pl.Float64,
            "total_volume": pl.Int64,
            "volatility_estimate": pl.Float64,
            "price_zscore": pl.Float64,
            "volume_zscore": pl.Float64,
        },
    )


def _load_sector_industry(session: Session, tickers: list[str]) -> pl.DataFrame:
    rows = session.execute(
        select(
            DimSectorIndustry.ticker,
            DimSectorIndustry.sector,
            DimSectorIndustry.industry,
        ).where(
            DimSectorIndustry.ticker.in_(tickers),
            DimSectorIndustry.is_current.is_(True),
        )
    ).all()
    return _rows_to_polars(
        rows, {"ticker": pl.Utf8, "sector": pl.Utf8, "industry": pl.Utf8}
    )


def _load_ratings(session: Session, tickers: list[str]) -> pl.DataFrame:
    rows = session.execute(
        select(
            DimRatingConsensus.ticker,
            DimRatingConsensus.strong_buy,
            DimRatingConsensus.buy,
            DimRatingConsensus.hold,
            DimRatingConsensus.sell,
            DimRatingConsensus.strong_sell,
        ).where(
            DimRatingConsensus.ticker.in_(tickers),
            DimRatingConsensus.is_current.is_(True),
        )
    ).all()
    ratings = _rows_to_polars(
        rows,
        {
            "ticker": pl.Utf8,
            "strong_buy": pl.Int64,
            "buy": pl.Int64,
            "hold": pl.Int64,
            "sell": pl.Int64,
            "strong_sell": pl.Int64,
        },
    )
    total = (
        pl.col("strong_buy")
        + pl.col("buy")
        + pl.col("hold")
        + pl.col("sell")
        + pl.col("strong_sell")
    )
    return ratings.with_columns(
        pl.when(total > 0)
        .then((pl.col("strong_buy") + pl.col("buy")) / total)
        .otherwise(0.0)
        .alias("rating_buy_ratio"),
        pl.when(total > 0)
        .then((pl.col("strong_sell") + pl.col("sell")) / total)
        .otherwise(0.0)
        .alias("rating_sell_ratio"),
    ).select(["ticker", "rating_buy_ratio", "rating_sell_ratio"])


def _load_split_flags(
    session: Session, tickers: list[str], now: datetime
) -> pl.DataFrame:
    cutoff = now - timedelta(days=SPLIT_LOOKBACK_DAYS)
    rows = session.execute(
        select(RawSplit.ticker)
        .where(RawSplit.ticker.in_(tickers), RawSplit.split_date >= cutoff)
        .distinct()
    ).all()
    flagged = {row[0] for row in rows}
    return pl.DataFrame(
        {"ticker": tickers, "split_recent_flag": [int(t in flagged) for t in tickers]}
    )


def _load_news_counts(
    session: Session, tickers: list[str], now: datetime
) -> pl.DataFrame:
    cutoff = now - timedelta(days=NEWS_LOOKBACK_DAYS)
    rows = session.execute(
        select(RawNews.scope_key, func.count(RawNews.id))
        .where(
            RawNews.scope == "company",
            RawNews.scope_key.in_(tickers),
            RawNews.published_at >= cutoff,
        )
        .group_by(RawNews.scope_key)
    ).all()
    counts = dict(rows)
    return pl.DataFrame(
        {"ticker": tickers, "news_count_recent": [counts.get(t, 0) for t in tickers]}
    )


def _load_earnings_flags(
    session: Session, tickers: list[str], now: datetime
) -> pl.DataFrame:
    near_cutoff = now + timedelta(days=EARNINGS_NEAR_DAYS)
    shift_cutoff = now - timedelta(days=ESTIMATE_SHIFT_LOOKBACK_DAYS)

    near_rows = session.execute(
        select(DimEarningsEstimate.ticker).where(
            DimEarningsEstimate.ticker.in_(tickers),
            DimEarningsEstimate.is_current.is_(True),
            DimEarningsEstimate.earnings_date.between(now, near_cutoff),
        )
    ).all()
    near_set = {row[0] for row in near_rows}

    shifted_rows = session.execute(
        select(DimEarningsEstimate.ticker)
        .where(
            DimEarningsEstimate.ticker.in_(tickers),
            DimEarningsEstimate.is_current.is_(False),
            DimEarningsEstimate.valid_to >= shift_cutoff,
        )
        .distinct()
    ).all()
    shifted_set = {row[0] for row in shifted_rows}

    return pl.DataFrame(
        {
            "ticker": tickers,
            "earnings_near_flag": [int(t in near_set) for t in tickers],
            "earnings_estimate_shifted_flag": [int(t in shifted_set) for t in tickers],
        }
    )
