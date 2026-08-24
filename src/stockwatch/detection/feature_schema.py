"""The numeric feature contract shared by feature engineering (features/build_features.py,
which reads Postgres) and the inference microservice (inference_service/, which never
touches Postgres - it only ever sees rows shaped like this over HTTP). Split out from
build_features.py specifically so the inference service can depend on this without
pulling in SQLAlchemy/DB code it has no business needing.

Deliberately excluded: raw `avg_price`/`total_volume`. Both are on absolute scales that
vary by orders of magnitude across tickers (a $5 stock vs. a $500 stock; a thinly-traded
name vs. a heavily-traded one), so a model shared across the whole watchlist would partly
learn "which ticker is this" rather than "is this unusual for its own history." Both stay
as feature-matrix columns (the dashboard charts `avg_price` directly) but are fed to the
model only via their ticker-relative forms - `price_zscore` and `volume_zscore`, both
computed via the same per-ticker Welford running stats (streaming/rolling_stats.py) that
also drive `volatility_estimate`.
"""

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
