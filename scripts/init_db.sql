-- Plain, from-scratch schema. No migration framework by design (see plan) --
-- read top to bottom to understand the whole database.

-- ---------------------------------------------------------------------------
-- Ticker universe (expandable watchlist, not a hardcoded list)
-- ---------------------------------------------------------------------------
CREATE TABLE watchlist (
    ticker     TEXT PRIMARY KEY,
    added_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_active  BOOLEAN NOT NULL DEFAULT TRUE
);

-- ---------------------------------------------------------------------------
-- Append-only raw / CDC logs.
-- Rows are never UPDATEd or DELETEd: each row is an immutable observation.
-- `ingested_at` = when *we* observed it; source-native timestamp columns
-- (as_of / published_at / split_date / fetched_at) are the source's own time.
-- ---------------------------------------------------------------------------
CREATE TABLE raw_price_ticks (
    id          BIGSERIAL PRIMARY KEY,
    ticker      TEXT NOT NULL,
    price       DOUBLE PRECISION NOT NULL,
    volume      BIGINT,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    as_of       TIMESTAMPTZ NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_raw_price_ticks_ticker_as_of ON raw_price_ticks (ticker, as_of);

CREATE TABLE raw_splits (
    id          BIGSERIAL PRIMARY KEY,
    ticker      TEXT NOT NULL,
    split_date  TIMESTAMPTZ NOT NULL,
    numerator   DOUBLE PRECISION NOT NULL,
    denominator DOUBLE PRECISION NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (ticker, split_date)
);

CREATE TABLE raw_news (
    id           BIGSERIAL PRIMARY KEY,
    scope        TEXT NOT NULL,   -- 'company' | 'sector' | 'industry'
    scope_key    TEXT NOT NULL,   -- ticker, sector name, or industry name
    headline     TEXT NOT NULL,
    link         TEXT NOT NULL,
    publisher    TEXT,
    snippet      TEXT,
    published_at TIMESTAMPTZ,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (scope, scope_key, link)
);
CREATE INDEX ix_raw_news_scope_key_published_at ON raw_news (scope, scope_key, published_at);

CREATE TABLE raw_index_snapshot (
    id          BIGSERIAL PRIMARY KEY,
    index_name  TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    fetched_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_raw_index_snapshot_index_fetched ON raw_index_snapshot (index_name, fetched_at);

-- ---------------------------------------------------------------------------
-- Streaming output: one row per (ticker, window_end) emitted by the PyFlink
-- job via streaming/stats_consumer.py. Append-only, same as the raw logs above.
-- ---------------------------------------------------------------------------
CREATE TABLE windowed_price_stats (
    id                  BIGSERIAL PRIMARY KEY,
    ticker              TEXT NOT NULL,
    window_end          TIMESTAMPTZ NOT NULL,
    avg_price           DOUBLE PRECISION NOT NULL,
    total_volume        BIGINT NOT NULL,
    volatility_estimate DOUBLE PRECISION,
    price_zscore        DOUBLE PRECISION,
    volume_zscore       DOUBLE PRECISION,
    ingested_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (ticker, window_end)
);
CREATE INDEX ix_windowed_price_stats_ticker_window ON windowed_price_stats (ticker, window_end);

-- ---------------------------------------------------------------------------
-- Async "explain this anomaly" job tracking (api/jobs.py). Postgres-backed
-- rather than in-memory so polling GET /explain/{job_id} works regardless of
-- which api replica a request lands on.
-- ---------------------------------------------------------------------------
CREATE TABLE explanation_jobs (
    job_id       TEXT PRIMARY KEY,
    ticker       TEXT NOT NULL,
    window_end   TIMESTAMPTZ NOT NULL,
    status       TEXT NOT NULL,   -- 'pending' | 'running' | 'done' | 'error'
    result       JSON,
    error        TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);

-- ---------------------------------------------------------------------------
-- SCD2 dimension tables. Each has valid_from/valid_to/is_current, and a
-- partial unique index enforcing "at most one current row per natural key" -
-- the invariant db/scd2.py's generic upsert function must never violate.
-- ---------------------------------------------------------------------------
CREATE TABLE dim_sector_industry (
    surrogate_id BIGSERIAL PRIMARY KEY,
    ticker       TEXT NOT NULL,
    sector       TEXT,
    industry     TEXT,
    valid_from   TIMESTAMPTZ NOT NULL,
    valid_to     TIMESTAMPTZ,
    is_current   BOOLEAN NOT NULL
);
CREATE UNIQUE INDEX ux_dim_sector_industry_current
    ON dim_sector_industry (ticker) WHERE is_current;

CREATE TABLE dim_index_membership (
    surrogate_id BIGSERIAL PRIMARY KEY,
    ticker       TEXT NOT NULL,
    index_name   TEXT NOT NULL,
    is_member    BOOLEAN NOT NULL,
    valid_from   TIMESTAMPTZ NOT NULL,
    valid_to     TIMESTAMPTZ,
    is_current   BOOLEAN NOT NULL
);
CREATE UNIQUE INDEX ux_dim_index_membership_current
    ON dim_index_membership (ticker, index_name) WHERE is_current;

CREATE TABLE dim_rating_consensus (
    surrogate_id BIGSERIAL PRIMARY KEY,
    ticker       TEXT NOT NULL,
    strong_buy   INTEGER NOT NULL DEFAULT 0,
    buy          INTEGER NOT NULL DEFAULT 0,
    hold         INTEGER NOT NULL DEFAULT 0,
    sell         INTEGER NOT NULL DEFAULT 0,
    strong_sell  INTEGER NOT NULL DEFAULT 0,
    valid_from   TIMESTAMPTZ NOT NULL,
    valid_to     TIMESTAMPTZ,
    is_current   BOOLEAN NOT NULL
);
CREATE UNIQUE INDEX ux_dim_rating_consensus_current
    ON dim_rating_consensus (ticker) WHERE is_current;

CREATE TABLE dim_earnings_estimate (
    surrogate_id  BIGSERIAL PRIMARY KEY,
    ticker        TEXT NOT NULL,
    earnings_date TIMESTAMPTZ NOT NULL,
    eps_estimate  DOUBLE PRECISION,
    valid_from    TIMESTAMPTZ NOT NULL,
    valid_to      TIMESTAMPTZ,
    is_current    BOOLEAN NOT NULL
);
CREATE UNIQUE INDEX ux_dim_earnings_estimate_current
    ON dim_earnings_estimate (ticker) WHERE is_current;
