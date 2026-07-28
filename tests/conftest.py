"""Shared test fixtures. Tests run against the same docker-compose Postgres
used for development, via a separate `stockwatch_test` database (created and
schema-applied once per test session) rather than testcontainers or SQLite -
it needs to be real Postgres so the SCD2 partial-unique-index invariant
actually gets exercised.
"""

import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from stockwatch.config import get_settings
from stockwatch.db import engine as db_engine

TEST_DB_NAME = "stockwatch_test"
INIT_SQL_PATH = Path(__file__).resolve().parent.parent / "scripts" / "init_db.sql"

_TABLES = [
    "watchlist",
    "raw_price_ticks",
    "raw_splits",
    "raw_news",
    "raw_index_snapshot",
    "windowed_price_stats",
    "dim_sector_industry",
    "dim_index_membership",
    "dim_rating_consensus",
    "dim_earnings_estimate",
]


@pytest.fixture(scope="session", autouse=True)
def _use_test_database() -> None:
    os.environ["POSTGRES_DB"] = TEST_DB_NAME
    get_settings.cache_clear()
    db_engine.get_engine.cache_clear()
    db_engine.get_session_factory.cache_clear()


@pytest.fixture(scope="session")
def test_engine(_use_test_database: None) -> Engine:
    settings = get_settings()
    admin_dsn = settings.postgres_dsn.rsplit("/", 1)[0] + "/postgres"
    admin_engine = create_engine(admin_dsn, isolation_level="AUTOCOMMIT")
    with admin_engine.connect() as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{TEST_DB_NAME}"'))
        conn.execute(text(f'CREATE DATABASE "{TEST_DB_NAME}"'))
    admin_engine.dispose()

    engine = db_engine.get_engine()
    with engine.begin() as conn:
        conn.execute(text(INIT_SQL_PATH.read_text()))
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(test_engine: Engine) -> Session:
    session = sessionmaker(bind=test_engine)()
    yield session
    session.close()


@pytest.fixture(autouse=True)
def _clean_tables(test_engine: Engine):
    yield
    with test_engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {', '.join(_TABLES)} RESTART IDENTITY CASCADE"))
