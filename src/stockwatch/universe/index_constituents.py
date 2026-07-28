"""Fetch/cache stock index constituent lists (Wikipedia via pandas.read_html).

Powers two things with the same function: (a) sampling the initial watchlist
seed, and (b) the SCD2 index-membership dimension in ingestion/index_membership.py.
"""

import io
from datetime import UTC, datetime, timedelta
from typing import Literal

import pandas as pd
import polars as pl
import requests
from sqlalchemy import func, select

from stockwatch.db.engine import session_scope
from stockwatch.db.models import RawIndexSnapshot

IndexName = Literal["sp500", "nasdaq100", "dow"]

_INDEX_URLS: dict[IndexName, str] = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "nasdaq100": "https://en.wikipedia.org/wiki/Nasdaq-100",
    "dow": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
}

_SYMBOL_COLUMN_NAMES = {"symbol", "ticker"}


def fetch_index_table(
    index_name: IndexName, max_age: timedelta = timedelta(hours=24)
) -> pl.DataFrame:
    """Return a one-column ("ticker") DataFrame of the index's current constituents,
    using a cached snapshot if one was fetched within `max_age`.
    """
    with session_scope() as session:
        cached = _get_cached_snapshot(session, index_name, max_age)
        if cached is not None:
            return cached

        tickers = _scrape_index_constituents(index_name)
        _store_snapshot(session, index_name, tickers)
        return pl.DataFrame({"ticker": tickers})


def _get_cached_snapshot(
    session, index_name: str, max_age: timedelta
) -> pl.DataFrame | None:
    latest_fetch = session.execute(
        select(func.max(RawIndexSnapshot.fetched_at)).where(
            RawIndexSnapshot.index_name == index_name
        )
    ).scalar_one_or_none()

    if latest_fetch is None or datetime.now(UTC) - latest_fetch > max_age:
        return None

    tickers = (
        session.execute(
            select(RawIndexSnapshot.ticker).where(
                RawIndexSnapshot.index_name == index_name,
                RawIndexSnapshot.fetched_at == latest_fetch,
            )
        )
        .scalars()
        .all()
    )
    return pl.DataFrame({"ticker": list(tickers)})


def _store_snapshot(session, index_name: str, tickers: list[str]) -> None:
    fetched_at = datetime.now(UTC)
    session.add_all(
        RawIndexSnapshot(index_name=index_name, ticker=ticker, fetched_at=fetched_at)
        for ticker in tickers
    )


def _scrape_index_constituents(index_name: IndexName) -> list[str]:
    # Wikipedia rejects requests without a browser-like User-Agent (403).
    response = requests.get(
        _INDEX_URLS[index_name], headers={"User-Agent": "Mozilla/5.0"}
    )
    response.raise_for_status()
    tables: list[pd.DataFrame] = pd.read_html(io.StringIO(response.text))
    for table in tables:
        for column in table.columns:
            if str(column).strip().lower() in _SYMBOL_COLUMN_NAMES:
                return (
                    table[column]
                    .astype(str)
                    .str.strip()
                    .str.replace(
                        ".", "-", regex=False
                    )  # e.g. BRK.B -> BRK-B for yfinance
                    .tolist()
                )
    raise ValueError(f"could not find a ticker/symbol column for index {index_name!r}")
