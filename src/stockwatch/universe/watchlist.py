"""The expandable ticker universe. Every ingestion/pipeline call reads its
ticker list from `get_active_tickers` - nothing is ever hardcoded.
"""

from datetime import UTC, datetime

from sqlalchemy import select

from stockwatch.db.engine import session_scope
from stockwatch.db.models import Watchlist


def get_active_tickers() -> list[str]:
    with session_scope() as session:
        stmt = select(Watchlist.ticker).where(Watchlist.is_active.is_(True))
        return list(session.execute(stmt).scalars().all())


def watchlist_count() -> int:
    return len(get_active_tickers())


def add_ticker(ticker: str) -> None:
    with session_scope() as session:
        existing = session.get(Watchlist, ticker)
        if existing is None:
            session.add(
                Watchlist(ticker=ticker, added_at=datetime.now(UTC), is_active=True)
            )
        else:
            existing.is_active = True


def deactivate_ticker(ticker: str) -> None:
    with session_scope() as session:
        existing = session.get(Watchlist, ticker)
        if existing is not None:
            existing.is_active = False
