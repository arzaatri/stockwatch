from sqlalchemy.orm import Session

from stockwatch.universe.watchlist import (
    add_ticker,
    deactivate_ticker,
    get_active_tickers,
    watchlist_count,
)


def test_add_ticker_makes_it_active(db_session: Session) -> None:
    add_ticker("AAPL")

    assert get_active_tickers() == ["AAPL"]
    assert watchlist_count() == 1


def test_deactivate_ticker_removes_it_from_active_list(db_session: Session) -> None:
    add_ticker("AAPL")
    add_ticker("MSFT")

    deactivate_ticker("AAPL")

    assert get_active_tickers() == ["MSFT"]


def test_add_ticker_twice_is_idempotent(db_session: Session) -> None:
    add_ticker("AAPL")
    add_ticker("AAPL")

    assert watchlist_count() == 1
