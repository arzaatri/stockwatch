"""ingestion.index_membership only calls fetch_index_table through a
monkeypatched seam - no real Wikipedia scraping happens in tests.
"""

import polars as pl
from sqlalchemy import select
from sqlalchemy.orm import Session

from stockwatch.db.models import DimIndexMembership
from stockwatch.ingestion import index_membership


def test_ingest_index_membership_true_when_ticker_is_a_constituent(
    monkeypatch, db_session: Session
) -> None:
    monkeypatch.setattr(
        index_membership,
        "fetch_index_table",
        lambda name: pl.DataFrame({"ticker": ["AAPL", "MSFT"]}),
    )

    index_membership.ingest_index_membership("AAPL")

    rows = (
        db_session.execute(
            select(DimIndexMembership).where(DimIndexMembership.ticker == "AAPL")
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1
    assert rows[0].is_member is True
    assert rows[0].index_name == "sp500"


def test_ingest_index_membership_false_when_ticker_is_not_a_constituent(
    monkeypatch, db_session: Session
) -> None:
    monkeypatch.setattr(
        index_membership,
        "fetch_index_table",
        lambda name: pl.DataFrame({"ticker": ["MSFT"]}),
    )

    index_membership.ingest_index_membership("AAPL")

    rows = (
        db_session.execute(
            select(DimIndexMembership).where(DimIndexMembership.ticker == "AAPL")
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1
    assert rows[0].is_member is False


def test_ingest_index_membership_versions_on_change(
    monkeypatch, db_session: Session
) -> None:
    monkeypatch.setattr(
        index_membership,
        "fetch_index_table",
        lambda name: pl.DataFrame({"ticker": ["AAPL"]}),
    )
    index_membership.ingest_index_membership("AAPL")

    monkeypatch.setattr(
        index_membership, "fetch_index_table", lambda name: pl.DataFrame({"ticker": []})
    )
    index_membership.ingest_index_membership("AAPL")

    rows = (
        db_session.execute(
            select(DimIndexMembership).where(DimIndexMembership.ticker == "AAPL")
        )
        .scalars()
        .all()
    )
    assert len(rows) == 2
    current = [row for row in rows if row.is_current]
    assert len(current) == 1
    assert current[0].is_member is False
