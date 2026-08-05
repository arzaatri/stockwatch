"""The most important test file: the generic SCD2 upsert must (a) insert on
first sight, (b) no-op when attributes are unchanged, (c) version the row
when attributes change, and (d) never end up with more than one current row
per natural key.
"""

from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from stockwatch.db.models import DimSectorIndustry
from stockwatch.db.scd2 import scd2_as_of, scd2_upsert


def _upsert(session: Session, observed_at: datetime, industry: str) -> None:
    scd2_upsert(
        session,
        DimSectorIndustry,
        natural_key={"ticker": "AAPL"},
        attributes={"sector": "Technology", "industry": industry},
        observed_at=observed_at,
    )
    session.commit()


def _rows(session: Session) -> list[DimSectorIndustry]:
    return list(
        session.execute(
            select(DimSectorIndustry)
            .where(DimSectorIndustry.ticker == "AAPL")
            .order_by(DimSectorIndustry.valid_from)
        )
        .scalars()
        .all()
    )


def test_first_upsert_creates_one_current_row(db_session: Session) -> None:
    _upsert(db_session, datetime.now(UTC), "Consumer Electronics")

    rows = _rows(db_session)
    assert len(rows) == 1
    assert rows[0].is_current is True
    assert rows[0].valid_to is None


def test_identical_attributes_is_a_noop(db_session: Session) -> None:
    t1 = datetime.now(UTC)
    _upsert(db_session, t1, "Consumer Electronics")
    _upsert(db_session, t1 + timedelta(days=1), "Consumer Electronics")

    assert len(_rows(db_session)) == 1


def test_changed_attributes_versions_the_row(db_session: Session) -> None:
    t1 = datetime.now(UTC)
    t2 = t1 + timedelta(days=1)
    _upsert(db_session, t1, "Consumer Electronics")
    _upsert(db_session, t2, "Something Else")

    rows = _rows(db_session)
    assert len(rows) == 2
    assert rows[0].is_current is False
    assert rows[0].valid_to == t2
    assert rows[1].is_current is True
    assert rows[1].valid_to is None
    assert rows[1].industry == "Something Else"


def test_is_current_invariant_holds_after_many_upserts(db_session: Session) -> None:
    t0 = datetime.now(UTC)
    for i in range(5):
        _upsert(db_session, t0 + timedelta(days=i), f"Industry {i}")

    current_rows = [row for row in _rows(db_session) if row.is_current]
    assert len(current_rows) == 1
    assert current_rows[0].industry == "Industry 4"


def test_as_of_before_any_version_returns_none(db_session: Session) -> None:
    t1 = datetime.now(UTC)
    _upsert(db_session, t1, "Consumer Electronics")

    row = scd2_as_of(
        db_session, DimSectorIndustry, {"ticker": "AAPL"}, t1 - timedelta(days=1)
    )

    assert row is None


def test_as_of_between_versions_returns_the_version_current_then(
    db_session: Session,
) -> None:
    t1 = datetime.now(UTC)
    t2 = t1 + timedelta(days=10)
    _upsert(db_session, t1, "Consumer Electronics")
    _upsert(db_session, t2, "Something Else")

    row = scd2_as_of(
        db_session, DimSectorIndustry, {"ticker": "AAPL"}, t1 + timedelta(days=1)
    )

    assert row is not None
    assert row.industry == "Consumer Electronics"


def test_as_of_after_latest_version_returns_the_current_row(
    db_session: Session,
) -> None:
    t1 = datetime.now(UTC)
    t2 = t1 + timedelta(days=10)
    _upsert(db_session, t1, "Consumer Electronics")
    _upsert(db_session, t2, "Something Else")

    row = scd2_as_of(
        db_session, DimSectorIndustry, {"ticker": "AAPL"}, t2 + timedelta(days=1)
    )

    assert row is not None
    assert row.industry == "Something Else"
