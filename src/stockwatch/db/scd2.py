"""The one place SCD2 (slowly changing dimension, type 2) logic lives.

Every dimension ingester (classification, index_membership, ratings,
earnings_calendar) calls `scd2_upsert` with its own natural key + attributes -
none of them re-implement the versioning logic themselves (DRY).
"""

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from stockwatch.db.models import Base


def scd2_upsert(
    session: Session,
    model: type[Base],
    natural_key: dict[str, Any],
    attributes: dict[str, Any],
    observed_at: datetime,
) -> None:
    """Upsert one dimension row using SCD2 semantics.

    - No current row for `natural_key` -> insert a new current row.
    - Current row exists and `attributes` are unchanged -> no-op (this is the
      point of SCD2: don't version a row just because we polled again).
    - Current row exists and `attributes` changed -> close the old row
      (valid_to=observed_at, is_current=False) and insert a new current row.
    """
    current = _get_current_row(session, model, natural_key)

    if current is None:
        session.add(
            model(
                **natural_key,
                **attributes,
                valid_from=observed_at,
                valid_to=None,
                is_current=True,
            )
        )
        return

    if _attributes_unchanged(current, attributes):
        return

    current.valid_to = observed_at
    current.is_current = False
    session.add(
        model(
            **natural_key,
            **attributes,
            valid_from=observed_at,
            valid_to=None,
            is_current=True,
        )
    )


def scd2_as_of(
    session: Session,
    model: type[Base],
    natural_key: dict[str, Any],
    as_of: datetime,
) -> Any | None:
    """The dimension row that was current at `as_of`: valid_from <= as_of AND
    (valid_to IS NULL OR valid_to > as_of). None if no row existed yet at that
    time. Symmetric to `scd2_upsert` - the generic point-in-time reader every
    as-of query wrapper (classification.py, ratings.py, ...) delegates to.
    """
    stmt = (
        select(model)
        .filter_by(**natural_key)
        .where(
            model.valid_from <= as_of,
            (model.valid_to.is_(None)) | (model.valid_to > as_of),
        )
    )
    return session.execute(stmt).scalar_one_or_none()


def _get_current_row(
    session: Session, model: type[Base], natural_key: dict[str, Any]
) -> Any:
    stmt = select(model).filter_by(is_current=True, **natural_key)
    return session.execute(stmt).scalar_one_or_none()


def _attributes_unchanged(current_row: Any, attributes: dict[str, Any]) -> bool:
    return all(
        getattr(current_row, column) == value for column, value in attributes.items()
    )
