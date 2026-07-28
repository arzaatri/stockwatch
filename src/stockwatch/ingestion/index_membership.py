"""Index membership (e.g. is this ticker currently in the S&P 500), tracked as
an SCD2 dimension. Reuses universe.index_constituents.fetch_index_table - the
same function that powers watchlist seeding.
"""

from datetime import UTC, datetime

from stockwatch.db.engine import session_scope
from stockwatch.db.models import DimIndexMembership
from stockwatch.db.scd2 import scd2_upsert
from stockwatch.universe.index_constituents import IndexName, fetch_index_table

TRACKED_INDICES: list[IndexName] = ["sp500"]


def ingest_index_membership(ticker: str) -> None:
    observed_at = datetime.now(UTC)
    memberships = {
        index_name: ticker in fetch_index_table(index_name)["ticker"].to_list()
        for index_name in TRACKED_INDICES
    }

    with session_scope() as session:
        for index_name, is_member in memberships.items():
            scd2_upsert(
                session,
                DimIndexMembership,
                natural_key={"ticker": ticker, "index_name": index_name},
                attributes={"is_member": is_member},
                observed_at=observed_at,
            )
