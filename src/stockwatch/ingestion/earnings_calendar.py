"""Upcoming earnings date + EPS estimate, tracked as an SCD2 dimension since
the estimate genuinely shifts in the run-up to the actual event.
"""

from datetime import UTC, datetime

from stockwatch.db.engine import session_scope
from stockwatch.db.models import DimEarningsEstimate
from stockwatch.db.scd2 import scd2_upsert
from stockwatch.ingestion.yfinance_client import get_earnings_estimate


def ingest_earnings_estimate(ticker: str) -> None:
    estimate = get_earnings_estimate(ticker)
    if estimate is None:
        return

    with session_scope() as session:
        scd2_upsert(
            session,
            DimEarningsEstimate,
            natural_key={"ticker": ticker},
            attributes={
                "earnings_date": estimate.earnings_date,
                "eps_estimate": estimate.eps_estimate,
            },
            observed_at=datetime.now(UTC),
        )
