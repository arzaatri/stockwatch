"""Stock splits: an append-only CDC log. yfinance returns the *full* split
history every call, so inserts are deduped on the (ticker, split_date) unique
constraint via ON CONFLICT DO NOTHING rather than re-checking existence first.
"""

from datetime import UTC, datetime

from sqlalchemy.dialects.postgresql import insert as pg_insert

from stockwatch.db.engine import session_scope
from stockwatch.db.models import RawSplit
from stockwatch.ingestion.yfinance_client import get_splits


def ingest_splits(ticker: str) -> None:
    splits = get_splits(ticker)
    if not splits:
        return

    ingested_at = datetime.now(UTC)
    with session_scope() as session:
        for split in splits:
            stmt = (
                pg_insert(RawSplit)
                .values(
                    ticker=split.ticker,
                    split_date=split.split_date,
                    numerator=split.numerator,
                    denominator=split.denominator,
                    ingested_at=ingested_at,
                )
                .on_conflict_do_nothing(index_elements=["ticker", "split_date"])
            )
            session.execute(stmt)
