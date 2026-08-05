"""Sector/industry classification, tracked as an SCD2 dimension."""

from datetime import UTC, datetime

from stockwatch.db.engine import session_scope
from stockwatch.db.models import DimSectorIndustry
from stockwatch.db.scd2 import scd2_as_of, scd2_upsert
from stockwatch.ingestion.yfinance_client import SectorIndustry, get_sector_industry


def ingest_classification(ticker: str) -> None:
    sector_industry = get_sector_industry(ticker)
    if sector_industry is None:
        return  # missing this cycle (known Yahoo-side flakiness) - no update, not an error

    with session_scope() as session:
        scd2_upsert(
            session,
            DimSectorIndustry,
            natural_key={"ticker": ticker},
            attributes={
                "sector": sector_industry.sector,
                "industry": sector_industry.industry,
            },
            observed_at=datetime.now(UTC),
        )


def get_sector_industry_as_of(ticker: str, as_of: datetime) -> SectorIndustry | None:
    with session_scope() as session:
        row = scd2_as_of(session, DimSectorIndustry, {"ticker": ticker}, as_of)
        if row is None or row.sector is None or row.industry is None:
            return None
        return SectorIndustry(ticker=ticker, sector=row.sector, industry=row.industry)
