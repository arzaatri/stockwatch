"""ingestion.classification only calls yfinance_client through a monkeypatched
seam - the real yfinance.Ticker is never touched in tests.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from stockwatch.db.models import DimSectorIndustry
from stockwatch.ingestion import classification
from stockwatch.ingestion.yfinance_client import SectorIndustry


def test_ingest_classification_writes_a_current_row(
    monkeypatch, db_session: Session
) -> None:
    monkeypatch.setattr(
        classification,
        "get_sector_industry",
        lambda ticker: SectorIndustry(
            ticker=ticker, sector="Technology", industry="Software"
        ),
    )

    classification.ingest_classification("MSFT")

    rows = (
        db_session.execute(
            select(DimSectorIndustry).where(DimSectorIndustry.ticker == "MSFT")
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1
    assert rows[0].sector == "Technology"
    assert rows[0].is_current is True


def test_ingest_classification_is_a_noop_when_yfinance_data_is_missing(
    monkeypatch, db_session: Session
) -> None:
    monkeypatch.setattr(classification, "get_sector_industry", lambda ticker: None)

    classification.ingest_classification("MSFT")

    rows = (
        db_session.execute(
            select(DimSectorIndustry).where(DimSectorIndustry.ticker == "MSFT")
        )
        .scalars()
        .all()
    )
    assert rows == []
