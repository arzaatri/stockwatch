"""ingestion.classification only calls yfinance_client through a monkeypatched
seam - the real yfinance.Ticker is never touched in tests.
"""

from datetime import UTC, datetime, timedelta

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


def test_get_sector_industry_as_of_returns_the_version_current_then(
    monkeypatch, db_session: Session
) -> None:
    t1 = datetime.now(UTC)
    t2 = t1 + timedelta(days=10)
    monkeypatch.setattr(
        classification,
        "get_sector_industry",
        lambda ticker: SectorIndustry(
            ticker=ticker, sector="Technology", industry="Old"
        ),
    )
    classification.ingest_classification("MSFT")
    monkeypatch.setattr(
        classification,
        "get_sector_industry",
        lambda ticker: SectorIndustry(
            ticker=ticker, sector="Technology", industry="New"
        ),
    )
    classification.ingest_classification("MSFT")

    # Force the two versions' valid_from/valid_to to known timestamps so the
    # as-of query below is unambiguous, regardless of real wall-clock timing.
    rows = (
        db_session.execute(
            select(DimSectorIndustry)
            .where(DimSectorIndustry.ticker == "MSFT")
            .order_by(DimSectorIndustry.valid_from)
        )
        .scalars()
        .all()
    )
    rows[0].valid_from, rows[0].valid_to = t1, t2
    rows[1].valid_from = t2
    db_session.commit()

    result = classification.get_sector_industry_as_of("MSFT", t1 + timedelta(days=1))

    assert result is not None
    assert result.industry == "Old"
