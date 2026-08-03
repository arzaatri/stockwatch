from datetime import UTC, datetime

from sqlalchemy.orm import Session

from stockwatch.db.models import (
    DimRatingConsensus,
    DimSectorIndustry,
    RawNews,
    WindowedPriceStats,
)
from stockwatch.features.build_features import FEATURE_COLUMNS, build_feature_matrix


def test_build_feature_matrix_is_empty_but_typed_when_no_price_stats(
    db_session: Session,
) -> None:
    matrix = build_feature_matrix(tickers=["ZZZZ"])

    assert matrix.height == 0
    for column in FEATURE_COLUMNS:
        assert column in matrix.columns


def test_build_feature_matrix_joins_context_onto_price_stats(
    db_session: Session,
) -> None:
    now = datetime.now(UTC)
    db_session.add(
        WindowedPriceStats(
            ticker="AAPL",
            window_end=now,
            avg_price=100.0,
            total_volume=1000,
            volatility_estimate=0.5,
            price_zscore=1.2,
            ingested_at=now,
        )
    )
    db_session.add(
        DimSectorIndustry(
            ticker="AAPL",
            sector="Technology",
            industry="Consumer Electronics",
            valid_from=now,
            valid_to=None,
            is_current=True,
        )
    )
    db_session.add(
        DimRatingConsensus(
            ticker="AAPL",
            strong_buy=5,
            buy=10,
            hold=3,
            sell=1,
            strong_sell=0,
            valid_from=now,
            valid_to=None,
            is_current=True,
        )
    )
    db_session.commit()

    matrix = build_feature_matrix(tickers=["AAPL"])

    assert matrix.height == 1
    row = matrix.row(0, named=True)
    assert row["sector"] == "Technology"
    assert row["rating_buy_ratio"] == (5 + 10) / (5 + 10 + 3 + 1 + 0)
    assert row["split_recent_flag"] == 0
    assert row["news_count_recent"] == 0
    for column in FEATURE_COLUMNS:
        assert column in matrix.columns


def test_build_feature_matrix_only_counts_company_scope_news(
    db_session: Session,
) -> None:
    now = datetime.now(UTC)
    db_session.add(
        WindowedPriceStats(
            ticker="AAPL",
            window_end=now,
            avg_price=100.0,
            total_volume=1000,
            volatility_estimate=0.5,
            price_zscore=1.2,
            ingested_at=now,
        )
    )
    db_session.add(
        RawNews(
            scope="company",
            scope_key="AAPL",
            headline="Apple news",
            link="https://example.com/company",
            publisher="Example",
            published_at=now,
            ingested_at=now,
        )
    )
    db_session.add(
        RawNews(
            scope="sector",
            scope_key="Technology",
            headline="Sector news",
            link="https://example.com/sector",
            publisher="Example",
            published_at=now,
            ingested_at=now,
        )
    )
    db_session.commit()

    matrix = build_feature_matrix(tickers=["AAPL"])

    row = matrix.row(0, named=True)
    assert row["news_count_recent"] == 1


def test_build_feature_matrix_fills_nulls_when_dimensions_are_missing(
    db_session: Session,
) -> None:
    now = datetime.now(UTC)
    db_session.add(
        WindowedPriceStats(
            ticker="NEW",
            window_end=now,
            avg_price=50.0,
            total_volume=10,
            volatility_estimate=0.1,
            price_zscore=0.0,
            ingested_at=now,
        )
    )
    db_session.commit()

    matrix = build_feature_matrix(tickers=["NEW"])

    row = matrix.row(0, named=True)
    assert row["rating_buy_ratio"] == 0.0
    assert row["split_recent_flag"] == 0
    assert row["sector"] is None
