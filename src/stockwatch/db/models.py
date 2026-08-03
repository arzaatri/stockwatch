"""ORM models mirroring scripts/init_db.sql. This module only declares table shape;
CDC/SCD2 behavior lives in db/scd2.py and the ingestion/* callers.
"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Double,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Watchlist(Base):
    __tablename__ = "watchlist"

    ticker: Mapped[str] = mapped_column(Text, primary_key=True)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class RawPriceTick(Base):
    __tablename__ = "raw_price_ticks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(Text)
    price: Mapped[float] = mapped_column(Double)
    volume: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    open: Mapped[float | None] = mapped_column(Double, nullable=True)
    high: Mapped[float | None] = mapped_column(Double, nullable=True)
    low: Mapped[float | None] = mapped_column(Double, nullable=True)
    close: Mapped[float | None] = mapped_column(Double, nullable=True)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class RawSplit(Base):
    __tablename__ = "raw_splits"
    __table_args__ = (UniqueConstraint("ticker", "split_date"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(Text)
    split_date: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    numerator: Mapped[float] = mapped_column(Double)
    denominator: Mapped[float] = mapped_column(Double)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class RawNews(Base):
    __tablename__ = "raw_news"
    __table_args__ = (UniqueConstraint("scope", "scope_key", "link"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    scope: Mapped[str] = mapped_column(Text)
    scope_key: Mapped[str] = mapped_column(Text)
    headline: Mapped[str] = mapped_column(Text)
    link: Mapped[str] = mapped_column(Text)
    publisher: Mapped[str | None] = mapped_column(Text, nullable=True)
    snippet: Mapped[str | None] = mapped_column(Text, nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class RawIndexSnapshot(Base):
    __tablename__ = "raw_index_snapshot"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    index_name: Mapped[str] = mapped_column(Text)
    ticker: Mapped[str] = mapped_column(Text)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class WindowedPriceStats(Base):
    __tablename__ = "windowed_price_stats"
    __table_args__ = (UniqueConstraint("ticker", "window_end"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(Text)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    avg_price: Mapped[float] = mapped_column(Double)
    total_volume: Mapped[int] = mapped_column(BigInteger)
    volatility_estimate: Mapped[float | None] = mapped_column(Double, nullable=True)
    price_zscore: Mapped[float | None] = mapped_column(Double, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class DimSectorIndustry(Base):
    __tablename__ = "dim_sector_industry"

    surrogate_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    ticker: Mapped[str] = mapped_column(Text)
    sector: Mapped[str | None] = mapped_column(Text, nullable=True)
    industry: Mapped[str | None] = mapped_column(Text, nullable=True)
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    valid_to: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    is_current: Mapped[bool] = mapped_column(Boolean)


class DimIndexMembership(Base):
    __tablename__ = "dim_index_membership"

    surrogate_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    ticker: Mapped[str] = mapped_column(Text)
    index_name: Mapped[str] = mapped_column(Text)
    is_member: Mapped[bool] = mapped_column(Boolean)
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    valid_to: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    is_current: Mapped[bool] = mapped_column(Boolean)


class DimRatingConsensus(Base):
    __tablename__ = "dim_rating_consensus"

    surrogate_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    ticker: Mapped[str] = mapped_column(Text)
    strong_buy: Mapped[int] = mapped_column(Integer, default=0)
    buy: Mapped[int] = mapped_column(Integer, default=0)
    hold: Mapped[int] = mapped_column(Integer, default=0)
    sell: Mapped[int] = mapped_column(Integer, default=0)
    strong_sell: Mapped[int] = mapped_column(Integer, default=0)
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    valid_to: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    is_current: Mapped[bool] = mapped_column(Boolean)


class DimEarningsEstimate(Base):
    __tablename__ = "dim_earnings_estimate"

    surrogate_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    ticker: Mapped[str] = mapped_column(Text)
    earnings_date: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    eps_estimate: Mapped[float | None] = mapped_column(Double, nullable=True)
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    valid_to: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    is_current: Mapped[bool] = mapped_column(Boolean)
