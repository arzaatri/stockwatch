"""The one seam that touches the real `yfinance` library.

Every other module calls these functions, never `yfinance.Ticker` directly -
that's what makes ingestion testable (tests monkeypatch these functions with
plain fixtures instead of mocking yfinance's internals).
"""

from datetime import UTC, datetime

import yfinance as yf
from pydantic import BaseModel


class Quote(BaseModel):
    ticker: str
    price: float
    volume: int | None
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    as_of: datetime


class SectorIndustry(BaseModel):
    ticker: str
    sector: str
    industry: str


class RatingConsensus(BaseModel):
    ticker: str
    strong_buy: int
    buy: int
    hold: int
    sell: int
    strong_sell: int


class SplitEvent(BaseModel):
    ticker: str
    split_date: datetime
    numerator: float
    denominator: float = 1.0


class EarningsEstimate(BaseModel):
    ticker: str
    earnings_date: datetime
    eps_estimate: float | None


class NewsItem(BaseModel):
    ticker: str
    headline: str
    link: str
    publisher: str | None
    published_at: datetime | None


def get_quote(ticker: str) -> Quote | None:
    """Latest intraday bar. Returns None if the market has no data for `ticker`
    right now (e.g. market closed and no cached intraday bar available).
    """
    history = yf.Ticker(ticker).history(period="1d", interval="1m")
    if history.empty:
        return None
    last = history.iloc[-1]
    return Quote(
        ticker=ticker,
        price=float(last["Close"]),
        volume=int(last["Volume"]),
        open=float(last["Open"]),
        high=float(last["High"]),
        low=float(last["Low"]),
        close=float(last["Close"]),
        as_of=history.index[-1].to_pydatetime(),
    )


def get_price_history(ticker: str, period: str, interval: str) -> list[Quote]:
    """Historical OHLCV bars for backfilling raw_price_ticks - unlike
    `get_quote` (always just the latest 1m bar), this pulls a full lookback
    window at whatever interval the caller's polling granularity maps to.
    """
    history = yf.Ticker(ticker).history(period=period, interval=interval)
    if history.empty:
        return []
    return [
        Quote(
            ticker=ticker,
            price=float(row["Close"]),
            volume=None if row["Volume"] != row["Volume"] else int(row["Volume"]),
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            as_of=timestamp.to_pydatetime(),
        )
        for timestamp, row in history.iterrows()
    ]


def get_sector_industry(ticker: str) -> SectorIndustry | None:
    """Returns None if Yahoo's `.info` is missing sector/industry this cycle -
    a known, occasionally-flaky field on Yahoo's side. Callers should treat
    that as "no update this cycle," not an error.
    """
    info = yf.Ticker(ticker).get_info()
    sector = info.get("sector")
    industry = info.get("industry")
    if not sector or not industry:
        return None
    return SectorIndustry(ticker=ticker, sector=sector, industry=industry)


def get_rating_consensus(ticker: str) -> RatingConsensus | None:
    summary = yf.Ticker(ticker).get_recommendations_summary()
    if summary is None or summary.empty:
        return None
    current = summary[summary["period"] == "0m"]
    if current.empty:
        current = summary.iloc[[0]]
    row = current.iloc[0]
    return RatingConsensus(
        ticker=ticker,
        strong_buy=int(row["strongBuy"]),
        buy=int(row["buy"]),
        hold=int(row["hold"]),
        sell=int(row["sell"]),
        strong_sell=int(row["strongSell"]),
    )


def get_splits(ticker: str) -> list[SplitEvent]:
    splits = yf.Ticker(ticker).splits
    return [
        SplitEvent(
            ticker=ticker, split_date=split_date.to_pydatetime(), numerator=float(ratio)
        )
        for split_date, ratio in splits.items()
    ]


def get_earnings_estimate(ticker: str) -> EarningsEstimate | None:
    """The *next upcoming* earnings date estimate - the value SCD2 tracks as it
    shifts over time in the run-up to the actual event.
    """
    earnings_dates = yf.Ticker(ticker).get_earnings_dates(limit=12)
    if earnings_dates is None or earnings_dates.empty:
        return None
    now = datetime.now(UTC)
    upcoming = earnings_dates[earnings_dates.index.to_pydatetime() > now]
    if upcoming.empty:
        return None
    next_date = upcoming.index.min()
    row = upcoming.loc[next_date]
    eps_estimate = row["EPS Estimate"]
    return EarningsEstimate(
        ticker=ticker,
        earnings_date=next_date.to_pydatetime(),
        eps_estimate=None
        if eps_estimate != eps_estimate
        else float(eps_estimate),  # NaN check
    )


def get_news(ticker: str, count: int = 10) -> list[NewsItem]:
    raw_items = yf.Ticker(ticker).get_news(count=count)
    news_items = []
    for item in raw_items:
        content = item.get("content", {})
        canonical_url = content.get("canonicalUrl") or {}
        provider = content.get("provider") or {}
        link = canonical_url.get("url")
        title = content.get("title")
        if not link or not title:
            continue
        news_items.append(
            NewsItem(
                ticker=ticker,
                headline=title,
                link=link,
                publisher=provider.get("displayName"),
                published_at=_parse_pub_date(content.get("pubDate")),
            )
        )
    return news_items


def _parse_pub_date(pub_date: str | None) -> datetime | None:
    if not pub_date:
        return None
    return datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
