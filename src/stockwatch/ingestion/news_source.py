"""Pluggable news source seam. `YFinanceNewsSource` is the only implementation
today; a `NewsApiSource` (NewsAPI.org) can be added later that satisfies the
same `NewsSource` protocol without touching any calling code.
"""

from typing import Protocol

from stockwatch.ingestion.yfinance_client import NewsItem
from stockwatch.ingestion.yfinance_client import get_news as _yfinance_get_news


class NewsSource(Protocol):
    def get_news(self, ticker: str, count: int = 10) -> list[NewsItem]: ...


class YFinanceNewsSource:
    def get_news(self, ticker: str, count: int = 10) -> list[NewsItem]:
        return _yfinance_get_news(ticker, count=count)
