"""The one seam that touches the real NewsAPI.org HTTP endpoint, mirroring
yfinance_client.py's role - other ingestion code calls `search_news` here,
never `requests`/NewsAPI directly, so it's mockable without a real API key.

Assumptions about NewsAPI's free "Developer" tier (verify once a real key is
in place): 100 requests/day, articles delayed ~24h, ~1 month lookback,
non-commercial use only. No client-side rate limiter here by design - see
pipeline/poll_loop.py's dedup-by-distinct-sector/industry, which keeps real
usage far under the daily cap without needing one.
"""

from datetime import datetime

import requests

from stockwatch.config import get_settings
from stockwatch.ingestion.yfinance_client import NewsItem, NewsScope
from stockwatch.logging_utils import get_logger

_EVERYTHING_URL = "https://newsapi.org/v2/everything"
_TIMEOUT_SECONDS = 10

logger = get_logger(__name__)


def search_news(query: str, scope: NewsScope, count: int = 10) -> list[NewsItem]:
    """Keyword search via NewsAPI's /v2/everything. Returns [] on a missing API
    key, HTTP error, or empty result set - callers treat that as "no news this
    cycle," not an error, same as yfinance_client's flakiness-tolerant functions.
    """
    api_key = get_settings().newsapi_api_key
    if not api_key:
        logger.warning("search_news: NEWSAPI_API_KEY not set, skipping")
        return []

    try:
        response = requests.get(
            _EVERYTHING_URL,
            params={
                "q": query,
                "pageSize": count,
                "sortBy": "publishedAt",
                "language": "en",
            },
            headers={"X-Api-Key": api_key},
            timeout=_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        logger.warning("search_news: request failed for query=%r: %s", query, error)
        return []

    articles = response.json().get("articles", [])
    news_items = []
    for article in articles:
        link = article.get("url")
        title = article.get("title")
        if not link or not title:
            continue
        source = article.get("source") or {}
        news_items.append(
            NewsItem(
                scope=scope,
                scope_key=query,
                headline=title,
                link=link,
                publisher=source.get("name"),
                snippet=article.get("description"),
                published_at=_parse_published_at(article.get("publishedAt")),
            )
        )
    return news_items


def _parse_published_at(published_at: str | None) -> datetime | None:
    if not published_at:
        return None
    return datetime.fromisoformat(published_at.replace("Z", "+00:00"))
