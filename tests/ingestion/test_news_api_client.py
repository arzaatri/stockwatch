from datetime import UTC, datetime

import pytest
import requests

from stockwatch.config import get_settings
from stockwatch.ingestion import news_api_client


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self) -> dict:
        return self._payload


@pytest.fixture
def _newsapi_key(monkeypatch) -> None:
    monkeypatch.setenv("NEWSAPI_API_KEY", "test-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_search_news_maps_articles_to_news_items(monkeypatch, _newsapi_key) -> None:
    payload = {
        "articles": [
            {
                "title": "Chipmakers rally",
                "url": "https://example.com/a",
                "source": {"name": "Example Wire"},
                "description": "A short summary.",
                "publishedAt": "2026-01-05T14:30:00Z",
            }
        ]
    }
    monkeypatch.setattr(
        news_api_client.requests, "get", lambda *a, **k: _FakeResponse(payload)
    )

    items = news_api_client.search_news("Technology", scope="sector", count=5)

    assert len(items) == 1
    item = items[0]
    assert item.scope == "sector"
    assert item.scope_key == "Technology"
    assert item.headline == "Chipmakers rally"
    assert item.link == "https://example.com/a"
    assert item.publisher == "Example Wire"
    assert item.snippet == "A short summary."
    assert item.published_at == datetime(2026, 1, 5, 14, 30, tzinfo=UTC)


def test_search_news_skips_articles_missing_url_or_title(
    monkeypatch, _newsapi_key
) -> None:
    payload = {"articles": [{"title": None, "url": "https://example.com/a"}]}
    monkeypatch.setattr(
        news_api_client.requests, "get", lambda *a, **k: _FakeResponse(payload)
    )

    items = news_api_client.search_news("Technology", scope="sector")

    assert items == []


def test_search_news_returns_empty_without_api_key(monkeypatch) -> None:
    monkeypatch.setenv("NEWSAPI_API_KEY", "")
    get_settings.cache_clear()

    items = news_api_client.search_news("Technology", scope="sector")

    assert items == []
    get_settings.cache_clear()


def test_search_news_returns_empty_on_http_error(monkeypatch, _newsapi_key) -> None:
    monkeypatch.setattr(
        news_api_client.requests,
        "get",
        lambda *a, **k: _FakeResponse({}, status_code=500),
    )

    items = news_api_client.search_news("Technology", scope="sector")

    assert items == []


def test_search_news_returns_empty_on_request_exception(
    monkeypatch, _newsapi_key
) -> None:
    def _raise(*args, **kwargs):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(news_api_client.requests, "get", _raise)

    items = news_api_client.search_news("Technology", scope="sector")

    assert items == []
