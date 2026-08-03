from datetime import UTC, datetime

from stockwatch.explain.shap_explainer import FeatureAttribution
from stockwatch.ingestion.yfinance_client import NewsItem
from stockwatch.llm.nodes import call_llm, prepare_prompt
from stockwatch.llm.schemas import AnomalyContext, ExplanationOutput, GraphState


def _context(recent_news: list[NewsItem] | None = None) -> AnomalyContext:
    return AnomalyContext(
        ticker="AAPL",
        as_of=datetime.now(UTC),
        anomaly_score=-0.42,
        top_features=[
            FeatureAttribution(feature="price_zscore", value=3.1, shap_value=-0.2)
        ],
        sector="Technology",
        industry="Consumer Electronics",
        recent_news=recent_news or [],
    )


def test_prepare_prompt_includes_ticker_and_top_features() -> None:
    update = prepare_prompt(GraphState(context=_context()))

    assert "AAPL" in update["prompt"]
    assert "price_zscore" in update["prompt"]


def test_prepare_prompt_handles_missing_rating_and_news() -> None:
    update = prepare_prompt(GraphState(context=_context()))

    assert "No rating data available." in update["prompt"]
    assert "None available." in update["prompt"]


def test_prepare_prompt_tags_news_by_scope_and_includes_snippet() -> None:
    news = [
        NewsItem(
            scope="company",
            scope_key="AAPL",
            headline="Apple beats earnings",
            link="https://example.com/company",
            publisher="Example Wire",
            snippet="A strong quarter.",
            published_at=datetime.now(UTC),
        ),
        NewsItem(
            scope="sector",
            scope_key="Technology",
            headline="Tech sector rallies",
            link="https://example.com/sector",
            publisher="Example Wire",
            snippet=None,
            published_at=datetime.now(UTC),
        ),
        NewsItem(
            scope="industry",
            scope_key="Consumer Electronics",
            headline="Chip shortage eases",
            link="https://example.com/industry",
            publisher="Example Wire",
            published_at=datetime.now(UTC),
        ),
    ]

    update = prepare_prompt(GraphState(context=_context(recent_news=news)))
    prompt = update["prompt"]

    assert "[company] Apple beats earnings" in prompt
    assert "A strong quarter." in prompt
    assert "[sector] Tech sector rallies" in prompt
    assert "[industry] Chip shortage eases" in prompt


class _FakeStructuredLLM:
    def invoke(self, prompt: str) -> ExplanationOutput:
        assert isinstance(prompt, str)
        return ExplanationOutput(
            summary="test summary",
            likely_cause_category="unclear",
            confidence="low",
            supporting_evidence=[],
        )


class _FakeLLM:
    def with_structured_output(self, schema: type) -> _FakeStructuredLLM:
        return _FakeStructuredLLM()


def test_call_llm_returns_an_explanation() -> None:
    state = GraphState(context=_context(), prompt="a prompt")

    update = call_llm(state, llm=_FakeLLM())

    assert isinstance(update["explanation"], ExplanationOutput)
    assert update["explanation"].summary == "test summary"
