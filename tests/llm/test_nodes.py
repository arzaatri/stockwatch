from datetime import UTC, datetime

from stockwatch.explain.shap_explainer import FeatureAttribution
from stockwatch.llm.nodes import call_llm, prepare_prompt
from stockwatch.llm.schemas import AnomalyContext, ExplanationOutput, GraphState


def _context() -> AnomalyContext:
    return AnomalyContext(
        ticker="AAPL",
        as_of=datetime.now(UTC),
        anomaly_score=-0.42,
        top_features=[
            FeatureAttribution(feature="price_zscore", value=3.1, shap_value=-0.2)
        ],
        sector="Technology",
        industry="Consumer Electronics",
    )


def test_prepare_prompt_includes_ticker_and_top_features() -> None:
    update = prepare_prompt(GraphState(context=_context()))

    assert "AAPL" in update["prompt"]
    assert "price_zscore" in update["prompt"]


def test_prepare_prompt_handles_missing_rating_and_news() -> None:
    update = prepare_prompt(GraphState(context=_context()))

    assert "No rating data available." in update["prompt"]
    assert "None available." in update["prompt"]


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
