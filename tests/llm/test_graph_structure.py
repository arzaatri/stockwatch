"""Exercises the full LangGraph (prepare_prompt -> call_llm -> END) with a
fake Runnable standing in for Gemini - no network access or API key needed.
"""

from datetime import UTC, datetime

from stockwatch.explain.shap_explainer import FeatureAttribution
from stockwatch.llm.graph import build_graph
from stockwatch.llm.schemas import AnomalyContext, ExplanationOutput, GraphState


class _FakeStructuredLLM:
    def invoke(self, prompt: str) -> ExplanationOutput:
        return ExplanationOutput(
            summary="stub summary",
            likely_cause_category="news_driven",
            confidence="medium",
            supporting_evidence=["stub evidence"],
        )


class _FakeLLM:
    def with_structured_output(self, schema: type) -> _FakeStructuredLLM:
        return _FakeStructuredLLM()


def _context() -> AnomalyContext:
    return AnomalyContext(
        ticker="AAPL",
        as_of=datetime.now(UTC),
        anomaly_score=-0.5,
        top_features=[
            FeatureAttribution(feature="price_zscore", value=3.0, shap_value=-0.2)
        ],
    )


def test_graph_contains_expected_nodes() -> None:
    graph = build_graph(_FakeLLM())

    node_names = set(graph.get_graph().nodes.keys())

    assert {"prepare_prompt", "call_llm"}.issubset(node_names)


def test_graph_produces_a_structured_explanation() -> None:
    graph = build_graph(_FakeLLM())

    result = graph.invoke(GraphState(context=_context()))

    assert result["prompt"] is not None
    assert result["explanation"].summary == "stub summary"
    assert result["explanation"].likely_cause_category == "news_driven"
