"""LangGraph node functions. `call_llm` is the only node that touches the
network - it takes the LLM client as a parameter rather than constructing one
itself, which is what lets graph.py swap in a fake Runnable for tests.
"""

from langchain_core.runnables import Runnable

from stockwatch.llm.schemas import ExplanationOutput, GraphState


def prepare_prompt(state: GraphState) -> dict:
    context = state.context

    features_text = "\n".join(
        f"- {feature.feature}: value={feature.value:.4f}, shap_contribution={feature.shap_value:.4f}"
        for feature in context.top_features
    )
    news_text = (
        "\n".join(
            f"- [{item.scope}] {item.headline} ({item.publisher})"
            + (f": {item.snippet}" if item.snippet else "")
            for item in context.recent_news
        )
        or "None available."
    )
    rating_text = (
        f"strong_buy={context.rating.strong_buy}, buy={context.rating.buy}, hold={context.rating.hold}, "
        f"sell={context.rating.sell}, strong_sell={context.rating.strong_sell}"
        if context.rating
        else "No rating data available."
    )

    prompt = f"""You are a financial analyst assistant. A statistical anomaly detector \
(Isolation Forest) flagged unusual trading behavior for {context.ticker} at \
{context.as_of.isoformat()} (anomaly score: {context.anomaly_score:.4f}, more negative \
means more anomalous).

Sector: {context.sector or "unknown"} / Industry: {context.industry or "unknown"}

The features that contributed most to this anomaly (via SHAP):
{features_text}

Current analyst rating consensus: {rating_text}

Recent news (tagged by scope - [company] is {context.ticker}-specific, [sector]/[industry] \
are broader market context, not necessarily about {context.ticker} directly):
{news_text}

Explain the most likely real-world cause(s) of this anomaly, using only the context \
given. Weigh company-scoped news more heavily than sector/industry news unless the \
evidence points to a sector- or industry-wide move. Be concise and specific about which \
signals support your reasoning."""

    return {"prompt": prompt}


def call_llm(state: GraphState, llm: Runnable) -> dict:
    structured_llm = llm.with_structured_output(ExplanationOutput)
    explanation = structured_llm.invoke(state.prompt)
    return {"explanation": explanation}
