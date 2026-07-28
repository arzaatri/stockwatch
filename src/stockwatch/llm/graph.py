"""Builds the anomaly-explanation LangGraph: prepare_prompt -> call_llm -> END.

`build_graph` takes the LLM client as a parameter (not constructed internally)
so tests can pass a fake Runnable and never touch the network or need an API
key; `build_default_graph` is what real pipeline code calls, wiring up Gemini.
"""

from functools import partial

from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from stockwatch.config import get_settings
from stockwatch.llm.nodes import call_llm, prepare_prompt
from stockwatch.llm.schemas import GraphState


def build_graph(llm: Runnable) -> CompiledStateGraph:
    graph = StateGraph(GraphState)
    graph.add_node("prepare_prompt", prepare_prompt)
    graph.add_node("call_llm", partial(call_llm, llm=llm))
    graph.set_entry_point("prepare_prompt")
    graph.add_edge("prepare_prompt", "call_llm")
    graph.add_edge("call_llm", END)
    return graph.compile()


def build_default_graph() -> CompiledStateGraph:
    settings = get_settings()
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.2,
    )
    return build_graph(llm)
