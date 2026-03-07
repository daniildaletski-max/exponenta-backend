"""
LangGraph multi-agent orchestrator.

Coordinates specialised agents (sentiment, trend, portfolio, recommendation)
into a unified analysis pipeline.

Graph topology:
  [start] → sentiment_agent → trend_agent → portfolio_agent → recommendation_agent → [end]
"""

from __future__ import annotations

from typing import TypedDict

import structlog
from langgraph.graph import END, StateGraph

from agents.sentiment_agent import run_sentiment_analysis
from agents.trend_agent import run_trend
from agents.portfolio_agent import run_portfolio
from agents.recommendation_agent import run_recommendation

log = structlog.get_logger()


class AgentState(TypedDict, total=False):
    user_id: str
    tickers: list[str]
    horizon_days: int
    holdings: list[dict]
    sentiment: dict
    trends: dict
    portfolio_analysis: dict
    recommendations: dict
    errors: list[str]


def _append_error(state: AgentState, error: str) -> list[str]:
    errors = list(state.get("errors", []))
    errors.append(error)
    return errors


async def _sentiment_node(state: AgentState) -> AgentState:
    try:
        result = await run_sentiment_analysis(tickers=state["tickers"])
        return {**state, "sentiment": result}
    except Exception as e:
        log.exception("orchestrator.sentiment_failed")
        return {**state, "sentiment": {}, "errors": _append_error(state, f"sentiment: {e}")}


async def _trend_node(state: AgentState) -> AgentState:
    try:
        result = await run_trend(state["tickers"], state.get("horizon_days", 30))
        return {**state, "trends": result}
    except Exception as e:
        log.exception("orchestrator.trend_failed")
        return {**state, "trends": {}, "errors": _append_error(state, f"trend: {e}")}


async def _portfolio_node(state: AgentState) -> AgentState:
    try:
        result = await run_portfolio(
            state.get("user_id", "anonymous"),
            sentiment=state.get("sentiment", {}),
            trends=state.get("trends", {}),
        )
        return {**state, "portfolio_analysis": result}
    except Exception as e:
        log.exception("orchestrator.portfolio_failed")
        return {**state, "portfolio_analysis": {}, "errors": _append_error(state, f"portfolio: {e}")}


async def _recommendation_node(state: AgentState) -> AgentState:
    try:
        result = await run_recommendation(
            state.get("user_id", "anonymous"),
            portfolio=state.get("portfolio_analysis", {}),
            sentiment=state.get("sentiment", {}),
            trends=state.get("trends", {}),
        )
        return {**state, "recommendations": result}
    except Exception as e:
        log.exception("orchestrator.recommendation_failed")
        return {**state, "recommendations": {}, "errors": _append_error(state, f"recommendation: {e}")}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("sentiment", _sentiment_node)
    graph.add_node("trend", _trend_node)
    graph.add_node("portfolio", _portfolio_node)
    graph.add_node("recommendation", _recommendation_node)

    graph.set_entry_point("sentiment")
    graph.add_edge("sentiment", "trend")
    graph.add_edge("trend", "portfolio")
    graph.add_edge("portfolio", "recommendation")
    graph.add_edge("recommendation", END)

    return graph.compile()


_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


async def run_full_analysis(
    tickers: list[str],
    user_id: str = "anonymous",
    horizon_days: int = 30,
) -> dict:
    """Run the complete multi-agent pipeline and return all results."""
    graph = _get_graph()
    try:
        final = await graph.ainvoke({
            "tickers": tickers,
            "user_id": user_id,
            "horizon_days": horizon_days,
        })
        return {
            "sentiment": final.get("sentiment", {}),
            "trends": final.get("trends", {}),
            "portfolio_analysis": final.get("portfolio_analysis", {}),
            "recommendations": final.get("recommendations", {}),
        }
    except Exception as e:
        log.exception("orchestrator.full_analysis_failed")
        return {"error": str(e)}
