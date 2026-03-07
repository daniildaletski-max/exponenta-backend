"""
LangGraph Portfolio Analysis Agent.

Pipeline:
  1. [fetch_prices]     — Polygon.io daily bars + snapshot per holding
  2. [compute_metrics]  — MPT risk metrics via core.portfolio_optimizer
  3. [optimize]         — PyPortfolioOpt max-Sharpe + RL PPO action ranking
  4. [build_response]   — merge into final response with suggestions

Gracefully degrades when PyPortfolioOpt or Polygon API key is missing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TypedDict

import numpy as np
import structlog
from langgraph.graph import END, StateGraph

from data.polygon_client import PolygonClient
from core.portfolio_optimizer import PortfolioOptimizer, PortfolioMetrics, OptimizationResult
from core.rl_optimizer import RLPortfolioOptimizer

log = structlog.get_logger()

LOOKBACK_DAYS = 365
BENCHMARK_TICKER = "SPY"


# ── State schema ──────────────────────────────────────────────────────

class PortfolioState(TypedDict, total=False):
    holdings: list[dict]                     # [{ticker, quantity, avg_price?}]
    prices: dict[str, float]                 # ticker -> current price
    daily_returns: dict[str, np.ndarray]     # ticker -> daily return series
    benchmark_returns: np.ndarray | None
    current_weights: dict[str, float]
    total_value: float
    metrics: PortfolioMetrics | None
    optimization: OptimizationResult | None
    rl_actions: list[dict]
    result: dict
    error: str | None


# ── Node 1: fetch prices from Polygon ────────────────────────────────

async def fetch_prices(state: PortfolioState) -> PortfolioState:
    """Fetch daily bars + current price for each holding + benchmark."""
    prices: dict[str, float] = {}
    daily_returns: dict[str, np.ndarray] = {}

    tickers = [h["ticker"] for h in state["holdings"]] + [BENCHMARK_TICKER]

    try:
        client = PolygonClient()
        try:
            for ticker in tickers:
                bars = await client.get_daily_bars(ticker, lookback_days=LOOKBACK_DAYS)
                if bars and len(bars) > 1:
                    closes = np.array([b["close"] for b in bars], dtype=np.float64)
                    rets = np.diff(closes) / closes[:-1]
                    daily_returns[ticker] = rets

                snap = await client.get_snapshot(ticker)
                prices[ticker] = snap.price
        finally:
            await client.close()
    except Exception:
        log.exception("fetch_prices_failed — using avg_price fallback")
        for h in state["holdings"]:
            prices.setdefault(h["ticker"], h.get("avg_price", 100.0))

    # Compute current weights
    total = 0.0
    for h in state["holdings"]:
        price = prices.get(h["ticker"], h.get("avg_price", 100.0))
        total += h["quantity"] * price

    current_weights: dict[str, float] = {}
    for h in state["holdings"]:
        price = prices.get(h["ticker"], h.get("avg_price", 100.0))
        mv = h["quantity"] * price
        current_weights[h["ticker"]] = mv / total if total > 0 else 0

    benchmark_returns = daily_returns.pop(BENCHMARK_TICKER, None)

    return {
        **state,
        "prices": prices,
        "daily_returns": daily_returns,
        "benchmark_returns": benchmark_returns,
        "current_weights": current_weights,
        "total_value": round(total, 2),
    }


# ── Node 2: compute risk metrics ─────────────────────────────────────

async def compute_metrics(state: PortfolioState) -> PortfolioState:
    """Run MPT metrics: Sharpe, Sortino, max drawdown, beta, diversification."""
    optimizer = PortfolioOptimizer()

    returns = state.get("daily_returns", {})
    weights = state.get("current_weights", {})
    benchmark = state.get("benchmark_returns")

    if not returns or not weights:
        return {**state, "metrics": None, "error": "No price data available"}

    # Align return series to same length
    min_len = min(len(r) for r in returns.values()) if returns else 0
    if min_len < 2:
        return {**state, "metrics": None, "error": "Insufficient return data for metrics"}
    if benchmark is not None:
        min_len = min(min_len, len(benchmark))
        benchmark = benchmark[-min_len:]
    aligned = {t: r[-min_len:] for t, r in returns.items()}

    try:
        metrics = optimizer.compute_metrics(aligned, weights, benchmark)
    except Exception as e:
        log.exception("metrics_failed")
        metrics = PortfolioMetrics(
            sharpe_ratio=0, sortino_ratio=0, max_drawdown=0,
            beta=1.0, annual_return=0, annual_volatility=0,
            diversification_score=0.5,
        )

    return {**state, "metrics": metrics}


# ── Node 3: optimize + RL actions ─────────────────────────────────────

async def optimize(state: PortfolioState) -> PortfolioState:
    """Run PyPortfolioOpt + RL PPO action ranking."""
    optimizer = PortfolioOptimizer()
    rl = RLPortfolioOptimizer()

    returns = state.get("daily_returns", {})
    current_weights = state.get("current_weights", {})

    # MPT optimization
    optimization: OptimizationResult | None = None
    if returns:
        min_len = min(len(r) for r in returns.values())
        aligned = {t: r[-min_len:] for t, r in returns.items()}
        try:
            optimization = optimizer.optimize(aligned, objective="max_sharpe")
        except Exception as e:
            log.exception("optimization_failed")

    # RL action ranking
    predicted_returns: dict[str, float] = {}
    sentiment_scores: dict[str, float] = {}
    for ticker, rets in returns.items():
        predicted_returns[ticker] = float(np.mean(rets[-20:]) * 252) if len(rets) >= 20 else 0
        sentiment_scores[ticker] = 0.0  # neutral when no sentiment data

    rl_actions_raw = rl.rank_actions(current_weights, predicted_returns, sentiment_scores)
    rl_actions = [
        {
            "ticker": a.ticker,
            "action": a.action,
            "magnitude": round(a.magnitude, 4),
            "conviction": round(a.conviction, 3),
        }
        for a in rl_actions_raw
    ]

    return {**state, "optimization": optimization, "rl_actions": rl_actions}


# ── Node 4: build response ───────────────────────────────────────────

async def build_response(state: PortfolioState) -> PortfolioState:
    """Assemble final analysis from metrics + optimization + RL."""
    metrics = state.get("metrics")
    optimization = state.get("optimization")
    rl_actions = state.get("rl_actions", [])
    current_weights = state.get("current_weights", {})
    prices = state.get("prices", {})
    total_value = state.get("total_value", 0)

    # Current allocation
    current_allocation = []
    for h in state["holdings"]:
        ticker = h["ticker"]
        price = prices.get(ticker, 0)
        mv = h["quantity"] * price
        current_allocation.append({
            "ticker": ticker,
            "weight": round(current_weights.get(ticker, 0), 4),
            "current_price": round(price, 2),
            "market_value": round(mv, 2),
        })

    # Risk level classification
    if metrics:
        vol = metrics.annual_volatility
        if vol > 0.25:
            risk_level = "aggressive"
        elif vol > 0.15:
            risk_level = "moderate"
        else:
            risk_level = "conservative"
    else:
        risk_level = "unknown"

    # Rebalancing actions from optimization + RL
    recommended_rebalance = []
    target_weights = optimization.weights if optimization else current_weights
    for rl_act in rl_actions:
        ticker = rl_act["ticker"]
        cw = current_weights.get(ticker, 0)
        tw = target_weights.get(ticker, cw)
        diff = tw - cw

        if abs(diff) < 0.01 and rl_act["action"] == "hold":
            continue

        if diff > 0.01:
            action = "increase"
            rationale = f"Underweight by {abs(diff):.1%}; optimization suggests higher allocation"
        elif diff < -0.01:
            action = "decrease"
            rationale = f"Overweight by {abs(diff):.1%}; reduce to improve risk-adjusted returns"
        else:
            action = rl_act["action"]
            rationale = f"RL signal: {action} with {rl_act['conviction']:.0%} conviction"

        recommended_rebalance.append({
            "ticker": ticker,
            "action": action,
            "current_weight": round(cw, 4),
            "target_weight": round(tw, 4),
            "conviction": rl_act["conviction"],
            "rationale": rationale,
        })

    # Suggestions
    suggestions = []
    if metrics and metrics.diversification_score < 0.5:
        suggestions.append("Portfolio is highly correlated — consider adding uncorrelated assets (bonds, international, alternatives)")
    if metrics and metrics.max_drawdown < -0.20:
        suggestions.append(f"Max drawdown of {metrics.max_drawdown:.1%} is elevated — consider reducing concentration risk")
    if metrics and metrics.sharpe_ratio < 0.5:
        suggestions.append("Risk-adjusted returns are below average — optimization may improve Sharpe ratio")
    if optimization and metrics:
        delta = optimization.expected_sharpe - metrics.sharpe_ratio
        if delta > 0.1:
            suggestions.append(f"Rebalancing could improve Sharpe ratio by +{delta:.2f}")
    if not suggestions:
        suggestions.append("Portfolio is well balanced — no immediate action required")

    risk_metrics = {}
    if metrics:
        risk_metrics = {
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "max_drawdown": metrics.max_drawdown,
            "beta": metrics.beta,
            "annual_return": metrics.annual_return,
            "annual_volatility": metrics.annual_volatility,
            "diversification_score": metrics.diversification_score,
        }

    result = {
        "total_value": total_value,
        "current_allocation": current_allocation,
        "risk_metrics": risk_metrics,
        "sharpe_ratio": metrics.sharpe_ratio if metrics else 0,
        "sortino_ratio": metrics.sortino_ratio if metrics else 0,
        "max_drawdown": metrics.max_drawdown if metrics else 0,
        "beta": metrics.beta if metrics else 1.0,
        "annual_return": metrics.annual_return if metrics else 0,
        "annual_volatility": metrics.annual_volatility if metrics else 0,
        "diversification_score": metrics.diversification_score if metrics else 0.5,
        "risk_level": risk_level,
        "recommended_rebalance": recommended_rebalance,
        "suggestions": suggestions,
    }

    return {**state, "result": result}


# ── Graph builder ─────────────────────────────────────────────────────

def build_portfolio_graph():
    graph = StateGraph(PortfolioState)

    graph.add_node("fetch_prices", fetch_prices)
    graph.add_node("compute_metrics", compute_metrics)
    graph.add_node("optimize", optimize)
    graph.add_node("build_response", build_response)

    graph.set_entry_point("fetch_prices")
    graph.add_edge("fetch_prices", "compute_metrics")
    graph.add_edge("compute_metrics", "optimize")
    graph.add_edge("optimize", "build_response")
    graph.add_edge("build_response", END)

    return graph.compile()


# ── Graph cache ──────────────────────────────────────────────────────

_compiled_portfolio_graph = None


def _get_portfolio_graph():
    global _compiled_portfolio_graph
    if _compiled_portfolio_graph is None:
        _compiled_portfolio_graph = build_portfolio_graph()
    return _compiled_portfolio_graph


# ── Public interface ──────────────────────────────────────────────────

async def run_portfolio_analysis(holdings: list[dict]) -> dict:
    """
    Entry point called by the FastAPI endpoint.

    Args:
        holdings: [{"ticker": "AAPL", "quantity": 50, "avg_price": 178.50}, ...]

    Returns full portfolio analysis: metrics, allocation, rebalancing, suggestions.
    Never raises — returns structured fallback on any failure.
    """
    if not holdings:
        return {"error": "Provide at least one holding"}

    for h in holdings:
        h["ticker"] = h["ticker"].upper().strip()

    try:
        graph = _get_portfolio_graph()
        final_state = await graph.ainvoke({"holdings": holdings})

        if "error" in final_state and final_state["error"]:
            log.warning("portfolio_graph_error", error=final_state["error"])
            return _build_fallback(holdings)

        result = final_state.get("result")
        if result is None:
            return _build_fallback(holdings)

        return result

    except Exception as e:
        log.exception("portfolio_analysis_crashed")
        return _build_fallback(holdings)


def _build_fallback(holdings: list[dict]) -> dict:
    """Graceful fallback when the full pipeline fails."""
    n = len(holdings)
    equal_weight = round(1 / n, 4) if n > 0 else 0

    total_value = 0.0
    current_allocation = []
    for h in holdings:
        price = h.get("avg_price") or 100.0
        mv = h["quantity"] * price
        total_value += mv
        current_allocation.append({
            "ticker": h["ticker"],
            "weight": equal_weight,
            "current_price": round(price, 2),
            "market_value": round(mv, 2),
        })

    return {
        "total_value": round(total_value, 2),
        "current_allocation": current_allocation,
        "risk_metrics": {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "beta": 1.0,
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "diversification_score": 0.5,
        },
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown": 0.0,
        "beta": 1.0,
        "annual_return": 0.0,
        "annual_volatility": 0.0,
        "diversification_score": 0.5,
        "risk_level": "unknown",
        "recommended_rebalance": [],
        "suggestions": [
            "Analysis ran with limited data — results are approximate",
            "Connect Polygon.io API key for live pricing and full metrics",
        ],
    }


async def run_portfolio(
    user_id: str,
    sentiment: dict | None = None,
    trends: dict | None = None,
) -> dict:
    """Async interface for the orchestrator — runs actual portfolio analysis."""
    log.info("portfolio_agent.run", user_id=user_id)

    # Try to get user holdings from settings
    try:
        from user_settings import load_settings
        settings = load_settings()
        holdings = settings.get("holdings", [
            {"ticker": "AAPL", "quantity": 50, "avg_price": 178.50},
            {"ticker": "VOO", "quantity": 30, "avg_price": 420.00},
            {"ticker": "NVDA", "quantity": 20, "avg_price": 480.00},
        ])
    except Exception:
        holdings = [
            {"ticker": "AAPL", "quantity": 50, "avg_price": 178.50},
            {"ticker": "VOO", "quantity": 30, "avg_price": 420.00},
            {"ticker": "NVDA", "quantity": 20, "avg_price": 480.00},
        ]

    try:
        result = await run_portfolio_analysis(holdings)
        return result
    except Exception as e:
        log.exception("portfolio_agent.run_failed")
        return _build_fallback(holdings)
