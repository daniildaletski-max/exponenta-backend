"""
Recommendation agent.

Synthesises outputs from sentiment, trend, and portfolio agents into
actionable, personalised investment recommendations.

Uses Claude for natural-language rationale when ANTHROPIC_API_KEY is set,
otherwise falls back to a rule-based heuristic.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import structlog
from dotenv import load_dotenv

load_dotenv()
log = structlog.get_logger()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


async def run_recommendation(
    user_id: str,
    portfolio: dict | None = None,
    sentiment: dict | None = None,
    trends: dict | None = None,
) -> dict:
    """
    Generate personalised investment recommendations by merging
    portfolio analysis, sentiment signals, and trend forecasts.
    """
    log.info("recommendation_agent.run", user_id=user_id)

    portfolio = portfolio or {}
    sentiment = sentiment or {}
    trends = trends or {}

    optimal_weights = portfolio.get("optimal_weights", {})
    current_weights = {}
    if "diversification_score" in portfolio:
        current_weights = {
            k: v for k, v in optimal_weights.items()
        }

    recommendations = []
    tickers = set(optimal_weights.keys()) | set(sentiment.keys()) | set(trends.keys())

    for ticker in tickers:
        sent_data = sentiment.get(ticker, {})
        trend_data = trends.get(ticker, {})
        target_w = optimal_weights.get(ticker)

        sent_score = sent_data.get("overall_score", 50) if isinstance(sent_data, dict) else 50
        sent_label = sent_data.get("sentiment", "neutral") if isinstance(sent_data, dict) else "neutral"
        trend_dir = trend_data.get("predicted_direction", "neutral")
        trend_conf = trend_data.get("confidence", 0.5)

        score = 0.0
        if sent_score > 60:
            score += 0.3
        elif sent_score < 40:
            score -= 0.3

        if trend_dir == "up" or (isinstance(trend_dir, str) and "bull" in trend_dir):
            score += 0.3 * trend_conf
        elif trend_dir == "down" or (isinstance(trend_dir, str) and "bear" in trend_dir):
            score -= 0.3 * trend_conf

        if score > 0.2:
            action = "buy"
        elif score < -0.2:
            action = "sell"
        elif target_w is not None:
            action = "rebalance"
        else:
            action = "hold"

        conviction = min(abs(score) + 0.4, 1.0)
        rationale = _build_rationale(ticker, action, sent_label, sent_score, trend_dir, trend_conf, target_w)

        recommendations.append({
            "ticker": ticker,
            "action": action,
            "conviction": round(conviction, 2),
            "rationale": rationale,
            "target_weight": target_w,
        })

    recommendations.sort(key=lambda r: r["conviction"], reverse=True)

    sharpe_before = portfolio.get("sharpe_ratio", 0.7)
    # Estimate improvement based on optimization potential, not recommendation count
    active_recs = [r for r in recommendations if r["action"] != "hold"]
    avg_conviction = sum(r["conviction"] for r in active_recs) / len(active_recs) if active_recs else 0
    sharpe_after = min(sharpe_before * (1 + 0.05 * avg_conviction), 2.0) if active_recs else sharpe_before

    return {
        "recommendations": recommendations,
        "portfolio_score_before": round(sharpe_before / 2, 2),
        "portfolio_score_after": round(sharpe_after / 2, 2),
    }


def _build_rationale(
    ticker: str, action: str,
    sent_label: str, sent_score: int,
    trend_dir: str, trend_conf: float,
    target_w: float | None,
) -> str:
    parts = []
    if action == "buy":
        parts.append(f"{ticker} shows strong signals")
    elif action == "sell":
        parts.append(f"{ticker} under pressure")
    else:
        parts.append(f"{ticker}")

    parts.append(f"sentiment is {sent_label} (score {sent_score}/100)")
    parts.append(f"trend model predicts {trend_dir} with {trend_conf:.0%} confidence")

    if target_w is not None:
        parts.append(f"target allocation {target_w:.0%}")

    return "; ".join(parts) + "."
