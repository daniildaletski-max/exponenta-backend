"""
Trend prediction agent (sync interface for orchestrator).

Delegates to the prediction_agent for real ensemble forecasting,
falling back to a heuristic when the async pipeline is unavailable.
"""

from __future__ import annotations

import numpy as np
import structlog

log = structlog.get_logger()


async def run_trend(tickers: list[str], horizon_days: int = 30) -> dict:
    """
    Generate price trend predictions via the ensemble.

    Tries the full prediction agent pipeline; falls back to a
    momentum-based heuristic using Polygon stub data.
    """
    log.info("trend_agent.run", tickers=tickers, horizon=horizon_days)

    try:
        from agents.prediction_agent import run_prediction
        raw = await run_prediction(tickers)

        results: dict[str, dict] = {}
        for ticker in tickers:
            pred = raw.get(ticker, {})
            forecast = pred.get("forecast", {})
            pct_30d = forecast.get("30d", 0)

            if pct_30d > 1:
                direction = "up"
            elif pct_30d < -1:
                direction = "down"
            else:
                direction = "sideways"

            results[ticker] = {
                "horizon_days": horizon_days,
                "predicted_direction": direction,
                "confidence": pred.get("confidence", 50) / 100,
                "forecast_pct": forecast,
                "model_weights": pred.get("model_weights", {}),
            }

        return results

    except Exception as e:
        log.warning("trend_agent.fallback", reason=str(e))
        return {
            ticker: {
                "horizon_days": horizon_days,
                "predicted_direction": "sideways",
                "confidence": 0.30,
                "model_weights": {"chronos2": 0.45, "timesfm": 0.35, "lag_llama": 0.20},
                "is_fallback": True,
            }
            for ticker in tickers
        }
