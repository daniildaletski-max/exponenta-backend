import logging
from datetime import datetime, timezone
from typing import Dict, Any, List

import numpy as np

from prediction.engine import get_engine
from prediction.features import fetch_ohlcv, compute_features

logger = logging.getLogger(__name__)


def predict_trends_real(tickers: List[str], real_prices: Dict[str, float] = None) -> Dict[str, Any]:
    if real_prices is None:
        real_prices = {}

    engine = get_engine()
    results = {}

    for ticker in tickers:
        try:
            ml_result = engine.predict(ticker)

            if "error" in ml_result:
                results[ticker] = _fallback_prediction(ticker, real_prices.get(ticker, 0))
                continue

            current_price = ml_result["price"]
            trend_5d = ml_result["predicted_trend_pct"]
            confidence = ml_result["confidence"]
            vol_ann = ml_result["volatility_ann"] / 100
            direction = ml_result["direction"]

            daily_vol = vol_ann / np.sqrt(252)
            forecasts = {}
            bands = {}

            for horizon, label in [(1, "1d"), (3, "3d"), (5, "5d"), (20, "20d")]:
                scale = horizon / 5.0
                pct = round(trend_5d * scale, 2)
                pred_price = round(current_price * (1 + pct / 100), 2)
                spread = current_price * daily_vol * np.sqrt(horizon) * 1.96

                forecasts[label] = pct
                bands[label] = {
                    "lower": round(pred_price - spread, 2),
                    "upper": round(pred_price + spread, 2),
                    "predicted_price": pred_price,
                }

            model_probas = ml_result.get("model_probas", {})
            model_weights = {}
            if "xgb" in model_probas:
                model_weights["XGBoost"] = ml_result.get("model_weights", {}).get("xgboost", 0.33)
            if "lgb" in model_probas:
                model_weights["LightGBM"] = ml_result.get("model_weights", {}).get("lightgbm", 0.33)
            if "cat" in model_probas:
                model_weights["CatBoost"] = ml_result.get("model_weights", {}).get("catboost", 0.33)

            shap = ml_result.get("shap_factors", [])
            reasoning = _build_reasoning(ticker, direction, trend_5d, confidence, shap)

            results[ticker] = {
                "ticker": ticker,
                "forecast": forecasts,
                "confidence": round(confidence),
                "confidence_bands": bands,
                "model_weights": model_weights,
                "last_price": current_price,
                "reasoning": reasoning,
                "direction": direction,
                "recommendation": ml_result.get("recommendation", "HOLD"),
                "model_accuracy": ml_result.get("model_accuracy", 0),
                "model_agreement": ml_result.get("model_agreement", 0),
                "shap_factors": shap[:5],
                "source": "ml_ensemble_v2",
                "predicted_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.warning(f"Prediction for {ticker} failed: {e}")
            results[ticker] = _fallback_prediction(ticker, real_prices.get(ticker, 0))

    return results


def _build_reasoning(ticker: str, direction: str, trend: float, confidence: float, shap: list) -> str:
    dir_text = "upward" if direction == "bullish" else "downward" if direction == "bearish" else "sideways"
    conf_text = "high" if confidence > 70 else "moderate" if confidence > 50 else "low"

    reasoning = (
        f"XGBoost+LightGBM+CatBoost ensemble predicts {dir_text} movement of {abs(trend):.1f}% "
        f"over 5 trading days with {conf_text} confidence ({confidence:.0f}%). "
    )

    if shap:
        top_factors = []
        for s in shap[:3]:
            fname = s.get("feature", "").replace("_", " ")
            sdir = s.get("direction", "neutral")
            top_factors.append(f"{fname} ({sdir})")
        if top_factors:
            reasoning += f"Key drivers: {', '.join(top_factors)}. "

    if abs(trend) < 1:
        reasoning += "Low magnitude suggests range-bound action; consider waiting for clearer signals."
    elif direction == "bullish":
        reasoning += "Momentum and technical alignment support the bullish thesis."
    else:
        reasoning += "Risk management is critical; consider protective positioning."

    return reasoning


def _fallback_prediction(ticker: str, price: float) -> dict:
    return {
        "ticker": ticker,
        "forecast": {"1d": 0, "3d": 0, "5d": 0, "20d": 0},
        "confidence": 0,
        "confidence_bands": {},
        "model_weights": {"XGBoost": 0.34, "LightGBM": 0.33, "CatBoost": 0.33},
        "last_price": price,
        "reasoning": f"Insufficient data to generate ML prediction for {ticker}. Model requires 200+ trading days of history.",
        "direction": "neutral",
        "recommendation": "HOLD",
        "source": "insufficient_data",
        "predicted_at": datetime.now(timezone.utc).isoformat(),
    }


def get_recommendations_real(tickers: List[str]) -> Dict[str, Any]:
    engine = get_engine()
    recs = []

    for ticker in tickers:
        try:
            ml_result = engine.predict(ticker)
            if "error" in ml_result:
                recs.append({
                    "ticker": ticker,
                    "action": "hold",
                    "conviction": 0.3,
                    "rationale": f"Insufficient data for {ticker} — defaulting to hold",
                    "target_weight": 0.1,
                })
                continue

            rec = ml_result.get("recommendation", "HOLD")
            confidence = ml_result.get("confidence", 50) / 100
            trend = ml_result.get("predicted_trend_pct", 0)
            direction = ml_result.get("direction", "neutral")
            sharpe = ml_result.get("sharpe_estimate", 0)

            action_map = {
                "STRONG_BUY": "buy",
                "BUY": "accumulate",
                "HOLD": "hold",
                "SELL": "sell",
                "STRONG_SELL": "sell",
            }
            action = action_map.get(rec, "hold")

            conviction = round(min(0.95, max(0.2, confidence * 0.7 + abs(trend) / 20 * 0.3)), 2)

            if action in ("buy", "accumulate"):
                target_weight = round(min(0.25, 0.05 + conviction * 0.15), 2)
            elif action == "sell":
                target_weight = 0.0
            else:
                target_weight = round(0.05 + conviction * 0.05, 2)

            shap = ml_result.get("shap_factors", [])
            top_driver = shap[0]["feature"].replace("_", " ") if shap else "technical signals"

            rationale = (
                f"ML ensemble ({ml_result.get('model_accuracy', 0):.0f}% accuracy) signals {direction} "
                f"with {trend:+.1f}% 5-day forecast. "
                f"Primary driver: {top_driver}. "
                f"Risk level: {ml_result.get('risk_level', 'MEDIUM')}"
            )

            recs.append({
                "ticker": ticker,
                "action": action,
                "conviction": conviction,
                "rationale": rationale,
                "target_weight": target_weight,
            })

        except Exception as e:
            logger.warning(f"Recommendation for {ticker} failed: {e}")
            recs.append({
                "ticker": ticker,
                "action": "hold",
                "conviction": 0.3,
                "rationale": f"Analysis unavailable for {ticker}",
                "target_weight": 0.1,
            })

    total_conviction = sum(r["conviction"] for r in recs)
    score_before = round(min(0.75, 0.4 + len(recs) * 0.03), 2)
    score_after = round(min(0.95, score_before + total_conviction / len(recs) * 0.2), 2) if recs else score_before

    return {
        "recommendations": recs,
        "portfolio_score_before": score_before,
        "portfolio_score_after": score_after,
        "source": "ml_ensemble_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
