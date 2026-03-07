import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prediction_history.json")


def _load_history() -> List[Dict[str, Any]]:
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load prediction history: {e}")
    return []


def _save_history(history: List[Dict[str, Any]]):
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to save prediction history: {e}")


class PredictionTracker:

    def track_prediction(self, ticker: str, prediction_data: Dict[str, Any]):
        try:
            history = _load_history()
            entry = {
                "ticker": ticker.upper(),
                "predicted_direction": prediction_data.get("direction", "neutral"),
                "predicted_change": prediction_data.get("predicted_trend_pct", 0),
                "confidence": prediction_data.get("confidence", 0),
                "actual_outcome": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_used": "ensemble",
            }
            history.append(entry)
            _save_history(history)
        except Exception as e:
            logger.error(f"Failed to track prediction for {ticker}: {e}")

    def record_outcome(self, ticker: str, actual_change: float):
        try:
            history = _load_history()
            for entry in reversed(history):
                if entry["ticker"] == ticker.upper() and entry["actual_outcome"] is None:
                    entry["actual_outcome"] = actual_change
                    break
            _save_history(history)
        except Exception as e:
            logger.error(f"Failed to record outcome for {ticker}: {e}")

    def get_accuracy_stats(self) -> Dict[str, Any]:
        try:
            history = _load_history()
            completed = [h for h in history if h["actual_outcome"] is not None]

            if not completed:
                return {
                    "total_predictions": len(history),
                    "completed_predictions": 0,
                    "overall_accuracy": 0,
                    "per_ticker": {},
                    "message": "No completed predictions yet",
                }

            correct = 0
            per_ticker: Dict[str, Dict[str, int]] = {}

            for entry in completed:
                predicted_dir = entry.get("predicted_direction", "neutral")
                actual = entry.get("actual_outcome", 0)
                ticker = entry["ticker"]

                if ticker not in per_ticker:
                    per_ticker[ticker] = {"correct": 0, "total": 0}
                per_ticker[ticker]["total"] += 1

                predicted_up = predicted_dir in ("bullish", 1, "1")
                actual_up = actual > 0

                if predicted_up == actual_up:
                    correct += 1
                    per_ticker[ticker]["correct"] += 1

            overall_accuracy = round(correct / len(completed) * 100, 1) if completed else 0

            ticker_stats = {}
            for t, stats in per_ticker.items():
                acc = round(stats["correct"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0
                ticker_stats[t] = {
                    "accuracy": acc,
                    "correct": stats["correct"],
                    "total": stats["total"],
                }

            return {
                "total_predictions": len(history),
                "completed_predictions": len(completed),
                "overall_accuracy": overall_accuracy,
                "per_ticker": ticker_stats,
            }
        except Exception as e:
            logger.error(f"Failed to get accuracy stats: {e}")
            return {"total_predictions": 0, "completed_predictions": 0, "overall_accuracy": 0, "per_ticker": {}, "error": str(e)}


class AdaptiveEnsemble:

    def __init__(self):
        self._history: Optional[List[Dict[str, Any]]] = None

    def _get_history(self) -> List[Dict[str, Any]]:
        if self._history is None:
            self._history = _load_history()
        return self._history

    def calculate_model_weights(self) -> Dict[str, float]:
        try:
            history = self._get_history()
            completed = [h for h in history if h["actual_outcome"] is not None]
            recent = completed[-30:] if len(completed) > 30 else completed

            if not recent:
                return {"xgboost": 0.34, "lightgbm": 0.33, "catboost": 0.33}

            model_scores = {"xgboost": 0, "lightgbm": 0, "catboost": 0}
            model_counts = {"xgboost": 0, "lightgbm": 0, "catboost": 0}

            for entry in recent:
                predicted_dir = entry.get("predicted_direction", "neutral")
                actual = entry.get("actual_outcome", 0)
                predicted_up = predicted_dir in ("bullish", 1, "1")
                actual_up = actual > 0
                is_correct = 1 if predicted_up == actual_up else 0

                for model in model_scores:
                    model_scores[model] += is_correct
                    model_counts[model] += 1

            accuracies = {}
            for model in model_scores:
                if model_counts[model] > 0:
                    accuracies[model] = model_scores[model] / model_counts[model]
                else:
                    accuracies[model] = 0.33

            total = sum(accuracies.values())
            if total > 0:
                weights = {m: round(a / total, 4) for m, a in accuracies.items()}
            else:
                weights = {"xgboost": 0.34, "lightgbm": 0.33, "catboost": 0.33}

            return weights
        except Exception as e:
            logger.error(f"Failed to calculate model weights: {e}")
            return {"xgboost": 0.34, "lightgbm": 0.33, "catboost": 0.33}

    def get_calibrated_confidence(self, raw_confidence: float, ticker: str) -> float:
        try:
            history = self._get_history()
            ticker_entries = [h for h in history if h["ticker"] == ticker.upper() and h["actual_outcome"] is not None]

            if len(ticker_entries) < 5:
                return raw_confidence

            recent = ticker_entries[-30:]
            correct = 0
            for entry in recent:
                predicted_dir = entry.get("predicted_direction", "neutral")
                actual = entry.get("actual_outcome", 0)
                predicted_up = predicted_dir in ("bullish", 1, "1")
                actual_up = actual > 0
                if predicted_up == actual_up:
                    correct += 1

            historical_accuracy = correct / len(recent)
            calibration_factor = 0.5 + historical_accuracy * 0.5
            calibrated = raw_confidence * calibration_factor
            return round(float(np.clip(calibrated, 5.0, 95.0)), 1)
        except Exception as e:
            logger.error(f"Failed to calibrate confidence for {ticker}: {e}")
            return raw_confidence

    def detect_market_regime(self, ticker: str) -> str:
        try:
            import yfinance as yf
            from ta.trend import ADXIndicator

            t = yf.Ticker(ticker)
            df = t.history(period="3mo")
            if df is None or len(df) < 30:
                return "ranging"

            close = df["Close"]
            high = df["High"]
            low = df["Low"]

            adx_indicator = ADXIndicator(high, low, close)
            adx = adx_indicator.adx()
            adx_val = float(adx.iloc[-1]) if not adx.empty and not np.isnan(adx.iloc[-1]) else 20

            sma_20 = close.rolling(20).mean().iloc[-1]
            sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma_20
            current_price = float(close.iloc[-1])

            volatility = float(close.pct_change().tail(20).std())
            high_volatility = volatility > 0.025

            if high_volatility and adx_val < 25:
                return "volatile"

            ma_aligned_up = current_price > sma_20 > sma_50
            ma_aligned_down = current_price < sma_20 < sma_50

            if adx_val > 25 and ma_aligned_up:
                return "trending_up"
            elif adx_val > 25 and ma_aligned_down:
                return "trending_down"
            elif adx_val < 20:
                return "ranging"
            elif high_volatility:
                return "volatile"
            else:
                return "ranging"
        except Exception as e:
            logger.error(f"Failed to detect market regime for {ticker}: {e}")
            return "ranging"


class SmartPortfolioScorer:

    def score_portfolio(self, holdings: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            if not holdings:
                return {
                    "diversification_score": 0,
                    "risk_adjusted_score": 0,
                    "momentum_score": 50,
                    "overall_grade": "F",
                }

            diversification = self._calc_diversification(holdings)
            risk_adjusted = self._calc_risk_adjusted(holdings)
            momentum = self._calc_momentum(holdings)

            overall = diversification * 0.3 + risk_adjusted * 0.4 + momentum * 0.3
            grade = self._score_to_grade(overall)

            return {
                "diversification_score": round(diversification, 1),
                "risk_adjusted_score": round(risk_adjusted, 1),
                "momentum_score": round(momentum, 1),
                "overall_grade": grade,
            }
        except Exception as e:
            logger.error(f"Failed to score portfolio: {e}")
            return {
                "diversification_score": 0,
                "risk_adjusted_score": 0,
                "momentum_score": 50,
                "overall_grade": "F",
            }

    def _calc_diversification(self, holdings: List[Dict[str, Any]]) -> float:
        try:
            import yfinance as yf

            if len(holdings) <= 1:
                return 20.0

            total_value = 0
            values = []
            sectors = set()

            for h in holdings:
                qty = h.get("quantity", 0)
                price = h.get("avg_price", 0)
                try:
                    info = yf.Ticker(h["ticker"]).info
                    price = info.get("regularMarketPrice", price) or price
                    sector = info.get("sector", "Unknown")
                    sectors.add(sector)
                except Exception:
                    sectors.add("Unknown")
                mv = price * qty
                values.append(mv)
                total_value += mv

            if total_value <= 0:
                return 20.0

            weights = [v / total_value for v in values]
            hhi = sum(w ** 2 for w in weights)
            concentration_penalty = hhi * 100
            concentration_score = max(0, 100 - concentration_penalty)

            n_holdings = len(holdings)
            holding_bonus = min(30, n_holdings * 5)

            n_sectors = len(sectors)
            sector_bonus = min(30, n_sectors * 10)

            score = concentration_score * 0.4 + holding_bonus + sector_bonus
            return float(np.clip(score, 0, 100))
        except Exception as e:
            logger.error(f"Failed to calc diversification: {e}")
            return 50.0

    def _calc_risk_adjusted(self, holdings: List[Dict[str, Any]]) -> float:
        try:
            from prediction.features import fetch_ohlcv

            returns_list = []
            weights = []
            total_value = 0

            for h in holdings:
                mv = h.get("quantity", 0) * h.get("avg_price", 0)
                total_value += mv

            for h in holdings:
                df = fetch_ohlcv(h["ticker"], days=180)
                if df is not None and len(df) > 20:
                    rets = df["close"].pct_change().dropna().tail(60).values
                    returns_list.append(rets)
                    mv = h.get("quantity", 0) * h.get("avg_price", 0)
                    weights.append(mv / total_value if total_value > 0 else 1 / len(holdings))

            if not returns_list:
                return 50.0

            min_len = min(len(r) for r in returns_list)
            returns_arr = np.array([r[-min_len:] for r in returns_list])
            w = np.array(weights[:len(returns_arr)])
            w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)

            portfolio_returns = (returns_arr * w[:, np.newaxis]).sum(axis=0)
            mean_ret = np.mean(portfolio_returns) * 252
            std_ret = np.std(portfolio_returns) * np.sqrt(252)

            sharpe = mean_ret / std_ret if std_ret > 0 else 0
            score = 50 + sharpe * 20
            return float(np.clip(score, 0, 100))
        except Exception as e:
            logger.error(f"Failed to calc risk adjusted: {e}")
            return 50.0

    def _calc_momentum(self, holdings: List[Dict[str, Any]]) -> float:
        try:
            from prediction.features import fetch_ohlcv

            momentum_scores = []
            for h in holdings:
                df = fetch_ohlcv(h["ticker"], days=90)
                if df is not None and len(df) > 20:
                    close = df["close"]
                    ret_20d = float((close.iloc[-1] / close.iloc[-20] - 1) * 100) if len(close) >= 20 else 0
                    ret_5d = float((close.iloc[-1] / close.iloc[-5] - 1) * 100) if len(close) >= 5 else 0

                    sma_20 = float(close.rolling(20).mean().iloc[-1])
                    above_sma = 1 if close.iloc[-1] > sma_20 else 0

                    ticker_momentum = 50 + ret_20d * 2 + ret_5d * 3 + above_sma * 10
                    momentum_scores.append(float(np.clip(ticker_momentum, 0, 100)))

            if not momentum_scores:
                return 50.0

            return float(np.clip(np.mean(momentum_scores), 0, 100))
        except Exception as e:
            logger.error(f"Failed to calc momentum: {e}")
            return 50.0

    def _score_to_grade(self, score: float) -> str:
        if score >= 80:
            return "A"
        elif score >= 65:
            return "B"
        elif score >= 50:
            return "C"
        elif score >= 35:
            return "D"
        else:
            return "F"


_tracker_instance = None
_adaptive_instance = None
_scorer_instance = None


def get_tracker() -> PredictionTracker:
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = PredictionTracker()
    return _tracker_instance


def get_adaptive_ensemble() -> AdaptiveEnsemble:
    global _adaptive_instance
    if _adaptive_instance is None:
        _adaptive_instance = AdaptiveEnsemble()
    return _adaptive_instance


def get_portfolio_scorer() -> SmartPortfolioScorer:
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = SmartPortfolioScorer()
    return _scorer_instance
