import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

SIGNAL_HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "signal_history.json")

SIGNAL_TYPES = [
    "ml_prediction",
    "exponenta_score",
    "smart_money",
    "trade_thesis",
    "sentiment",
]

EVALUATION_WINDOWS = {
    "5d": 5,
    "20d": 20,
}


def _load_history() -> List[Dict]:
    if os.path.exists(SIGNAL_HISTORY_FILE):
        try:
            with open(SIGNAL_HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_history(records: List[Dict]):
    try:
        with open(SIGNAL_HISTORY_FILE, "w") as f:
            json.dump(records[-2000:], f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save signal history: {e}")


def record_signal(
    signal_type: str,
    ticker: str,
    direction: str,
    score: float,
    confidence: float,
    price_at_signal: float,
    details: Optional[Dict] = None,
) -> Dict[str, Any]:
    if signal_type not in SIGNAL_TYPES:
        return {"error": f"Unknown signal type: {signal_type}"}

    record = {
        "id": f"{signal_type}_{ticker}_{int(time.time())}",
        "signal_type": signal_type,
        "ticker": ticker.upper(),
        "direction": direction,
        "score": round(score, 2),
        "confidence": round(confidence, 2),
        "price_at_signal": round(price_at_signal, 4),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ts": time.time(),
        "outcome_5d": None,
        "outcome_20d": None,
        "actual_return_5d": None,
        "actual_return_20d": None,
        "evaluated": False,
        "details": details or {},
    }

    history = _load_history()
    history.append(record)
    _save_history(history)

    return {"success": True, "signal_id": record["id"]}


def evaluate_signals() -> Dict[str, Any]:
    from prediction.features import fetch_ohlcv

    history = _load_history()
    if not history:
        return {"evaluated": 0}

    now = time.time()
    evaluated_count = 0

    for record in history:
        if record.get("evaluated"):
            continue

        age_days = (now - record.get("ts", now)) / 86400

        if age_days < 5:
            continue

        ticker = record["ticker"]
        try:
            df = fetch_ohlcv(ticker, days=60)
            if df is None or len(df) < 5:
                continue

            signal_date = record.get("timestamp", "")[:10]
            df["date_str"] = df["date"].astype(str).str[:10]
            mask = df["date_str"] >= signal_date
            future = df[mask]

            if len(future) < 5:
                continue

            price_at = record["price_at_signal"]
            direction = record["direction"]

            if len(future) >= 5:
                price_5d = float(future.iloc[4]["close"])
                ret_5d = (price_5d - price_at) / price_at * 100
                record["actual_return_5d"] = round(ret_5d, 4)

                if direction in ("bullish", "positive", "buy", "strong_buy", "STRONG_BUY", "BUY"):
                    record["outcome_5d"] = "correct" if ret_5d > 0 else "incorrect"
                elif direction in ("bearish", "negative", "sell", "strong_sell", "STRONG_SELL", "SELL"):
                    record["outcome_5d"] = "correct" if ret_5d < 0 else "incorrect"
                else:
                    record["outcome_5d"] = "correct" if abs(ret_5d) < 2 else "incorrect"

            if age_days >= 20 and len(future) >= 20:
                price_20d = float(future.iloc[19]["close"])
                ret_20d = (price_20d - price_at) / price_at * 100
                record["actual_return_20d"] = round(ret_20d, 4)

                if direction in ("bullish", "positive", "buy", "strong_buy", "STRONG_BUY", "BUY"):
                    record["outcome_20d"] = "correct" if ret_20d > 0 else "incorrect"
                elif direction in ("bearish", "negative", "sell", "strong_sell", "STRONG_SELL", "SELL"):
                    record["outcome_20d"] = "correct" if ret_20d < 0 else "incorrect"
                else:
                    record["outcome_20d"] = "correct" if abs(ret_20d) < 3 else "incorrect"

            if record["outcome_5d"] is not None:
                record["evaluated"] = True
                evaluated_count += 1

        except Exception as e:
            logger.debug(f"Eval failed for {ticker}: {e}")

    _save_history(history)
    return {"evaluated": evaluated_count}


def get_signal_performance() -> Dict[str, Any]:
    history = _load_history()

    try:
        evaluate_signals()
        history = _load_history()
    except Exception:
        pass

    type_stats: Dict[str, Dict] = {}
    for st in SIGNAL_TYPES:
        signals = [r for r in history if r["signal_type"] == st]
        evaluated = [r for r in signals if r.get("evaluated")]

        correct_5d = sum(1 for r in evaluated if r.get("outcome_5d") == "correct")
        incorrect_5d = sum(1 for r in evaluated if r.get("outcome_5d") == "incorrect")
        total_5d = correct_5d + incorrect_5d

        correct_20d = sum(1 for r in evaluated if r.get("outcome_20d") == "correct")
        incorrect_20d = sum(1 for r in evaluated if r.get("outcome_20d") == "incorrect")
        total_20d = correct_20d + incorrect_20d

        returns_5d = [r["actual_return_5d"] for r in evaluated if r.get("actual_return_5d") is not None]
        returns_20d = [r["actual_return_20d"] for r in evaluated if r.get("actual_return_20d") is not None]

        accuracy_5d = round(correct_5d / total_5d * 100, 1) if total_5d > 0 else None
        accuracy_20d = round(correct_20d / total_20d * 100, 1) if total_20d > 0 else None

        avg_return_5d = round(float(np.mean(returns_5d)), 2) if returns_5d else None
        avg_return_20d = round(float(np.mean(returns_20d)), 2) if returns_20d else None

        precision = None
        recall = None
        if total_5d > 0:
            bullish_signals = [r for r in evaluated if r["direction"] in ("bullish", "positive", "buy", "strong_buy", "STRONG_BUY", "BUY")]
            true_positives = sum(1 for r in bullish_signals if r.get("outcome_5d") == "correct")
            false_positives = sum(1 for r in bullish_signals if r.get("outcome_5d") == "incorrect")
            actual_positives = sum(1 for r in evaluated if r.get("actual_return_5d") is not None and r["actual_return_5d"] > 0)

            if true_positives + false_positives > 0:
                precision = round(true_positives / (true_positives + false_positives) * 100, 1)
            if actual_positives > 0:
                recall = round(true_positives / actual_positives * 100, 1)

        type_stats[st] = {
            "signal_type": st,
            "display_name": st.replace("_", " ").title(),
            "total_signals": len(signals),
            "evaluated_signals": len(evaluated),
            "pending_signals": len(signals) - len(evaluated),
            "accuracy_5d": accuracy_5d,
            "accuracy_20d": accuracy_20d,
            "correct_5d": correct_5d,
            "incorrect_5d": incorrect_5d,
            "correct_20d": correct_20d,
            "incorrect_20d": incorrect_20d,
            "avg_return_5d": avg_return_5d,
            "avg_return_20d": avg_return_20d,
            "precision": precision,
            "recall": recall,
        }

    leaderboard = sorted(
        [s for s in type_stats.values() if s["accuracy_5d"] is not None],
        key=lambda x: x["accuracy_5d"],
        reverse=True,
    )

    if not leaderboard:
        leaderboard = sorted(type_stats.values(), key=lambda x: x["total_signals"], reverse=True)

    recent = sorted(history, key=lambda x: x.get("ts", 0), reverse=True)[:30]
    recent_display = []
    for r in recent:
        recent_display.append({
            "id": r["id"],
            "signal_type": r["signal_type"],
            "ticker": r["ticker"],
            "direction": r["direction"],
            "score": r["score"],
            "confidence": r["confidence"],
            "price_at_signal": r["price_at_signal"],
            "timestamp": r["timestamp"],
            "outcome_5d": r.get("outcome_5d", "pending"),
            "outcome_20d": r.get("outcome_20d", "pending"),
            "actual_return_5d": r.get("actual_return_5d"),
            "actual_return_20d": r.get("actual_return_20d"),
        })

    all_evaluated = [r for r in history if r.get("evaluated")]
    all_correct_5d = sum(1 for r in all_evaluated if r.get("outcome_5d") == "correct")
    all_total_5d = sum(1 for r in all_evaluated if r.get("outcome_5d") in ("correct", "incorrect"))
    overall_accuracy = round(all_correct_5d / all_total_5d * 100, 1) if all_total_5d > 0 else None

    best_type = leaderboard[0]["signal_type"] if leaderboard and leaderboard[0].get("accuracy_5d") is not None else None

    accuracy_trend = _compute_accuracy_trend(history)

    return {
        "leaderboard": leaderboard,
        "signal_stats": type_stats,
        "recent_signals": recent_display,
        "overall_accuracy": overall_accuracy,
        "total_signals": len(history),
        "total_evaluated": len(all_evaluated),
        "best_signal_type": best_type,
        "accuracy_trend": accuracy_trend,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _compute_accuracy_trend(history: List[Dict]) -> Dict[str, List[Dict]]:
    trends: Dict[str, List[Dict]] = {}

    for st in SIGNAL_TYPES:
        signals = sorted(
            [r for r in history if r["signal_type"] == st and r.get("evaluated")],
            key=lambda x: x.get("ts", 0),
        )

        if len(signals) < 3:
            trends[st] = []
            continue

        window = max(5, len(signals) // 5)
        points = []
        for i in range(window, len(signals) + 1):
            batch = signals[max(0, i - window):i]
            correct = sum(1 for r in batch if r.get("outcome_5d") == "correct")
            total = sum(1 for r in batch if r.get("outcome_5d") in ("correct", "incorrect"))
            if total > 0:
                acc = round(correct / total * 100, 1)
                points.append({
                    "index": i,
                    "accuracy": acc,
                    "sample_size": total,
                    "date": batch[-1].get("timestamp", "")[:10],
                })

        trends[st] = points

    return trends


def auto_record_ml_prediction(ticker: str, prediction: Dict):
    try:
        direction = prediction.get("direction", "neutral")
        confidence = prediction.get("confidence", 50)
        price = prediction.get("price", 0)
        score = prediction.get("predicted_trend_pct", 0)

        if price <= 0:
            return

        record_signal(
            signal_type="ml_prediction",
            ticker=ticker,
            direction=direction,
            score=abs(score),
            confidence=confidence,
            price_at_signal=price,
            details={
                "recommendation": prediction.get("recommendation"),
                "model_accuracy": prediction.get("model_accuracy"),
                "predicted_trend_pct": prediction.get("predicted_trend_pct"),
            },
        )
    except Exception as e:
        logger.debug(f"Auto-record ML signal failed: {e}")


def auto_record_composite_score(ticker: str, result: Dict):
    try:
        score = result.get("composite_score", 50)
        direction = result.get("direction", "neutral")
        confidence = result.get("confidence", 0.5)

        from prediction.features import fetch_ohlcv
        df = fetch_ohlcv(ticker, days=5)
        if df is None or len(df) < 1:
            return

        price = float(df["close"].iloc[-1])

        record_signal(
            signal_type="exponenta_score",
            ticker=ticker,
            direction=direction,
            score=score,
            confidence=confidence * 100 if confidence <= 1 else confidence,
            price_at_signal=price,
            details={
                "signal_label": result.get("signal_label"),
                "composite_score": score,
            },
        )
    except Exception as e:
        logger.debug(f"Auto-record composite signal failed: {e}")


def auto_record_sentiment(ticker: str, result: Dict):
    try:
        score = result.get("overall_score", 50)
        sentiment = result.get("sentiment", "neutral")
        confidence = result.get("confidence", 50)
        price = result.get("market_data", {}).get("price", 0)

        if price <= 0:
            return

        record_signal(
            signal_type="sentiment",
            ticker=ticker,
            direction=sentiment,
            score=score,
            confidence=confidence,
            price_at_signal=price,
            details={
                "source": result.get("source"),
            },
        )
    except Exception as e:
        logger.debug(f"Auto-record sentiment signal failed: {e}")


def auto_record_thesis(ticker: str, result: Dict):
    try:
        thesis = result.get("thesis", {})
        conviction = thesis.get("conviction_score", 50)
        direction_label = thesis.get("direction", "neutral")
        price = thesis.get("current_price", 0)

        if price <= 0:
            return

        record_signal(
            signal_type="trade_thesis",
            ticker=ticker,
            direction=direction_label,
            score=conviction,
            confidence=conviction,
            price_at_signal=price,
            details={
                "time_horizon": thesis.get("time_horizon"),
                "thesis_summary": thesis.get("summary", "")[:200],
            },
        )
    except Exception as e:
        logger.debug(f"Auto-record thesis signal failed: {e}")


def auto_record_smart_money(ticker: str, result: Dict):
    try:
        score = result.get("smart_money_score", 50)
        direction = "bullish" if score > 60 else "bearish" if score < 40 else "neutral"

        from prediction.features import fetch_ohlcv
        df = fetch_ohlcv(ticker, days=5)
        if df is None or len(df) < 1:
            return

        price = float(df["close"].iloc[-1])

        record_signal(
            signal_type="smart_money",
            ticker=ticker,
            direction=direction,
            score=score,
            confidence=min(100, abs(score - 50) * 2 + 50),
            price_at_signal=price,
            details={
                "insider_clusters": len(result.get("insider_clusters", [])),
            },
        )
    except Exception as e:
        logger.debug(f"Auto-record smart money signal failed: {e}")
