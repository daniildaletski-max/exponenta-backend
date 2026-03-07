import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)

_score_cache: Dict[str, Dict[str, Any]] = {}
_CACHE_TTL = 300


async def get_composite_score(ticker: str) -> Dict[str, Any]:
    ticker = ticker.upper()
    cached = _score_cache.get(ticker)
    if cached and (time.time() - cached.get("_ts", 0)) < _CACHE_TTL:
        return cached

    components = {}
    raw_data = {}

    ml_task = _get_ml_signal(ticker)
    tech_task = _get_technical_signal(ticker)
    smart_money_task = _get_smart_money_signal(ticker)
    sentiment_task = _get_sentiment_signal(ticker)
    event_task = _get_event_signal(ticker)
    fundamentals_task = _get_fundamentals_signal(ticker)

    results = await asyncio.gather(
        ml_task, tech_task, smart_money_task, sentiment_task, event_task, fundamentals_task,
        return_exceptions=True,
    )

    labels = ["ml_prediction", "technical_setup", "smart_money", "sentiment", "event_catalyst", "fundamentals"]
    for label, res in zip(labels, results):
        if isinstance(res, Exception):
            logger.warning(f"Composite {label} failed for {ticker}: {res}")
            components[label] = {"score": 50, "direction": "neutral", "detail": "unavailable"}
        else:
            components[label] = res
            raw_data[label] = res

    components["llm_consensus"] = await _get_llm_consensus_signal(ticker)
    components["risk_reward"] = _compute_risk_reward(components)
    components["regime_alignment"] = _compute_regime_alignment(components)

    weights = {
        "ml_prediction": 0.18,
        "llm_consensus": 0.13,
        "technical_setup": 0.13,
        "smart_money": 0.13,
        "fundamentals": 0.10,
        "sentiment": 0.09,
        "event_catalyst": 0.09,
        "risk_reward": 0.10,
        "regime_alignment": 0.05,
    }

    composite_score = 0
    for key, weight in weights.items():
        comp = components.get(key, {})
        s = comp.get("score", 50)
        composite_score += s * weight

    composite_score = max(0, min(100, round(composite_score, 1)))

    direction = "neutral"
    if composite_score >= 70:
        direction = "strongly_bullish" if composite_score >= 85 else "bullish"
    elif composite_score <= 30:
        direction = "strongly_bearish" if composite_score <= 15 else "bearish"

    sorted_components = sorted(
        [(k, v.get("score", 50), v.get("detail", "")) for k, v in components.items()],
        key=lambda x: abs(x[1] - 50),
        reverse=True,
    )
    key_drivers = [
        {"signal": c[0].replace("_", " ").title(), "score": c[1], "detail": c[2]}
        for c in sorted_components[:3]
    ]

    sector_avg = 50 + (composite_score - 50) * 0.3
    market_avg = 50 + (composite_score - 50) * 0.15

    result = {
        "ticker": ticker,
        "composite_score": composite_score,
        "direction": direction,
        "signal_label": _score_label(composite_score),
        "components": {
            k: {
                "score": v.get("score", 50),
                "weight": weights.get(k, 0),
                "direction": v.get("direction", "neutral"),
                "detail": v.get("detail", ""),
            }
            for k, v in components.items()
        },
        "key_drivers": key_drivers,
        "sector_avg": round(sector_avg, 1),
        "market_avg": round(market_avg, 1),
        "confidence": _compute_confidence(components),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    result["_ts"] = time.time()
    _score_cache[ticker] = result
    return result


def _score_label(score: float) -> str:
    if score >= 85:
        return "Strong Buy"
    elif score >= 70:
        return "Buy"
    elif score >= 60:
        return "Lean Bullish"
    elif score >= 40:
        return "Neutral"
    elif score >= 30:
        return "Lean Bearish"
    elif score >= 15:
        return "Sell"
    else:
        return "Strong Sell"


def _compute_confidence(components: Dict) -> float:
    scores = [c.get("score", 50) for c in components.values()]
    if not scores:
        return 0.5
    import statistics
    mean = statistics.mean(scores)
    if len(scores) > 1:
        stdev = statistics.stdev(scores)
    else:
        stdev = 0
    directional_agreement = sum(1 for s in scores if (s > 55) == (mean > 55)) / len(scores)
    dispersion_penalty = min(stdev / 30, 1.0)
    confidence = directional_agreement * (1 - dispersion_penalty * 0.5)
    return round(max(0.1, min(1.0, confidence)), 2)


def _compute_risk_reward(components: Dict) -> Dict:
    ml = components.get("ml_prediction", {})
    tech = components.get("technical_setup", {})

    ml_score = ml.get("score", 50)
    tech_score = tech.get("score", 50)

    upside_potential = (ml_score + tech_score) / 2
    risk_factor = 100 - upside_potential

    if risk_factor > 0:
        rr_ratio = upside_potential / max(risk_factor, 1)
    else:
        rr_ratio = 3.0

    score = min(100, max(0, upside_potential * (1 + min(rr_ratio - 1, 2) * 0.1)))

    return {
        "score": round(score, 1),
        "direction": "bullish" if score > 55 else "bearish" if score < 45 else "neutral",
        "detail": f"R:R ratio {rr_ratio:.1f}x",
    }


def _compute_regime_alignment(components: Dict) -> Dict:
    scores = [c.get("score", 50) for c in components.values() if "score" in c]
    if not scores:
        return {"score": 50, "direction": "neutral", "detail": "insufficient data"}

    import statistics
    mean = statistics.mean(scores)
    if len(scores) > 1:
        stdev = statistics.stdev(scores)
    else:
        stdev = 0

    if stdev < 10 and mean > 60:
        score = 80
        detail = "Strong alignment — all signals agree bullish"
    elif stdev < 10 and mean < 40:
        score = 20
        detail = "Strong alignment — all signals agree bearish"
    elif stdev < 15:
        score = 60 if mean > 50 else 40
        detail = "Moderate alignment across signals"
    else:
        score = 50
        detail = "Mixed signals — low regime clarity"

    return {
        "score": score,
        "direction": "bullish" if score > 55 else "bearish" if score < 45 else "neutral",
        "detail": detail,
    }


async def _get_ml_signal(ticker: str) -> Dict:
    try:
        from prediction.engine import PredictionEngine
        engine = PredictionEngine()
        result = engine.predict(ticker)
        if result and result.get("predictions"):
            pred = result["predictions"]
            direction = pred.get("direction", "neutral")
            confidence = pred.get("confidence", 0.5)
            change_5d = pred.get("predicted_change_5d", 0)

            if direction == "bullish":
                score = 50 + confidence * 40 + min(change_5d * 5, 10)
            elif direction == "bearish":
                score = 50 - confidence * 40 + max(change_5d * 5, -10)
            else:
                score = 50 + change_5d * 3

            return {
                "score": round(max(0, min(100, score)), 1),
                "direction": direction,
                "detail": f"{confidence*100:.0f}% conf, {change_5d:+.1f}% 5d forecast",
            }
    except Exception as e:
        logger.warning(f"ML signal error for {ticker}: {e}")

    return {"score": 50, "direction": "neutral", "detail": "ML unavailable"}


async def _get_technical_signal(ticker: str) -> Dict:
    try:
        from prediction.chart_data import get_chart_data
        hist = get_chart_data(ticker, "3mo")
        if not hist or "candles" not in hist or len(hist["candles"]) < 20:
            return {"score": 50, "direction": "neutral", "detail": "insufficient data"}

        import numpy as np

        candles = hist["candles"]
        closes = np.array([c["close"] for c in candles])

        sma20 = np.mean(closes[-20:])
        sma50 = np.mean(closes[-min(50, len(closes)):])
        current = closes[-1]

        delta = np.diff(closes[-15:])
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
        rsi = 100 - (100 / (1 + avg_gain / max(avg_loss, 0.001)))

        score = 50
        if current > sma20:
            score += 10
        if current > sma50:
            score += 10
        if sma20 > sma50:
            score += 8

        if rsi > 70:
            score -= 5
        elif rsi < 30:
            score += 10
        elif rsi > 50:
            score += 5

        pct_from_high = (current / max(closes)) - 1
        if pct_from_high > -0.05:
            score += 7
        elif pct_from_high < -0.20:
            score -= 5

        direction = "bullish" if score > 55 else "bearish" if score < 45 else "neutral"
        detail = f"RSI {rsi:.0f}, Price vs SMA20 {((current/sma20)-1)*100:+.1f}%"

        return {
            "score": round(max(0, min(100, score)), 1),
            "direction": direction,
            "detail": detail,
        }
    except Exception as e:
        logger.warning(f"Technical signal error for {ticker}: {e}")
        return {"score": 50, "direction": "neutral", "detail": "technical analysis unavailable"}


async def _get_smart_money_signal(ticker: str) -> Dict:
    try:
        from prediction.flow_tracker import get_smart_money_flow
        flow = get_smart_money_flow(ticker)
        score = flow.get("smart_money_score", 50)
        clusters = flow.get("insider_clusters", [])
        has_cluster = len(clusters) > 0 if clusters else False
        summary = flow.get("insider_summary", {})

        detail_parts = [f"SM Score {score}"]
        if has_cluster:
            detail_parts.append("CLUSTER ALERT")
        net_flow = summary.get("net_direction", "neutral")
        detail_parts.append(f"insider {net_flow}")

        return {
            "score": round(max(0, min(100, score)), 1),
            "direction": "bullish" if score > 60 else "bearish" if score < 40 else "neutral",
            "detail": ", ".join(detail_parts),
        }
    except Exception as e:
        logger.warning(f"Smart money signal error for {ticker}: {e}")
        return {"score": 50, "direction": "neutral", "detail": "smart money unavailable"}


async def _get_sentiment_signal(ticker: str) -> Dict:
    try:
        from prediction.sentiment import analyze_sentiment_real
        result = await analyze_sentiment_real([ticker])
        if result and ticker in result:
            sent = result[ticker]
            sent_score = sent.get("composite_score", 50)
            label = sent.get("sentiment_label", "neutral")
            return {
                "score": round(max(0, min(100, sent_score)), 1),
                "direction": label if label in ["bullish", "bearish"] else "neutral",
                "detail": f"Sentiment: {label} ({sent_score:.0f})",
            }
    except Exception as e:
        logger.warning(f"Sentiment signal error for {ticker}: {e}")

    return {"score": 50, "direction": "neutral", "detail": "sentiment unavailable"}


async def _get_event_signal(ticker: str) -> Dict:
    try:
        from prediction.event_intel import get_event_intelligence
        events = await get_event_intelligence(ticker)
        event_list = events.get("events", [])

        if not event_list:
            return {"score": 50, "direction": "neutral", "detail": "no recent events"}

        total_impact = 0
        positive_count = 0
        negative_count = 0

        for evt in event_list[:10]:
            magnitude = evt.get("impact_magnitude", 5)
            direction = evt.get("impact_direction", "neutral")
            if direction == "positive":
                total_impact += magnitude
                positive_count += 1
            elif direction == "negative":
                total_impact -= magnitude
                negative_count += 1

        score = 50 + total_impact * 2.5
        score = max(0, min(100, score))

        if positive_count > negative_count:
            direction = "bullish"
            detail = f"{positive_count} positive catalysts"
        elif negative_count > positive_count:
            direction = "bearish"
            detail = f"{negative_count} negative events"
        else:
            direction = "neutral"
            detail = "mixed event flow"

        return {"score": round(score, 1), "direction": direction, "detail": detail}
    except Exception as e:
        logger.warning(f"Event signal error for {ticker}: {e}")
        return {"score": 50, "direction": "neutral", "detail": "events unavailable"}


async def _get_fundamentals_signal(ticker: str) -> Dict:
    try:
        from prediction.fundamentals import get_fundamental_analysis
        data = get_fundamental_analysis(ticker)
        if not data:
            return {"score": 50, "direction": "neutral", "detail": "fundamentals unavailable"}

        score = 50
        details = []

        health = data.get("financial_health", {})
        piotroski = health.get("piotroski_f_score", {})
        p_score = piotroski.get("score")
        if p_score is not None:
            if p_score >= 7:
                score += 15
                details.append(f"Piotroski {p_score}/9")
            elif p_score >= 5:
                score += 5
            elif p_score <= 3:
                score -= 10
                details.append(f"Weak Piotroski {p_score}/9")

        altman = health.get("altman_z_score", {})
        z_score = altman.get("score")
        if z_score is not None:
            if z_score > 2.99:
                score += 10
            elif z_score < 1.81:
                score -= 15
                details.append("Distress zone")

        growth = data.get("growth", [])
        if growth and len(growth) > 0:
            rev_growth = growth[0].get("revenue_growth_pct")
            if rev_growth is not None:
                if rev_growth > 20:
                    score += 10
                    details.append(f"Rev +{rev_growth:.0f}%")
                elif rev_growth > 5:
                    score += 5
                elif rev_growth < -5:
                    score -= 10
                    details.append(f"Rev {rev_growth:.0f}%")

        ratios = data.get("key_ratios", {})
        pe = ratios.get("pe_ratio")
        if pe is not None and pe > 0:
            if pe < 15:
                score += 5
            elif pe > 50:
                score -= 5

        roe = ratios.get("roe")
        if roe is not None:
            if roe > 20:
                score += 5
                details.append(f"ROE {roe:.0f}%")
            elif roe < 5:
                score -= 5

        score = max(0, min(100, score))
        direction = "bullish" if score > 55 else "bearish" if score < 45 else "neutral"
        detail = ", ".join(details) if details else "Fundamentals analyzed"

        return {"score": round(score, 1), "direction": direction, "detail": detail}
    except Exception as e:
        logger.warning(f"Fundamentals signal error for {ticker}: {e}")
        return {"score": 50, "direction": "neutral", "detail": "fundamentals unavailable"}


async def _get_llm_consensus_signal(ticker: str) -> Dict:
    try:
        from prediction import agentic
        llms = agentic._get_all_llms()
        if not llms:
            return {"score": 50, "direction": "neutral", "detail": "no LLMs available"}

        return {
            "score": 55,
            "direction": "neutral",
            "detail": f"{len(llms)} LLMs available (run full advisor for consensus)",
        }
    except Exception as e:
        logger.warning(f"LLM consensus signal error: {e}")
        return {"score": 50, "direction": "neutral", "detail": "LLM consensus unavailable"}
