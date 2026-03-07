import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
import numpy as np

from prediction.features import fetch_ohlcv, compute_features

logger = logging.getLogger(__name__)

_watchlist_cache: Dict[str, tuple] = {}
CACHE_TTL = 180


def _compute_opportunity_score(symbol: str) -> Optional[Dict[str, Any]]:
    df = fetch_ohlcv(symbol, days=180)
    if df is None or len(df) < 50:
        return None

    df = compute_features(df)
    latest = df.iloc[-1]
    close = float(latest["close"])
    prev_close = float(df["close"].iloc[-2]) if len(df) > 1 else close

    factors = {}

    rsi = latest.get("rsi")
    rsi_score = 50.0
    if rsi is not None and not np.isnan(rsi):
        if rsi < 30:
            rsi_score = 85.0
        elif rsi < 40:
            rsi_score = 70.0
        elif rsi > 70:
            rsi_score = 20.0
        elif rsi > 60:
            rsi_score = 35.0
        else:
            rsi_score = 50.0 + (50.0 - rsi)
        factors["rsi"] = {"value": round(float(rsi), 1), "score": round(rsi_score, 1)}

    ma_score = 50.0
    ma_points = 0
    ma_checks = 0
    for period in [20, 50, 200]:
        col = f"sma_{period}"
        if col in df.columns:
            sma_val = latest.get(col)
            if sma_val is not None and not np.isnan(sma_val):
                ma_checks += 1
                if close > sma_val:
                    ma_points += 1
    if ma_checks > 0:
        alignment_ratio = ma_points / ma_checks
        ma_score = alignment_ratio * 100
    factors["ma_alignment"] = {"aligned": ma_points, "total": ma_checks, "score": round(ma_score, 1)}

    vol_score = 50.0
    vol_20 = df["volume"].tail(20).mean()
    vol_5 = df["volume"].tail(5).mean()
    if vol_20 > 0:
        vol_ratio = vol_5 / vol_20
        if vol_ratio > 1.5:
            vol_score = 80.0
        elif vol_ratio > 1.2:
            vol_score = 65.0
        elif vol_ratio > 0.8:
            vol_score = 50.0
        else:
            vol_score = 30.0
        factors["volume_trend"] = {"ratio": round(float(vol_ratio), 2), "score": round(vol_score, 1)}

    momentum_score = 50.0
    ret_5d = (close / float(df["close"].iloc[-5]) - 1) * 100 if len(df) > 5 else 0
    ret_20d = (close / float(df["close"].iloc[-20]) - 1) * 100 if len(df) > 20 else 0
    momentum_score = float(np.clip(50 + ret_5d * 3 + ret_20d * 1, 0, 100))
    factors["momentum"] = {
        "ret_5d": round(ret_5d, 2),
        "ret_20d": round(ret_20d, 2),
        "score": round(momentum_score, 1),
    }

    tech_signal_score = 50.0
    tech_signals = 0
    bullish_signals = 0
    bearish_signals = 0

    macd_hist = latest.get("macd_hist")
    if macd_hist is not None and not np.isnan(macd_hist):
        tech_signals += 1
        if macd_hist > 0:
            bullish_signals += 1
        else:
            bearish_signals += 1

    stoch_k = latest.get("stoch_k")
    if stoch_k is not None and not np.isnan(stoch_k):
        tech_signals += 1
        if stoch_k < 20:
            bullish_signals += 1
        elif stoch_k > 80:
            bearish_signals += 1

    adx = latest.get("adx")
    if adx is not None and not np.isnan(adx):
        tech_signals += 1
        if adx > 25:
            bullish_signals += 1

    if tech_signals > 0:
        tech_signal_score = float(np.clip(50 + (bullish_signals - bearish_signals) * 20, 0, 100))
    factors["technical_signals"] = {
        "bullish": bullish_signals,
        "bearish": bearish_signals,
        "score": round(tech_signal_score, 1),
    }

    composite = (
        tech_signal_score * 0.25 +
        rsi_score * 0.20 +
        ma_score * 0.20 +
        momentum_score * 0.20 +
        vol_score * 0.15
    )
    composite = float(np.clip(composite, 0, 100))

    setup_forming = False
    setup_type = None
    bb_pctb = latest.get("bb_pctb")
    if bb_pctb is not None and not np.isnan(bb_pctb):
        if bb_pctb < 0.05 and rsi is not None and not np.isnan(rsi) and rsi < 35:
            setup_forming = True
            setup_type = "Bullish reversal forming"
        elif bb_pctb > 0.95 and rsi is not None and not np.isnan(rsi) and rsi > 65:
            setup_forming = True
            setup_type = "Bearish reversal forming"

    if not setup_forming and adx is not None and not np.isnan(adx):
        if adx < 20 and vol_score > 60:
            setup_forming = True
            setup_type = "Breakout potential — low ADX + rising volume"

    if not setup_forming:
        sma_20 = latest.get("sma_20")
        sma_50 = latest.get("sma_50")
        if (sma_20 is not None and sma_50 is not None and
                not np.isnan(sma_20) and not np.isnan(sma_50)):
            gap_pct = abs(sma_20 - sma_50) / sma_50 * 100
            if gap_pct < 1.0:
                setup_forming = True
                setup_type = "MA convergence — potential crossover"

    change_pct = (close - prev_close) / prev_close * 100
    ann_vol = float(np.std(df["close"].pct_change().dropna().values[-20:]) * np.sqrt(252)) if len(df) > 20 else 0.25

    return {
        "symbol": symbol,
        "price": round(close, 2),
        "change_pct": round(change_pct, 2),
        "opportunity_score": round(composite, 1),
        "factors": factors,
        "setup_forming": setup_forming,
        "setup_type": setup_type,
        "volatility_ann": round(ann_vol * 100, 1),
        "technicals": {
            "rsi": round(float(rsi), 1) if rsi is not None and not np.isnan(rsi) else None,
            "macd_hist": round(float(macd_hist), 4) if macd_hist is not None and not np.isnan(macd_hist) else None,
            "adx": round(float(adx), 1) if adx is not None and not np.isnan(adx) else None,
            "bb_pctb": round(float(bb_pctb), 3) if bb_pctb is not None and not np.isnan(bb_pctb) else None,
        },
    }


async def _generate_ai_summaries(top_items: List[Dict[str, Any]]) -> Dict[str, str]:
    if not top_items:
        return {}

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    xai_key = os.environ.get("XAI_API_KEY", "")

    if not openai_key and not xai_key:
        return {}

    summaries = {}
    for item in top_items[:3]:
        symbol = item["symbol"]
        factors = item.get("factors", {})
        prompt = (
            f"Give a 1-line trading summary for {symbol}. "
            f"Price: ${item['price']}, Score: {item['opportunity_score']}/100, "
            f"RSI: {item['technicals'].get('rsi')}, "
            f"5D return: {factors.get('momentum', {}).get('ret_5d', 0):.1f}%, "
            f"20D return: {factors.get('momentum', {}).get('ret_20d', 0):.1f}%, "
            f"MA alignment: {factors.get('ma_alignment', {}).get('aligned', 0)}/{factors.get('ma_alignment', {}).get('total', 0)}, "
            f"Volume ratio: {factors.get('volume_trend', {}).get('ratio', 1.0)}, "
            f"Setup: {'Yes - ' + (item.get('setup_type') or '') if item.get('setup_forming') else 'No'}. "
            f"Respond with ONLY the 1-line summary, no JSON, no prefix."
        )
        system = "You are a concise quantitative trading analyst. Provide a single actionable sentence."

        try:
            if openai_key:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(
                        "https://api.openai.com/v1/responses",
                        headers={"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"},
                        json={"model": "gpt-5.2-pro", "instructions": system, "input": prompt},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    content = ""
                    for out_item in data.get("output", []):
                        if out_item.get("type") == "message":
                            for c in out_item.get("content", []):
                                if c.get("type") == "output_text":
                                    content = c.get("text", "")
                    summaries[symbol] = content.strip()
            elif xai_key:
                from langchain_openai import ChatOpenAI
                from langchain_core.messages import HumanMessage, SystemMessage
                llm = ChatOpenAI(
                    model="grok-4-fast-reasoning", temperature=0.1,
                    api_key=xai_key, base_url="https://api.x.ai/v1",
                )
                response = await llm.ainvoke([
                    SystemMessage(content=system),
                    HumanMessage(content=prompt),
                ])
                summaries[symbol] = response.content.strip()
        except Exception as e:
            logger.warning(f"AI summary failed for {symbol}: {e}")

    return summaries


async def get_smart_watchlist(tickers: List[str]) -> Dict[str, Any]:
    cache_key = ",".join(sorted(tickers))
    if cache_key in _watchlist_cache:
        cached_time, cached_result = _watchlist_cache[cache_key]
        if time.time() - cached_time < CACHE_TTL:
            return cached_result

    scored_items = []
    for ticker in tickers:
        try:
            result = _compute_opportunity_score(ticker)
            if result is not None:
                scored_items.append(result)
        except Exception as e:
            logger.error(f"Smart watchlist scoring failed for {ticker}: {e}")

    scored_items.sort(key=lambda x: x["opportunity_score"], reverse=True)

    for rank, item in enumerate(scored_items, 1):
        item["rank"] = rank

    ai_summaries = {}
    try:
        ai_summaries = await _generate_ai_summaries(scored_items[:3])
    except Exception as e:
        logger.warning(f"AI summaries generation failed: {e}")

    for item in scored_items:
        item["ai_summary"] = ai_summaries.get(item["symbol"])

    setup_count = sum(1 for item in scored_items if item.get("setup_forming"))
    scores = [item["opportunity_score"] for item in scored_items]
    avg_score = round(float(np.mean(scores)), 1) if scores else 0

    result = {
        "watchlist": scored_items,
        "total": len(scored_items),
        "avg_score": avg_score,
        "setups_forming": setup_count,
        "top_opportunity": scored_items[0]["symbol"] if scored_items else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    _watchlist_cache[cache_key] = (time.time(), result)
    return result
