import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import httpx
import numpy as np

from prediction.features import fetch_ohlcv, compute_features

logger = logging.getLogger(__name__)

_x_sentiment_cache: Dict[str, tuple] = {}
X_CACHE_TTL = 300


def _compute_technical_sentiment(symbol: str) -> Dict[str, Any]:
    df = fetch_ohlcv(symbol, days=90)
    if df is None or len(df) < 30:
        return {"error": "Insufficient data"}

    df = compute_features(df)
    latest = df.iloc[-1]

    signals = []
    bullish_count = 0
    bearish_count = 0
    total_signals = 0

    rsi = latest.get("rsi")
    if rsi is not None and not np.isnan(rsi):
        total_signals += 1
        if rsi < 30:
            signals.append({"factor": f"RSI oversold at {rsi:.1f} — potential bounce", "type": "bullish"})
            bullish_count += 1
        elif rsi > 70:
            signals.append({"factor": f"RSI overbought at {rsi:.1f} — potential pullback", "type": "bearish"})
            bearish_count += 1
        elif rsi > 50:
            signals.append({"factor": f"RSI at {rsi:.1f} showing bullish momentum", "type": "bullish"})
            bullish_count += 1
        else:
            signals.append({"factor": f"RSI at {rsi:.1f} showing bearish momentum", "type": "bearish"})
            bearish_count += 1

    macd_hist = latest.get("macd_hist")
    if macd_hist is not None and not np.isnan(macd_hist):
        total_signals += 1
        prev_macd = df["macd_hist"].iloc[-2] if len(df) > 1 and "macd_hist" in df.columns else 0
        if not np.isnan(prev_macd):
            if macd_hist > 0 and prev_macd <= 0:
                signals.append({"factor": "MACD bullish crossover — buy signal", "type": "bullish"})
                bullish_count += 2
            elif macd_hist < 0 and prev_macd >= 0:
                signals.append({"factor": "MACD bearish crossover — sell signal", "type": "bearish"})
                bearish_count += 2
            elif macd_hist > 0:
                signals.append({"factor": "MACD histogram positive — uptrend intact", "type": "bullish"})
                bullish_count += 1
            else:
                signals.append({"factor": "MACD histogram negative — downtrend pressure", "type": "bearish"})
                bearish_count += 1

    adx = latest.get("adx")
    if adx is not None and not np.isnan(adx):
        total_signals += 1
        if adx > 25:
            signals.append({"factor": f"ADX at {adx:.1f} — strong trend in place", "type": "neutral"})
        else:
            signals.append({"factor": f"ADX at {adx:.1f} — weak/range-bound trend", "type": "neutral"})

    bb_pctb = latest.get("bb_pctb")
    if bb_pctb is not None and not np.isnan(bb_pctb):
        total_signals += 1
        if bb_pctb > 1.0:
            signals.append({"factor": f"Price above upper Bollinger Band — overbought", "type": "bearish"})
            bearish_count += 1
        elif bb_pctb < 0.0:
            signals.append({"factor": f"Price below lower Bollinger Band — oversold", "type": "bullish"})
            bullish_count += 1
        elif bb_pctb > 0.5:
            signals.append({"factor": f"Price in upper Bollinger range — bullish bias", "type": "bullish"})
            bullish_count += 1
        else:
            signals.append({"factor": f"Price in lower Bollinger range — bearish bias", "type": "bearish"})
            bearish_count += 1

    close = df["close"]
    vol_current = float(close.tail(5).pct_change().std())
    vol_30d = float(close.tail(30).pct_change().std())
    if vol_30d > 0:
        vol_ratio = vol_current / vol_30d
        total_signals += 1
        if vol_ratio > 1.5:
            signals.append({"factor": f"Volatility spike ({vol_ratio:.1f}x 30-day avg) — high uncertainty", "type": "bearish"})
            bearish_count += 1
        elif vol_ratio < 0.7:
            signals.append({"factor": f"Low volatility ({vol_ratio:.1f}x 30-day avg) — calm market", "type": "bullish"})
            bullish_count += 1

    if "volume" in df.columns:
        vol_avg_20 = float(df["volume"].tail(20).mean())
        vol_recent = float(df["volume"].tail(3).mean())
        if vol_avg_20 > 0:
            vol_ratio_v = vol_recent / vol_avg_20
            total_signals += 1
            if vol_ratio_v > 1.5:
                chg_5d = float((close.iloc[-1] / close.iloc[-5] - 1) * 100) if len(close) > 5 else 0
                if chg_5d > 0:
                    signals.append({"factor": f"Volume surge ({vol_ratio_v:.1f}x avg) on price rise — strong buying", "type": "bullish"})
                    bullish_count += 2
                else:
                    signals.append({"factor": f"Volume surge ({vol_ratio_v:.1f}x avg) on decline — selling pressure", "type": "bearish"})
                    bearish_count += 2
            elif vol_ratio_v < 0.6:
                signals.append({"factor": f"Below-average volume ({vol_ratio_v:.1f}x) — low conviction", "type": "neutral"})

    sma_20 = float(close.tail(20).mean())
    sma_50 = float(close.tail(50).mean()) if len(close) >= 50 else sma_20
    current_price = float(close.iloc[-1])
    total_signals += 1
    if current_price > sma_20 > sma_50:
        signals.append({"factor": "Price above SMA20 > SMA50 — bullish alignment", "type": "bullish"})
        bullish_count += 1
    elif current_price < sma_20 < sma_50:
        signals.append({"factor": "Price below SMA20 < SMA50 — bearish alignment", "type": "bearish"})
        bearish_count += 1
    elif current_price > sma_20:
        signals.append({"factor": "Price above SMA20 — short-term bullish", "type": "bullish"})
        bullish_count += 1
    else:
        signals.append({"factor": "Price below SMA20 — short-term bearish", "type": "bearish"})
        bearish_count += 1

    stoch_k = latest.get("stoch_k")
    if stoch_k is not None and not np.isnan(stoch_k):
        total_signals += 1
        if stoch_k < 20:
            signals.append({"factor": f"Stochastic oversold at {stoch_k:.1f}", "type": "bullish"})
            bullish_count += 1
        elif stoch_k > 80:
            signals.append({"factor": f"Stochastic overbought at {stoch_k:.1f}", "type": "bearish"})
            bearish_count += 1

    cci = latest.get("cci")
    if cci is not None and not np.isnan(cci):
        total_signals += 1
        if cci > 100:
            signals.append({"factor": f"CCI at {cci:.0f} — strong bullish momentum", "type": "bullish"})
            bullish_count += 1
        elif cci < -100:
            signals.append({"factor": f"CCI at {cci:.0f} — strong bearish momentum", "type": "bearish"})
            bearish_count += 1

    chg_1d = float(close.pct_change().iloc[-1] * 100)
    chg_5d = float((close.iloc[-1] / close.iloc[-5] - 1) * 100) if len(close) > 5 else 0
    chg_20d = float((close.iloc[-1] / close.iloc[-20] - 1) * 100) if len(close) > 20 else 0

    total_weight = bullish_count + bearish_count
    if total_weight == 0:
        score = 50
    else:
        score = int(round(bullish_count / total_weight * 100))
    score = max(5, min(95, score))

    momentum_bonus = 0
    if chg_5d > 3:
        momentum_bonus = 5
    elif chg_5d < -3:
        momentum_bonus = -5
    if chg_20d > 10:
        momentum_bonus += 5
    elif chg_20d < -10:
        momentum_bonus -= 5
    score = max(5, min(95, score + momentum_bonus))

    if score >= 65:
        sentiment = "bullish"
    elif score <= 35:
        sentiment = "bearish"
    else:
        sentiment = "neutral"

    total_possible = max(total_signals, 1)
    confidence = min(95, max(40, int(60 + (total_signals / 8) * 30)))

    return {
        "score": score,
        "sentiment": sentiment,
        "confidence": confidence,
        "signals": signals,
        "technicals": {
            "rsi": round(float(rsi), 1) if rsi is not None and not np.isnan(rsi) else None,
            "macd_hist": round(float(macd_hist), 4) if macd_hist is not None and not np.isnan(macd_hist) else None,
            "adx": round(float(adx), 1) if adx is not None and not np.isnan(adx) else None,
            "bb_pctb": round(float(bb_pctb), 3) if bb_pctb is not None and not np.isnan(bb_pctb) else None,
            "stoch_k": round(float(stoch_k), 1) if stoch_k is not None and not np.isnan(stoch_k) else None,
            "cci": round(float(cci), 1) if cci is not None and not np.isnan(cci) else None,
        },
        "price_action": {
            "price": round(current_price, 2),
            "change_1d": round(chg_1d, 2),
            "change_5d": round(chg_5d, 2),
            "change_20d": round(chg_20d, 2),
            "sma_20": round(sma_20, 2),
            "sma_50": round(sma_50, 2),
        },
        "bullish_signals": bullish_count,
        "bearish_signals": bearish_count,
    }


def _clean_grok_text(text: str) -> str:
    text = re.sub(r'<grok:render[^>]*>.*?</grok:render>', '', text, flags=re.DOTALL)
    text = re.sub(r'<argument[^>]*>[^<]*</argument>', '', text)
    return text.strip()


async def _x_social_sentiment(symbol: str) -> Optional[Dict[str, Any]]:
    xai_key = os.environ.get("XAI_API_KEY", "")
    if not xai_key:
        return None

    cache_key = symbol.upper()
    cached = _x_sentiment_cache.get(cache_key)
    if cached and (time.time() - cached[0]) < X_CACHE_TTL:
        return cached[1]

    ticker_name_map = {
        "AAPL": "Apple", "TSLA": "Tesla", "NVDA": "NVIDIA", "GOOGL": "Google/Alphabet",
        "MSFT": "Microsoft", "AMZN": "Amazon", "META": "Meta", "AMD": "AMD",
        "VOO": "Vanguard S&P 500 ETF", "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum",
    }
    company = ticker_name_map.get(symbol, symbol)

    try:
        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.post(
                "https://api.x.ai/v1/responses",
                headers={"Authorization": f"Bearer {xai_key}", "Content-Type": "application/json"},
                json={
                    "model": "grok-4-1-fast-non-reasoning",
                    "input": [
                        {"role": "system", "content": "You are a financial social media analyst. Search X (Twitter) for recent posts about the given stock. Analyze the social sentiment. Return valid JSON only, no markdown."},
                        {"role": "user", "content": f"Search X for the most recent posts and discussions about ${symbol} ({company} stock). Find real posts from the last few days — news, analyst opinions, earnings reactions, product announcements, retail investor sentiment. Return JSON: {{\"x_posts\": [{{\"summary\": \"brief summary of the post\", \"sentiment\": \"positive/negative/neutral\", \"topic\": \"earnings/product/market/analyst/regulatory/macro/other\", \"url\": \"post URL if available\"}}], \"x_sentiment_score\": <0-100 where 0=extremely bearish, 100=extremely bullish>, \"x_sentiment\": \"bullish/bearish/neutral\", \"key_narratives\": [\"major theme on X\"], \"notable_news\": [\"headline from X posts\"], \"post_volume\": \"high/medium/low\"}}"}
                    ],
                    "temperature": 0.1,
                    "tools": [{"type": "x_search"}],
                },
            )

        if resp.status_code != 200:
            logger.warning(f"X sentiment API error for {symbol}: {resp.status_code} {resp.text[:200]}")
            return None

        data = resp.json()
        output = data.get("output", [])

        text_content = None
        x_urls = []
        for item in output:
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") in ("text", "output_text"):
                        text_content = c["text"]
                        for ann in c.get("annotations", []):
                            if ann.get("type") == "url_citation":
                                x_urls.append(ann["url"])

        if not text_content:
            return None

        cleaned = _clean_grok_text(text_content)
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        parsed = json.loads(cleaned)

        for i, post in enumerate(parsed.get("x_posts", [])):
            if not post.get("url") and i < len(x_urls):
                post["url"] = x_urls[i]

        result = {
            "x_posts": parsed.get("x_posts", [])[:8],
            "x_sentiment_score": max(0, min(100, int(parsed.get("x_sentiment_score", 50)))),
            "x_sentiment": parsed.get("x_sentiment", "neutral"),
            "key_narratives": parsed.get("key_narratives", [])[:5],
            "notable_news": parsed.get("notable_news", [])[:5],
            "post_volume": parsed.get("post_volume", "medium"),
            "x_urls": x_urls[:10],
            "source": "x_search_grok4",
        }

        _x_sentiment_cache[cache_key] = (time.time(), result)
        return result

    except json.JSONDecodeError as e:
        logger.warning(f"X sentiment JSON parse error for {symbol}: {e}")
        return None
    except Exception as e:
        logger.warning(f"X sentiment failed for {symbol}: {e}")
        return None


async def _llm_sentiment(symbol: str, tech_data: dict) -> Dict[str, Any]:
    llms = []
    xai_key = os.environ.get("XAI_API_KEY", "")
    if xai_key:
        from langchain_openai import ChatOpenAI
        llms.append(("grok", ChatOpenAI(
            model="grok-4-fast-reasoning", temperature=0.1,
            api_key=xai_key, base_url="https://api.x.ai/v1",
        )))
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        llms.append(("gpt", {"use_responses_api": True, "api_key": openai_key}))

    if not llms:
        return {}

    price_action = tech_data.get("price_action", {})
    technicals = tech_data.get("technicals", {})
    prompt = f"""Analyze market sentiment for {symbol} based on this real technical data:

Price: ${price_action.get('price', 0)} | 1D: {price_action.get('change_1d', 0):.2f}% | 5D: {price_action.get('change_5d', 0):.2f}% | 20D: {price_action.get('change_20d', 0):.2f}%
RSI: {technicals.get('rsi')} | MACD: {technicals.get('macd_hist')} | ADX: {technicals.get('adx')} | BB%B: {technicals.get('bb_pctb')}
Technical Score: {tech_data.get('score', 50)}/100 ({tech_data.get('sentiment', 'neutral')})
Bullish signals: {tech_data.get('bullish_signals', 0)} | Bearish: {tech_data.get('bearish_signals', 0)}

Provide JSON with your independent sentiment analysis:
{{
  "llm_score": <0-100 sentiment score>,
  "key_catalysts": ["<factor1>", "<factor2>", "<factor3>"],
  "risk_factors": ["<risk1>", "<risk2>"],
  "short_term_outlook": "<1-2 sentence outlook>"
}}"""

    from langchain_core.messages import HumanMessage, SystemMessage
    system = "You are a quantitative market analyst. Analyze based only on the provided data. Respond with valid JSON only."

    results = {}
    for name, llm in llms[:1]:
        try:
            if isinstance(llm, dict) and llm.get("use_responses_api"):
                async with httpx.AsyncClient(timeout=180) as client:
                    resp = await client.post(
                        "https://api.openai.com/v1/responses",
                        headers={"Authorization": f"Bearer {llm['api_key']}", "Content-Type": "application/json"},
                        json={"model": "gpt-5.2-pro", "instructions": system, "input": prompt},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    content = ""
                    for item in data.get("output", []):
                        if item.get("type") == "message":
                            for c in item.get("content", []):
                                if c.get("type") == "output_text":
                                    content = c.get("text", "")
                    content = content.strip()
            else:
                response = await llm.ainvoke([
                    SystemMessage(content=system),
                    HumanMessage(content=prompt),
                ])
                content = response.content.strip()
            if content.startswith("```"):
                parts = content.split("```")
                if len(parts) >= 2:
                    block = parts[1]
                    if block.startswith("json"):
                        block = block[4:]
                    content = block.strip()
            try:
                results[name] = json.loads(content)
            except json.JSONDecodeError:
                import re
                match = re.search(r'\{[\s\S]*\}', content)
                if match:
                    results[name] = json.loads(match.group())
        except Exception as e:
            logger.warning(f"LLM sentiment {name} failed: {e}")

    return results


async def analyze_sentiment_real(tickers: List[str], real_prices: Dict = None) -> Dict[str, Any]:
    if real_prices is None:
        real_prices = {}

    x_tasks = {t: asyncio.create_task(_x_social_sentiment(t)) for t in tickers}

    results = {}
    for ticker in tickers:
        tech = _compute_technical_sentiment(ticker)
        if "error" in tech:
            real_q = real_prices.get(ticker, {})
            results[ticker] = {
                "ticker": ticker,
                "overall_score": 50,
                "sentiment": "neutral",
                "confidence": 30,
                "key_factors": ["Insufficient historical data for technical analysis"],
                "top_articles": [],
                "market_data": {
                    "price": real_q.get("price", 0),
                    "change": real_q.get("change", 0),
                    "change_pct": real_q.get("change_pct", 0),
                    "volume": real_q.get("volume", 0),
                },
                "source": "insufficient_data",
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
            }
            continue

        batch_result = None
        try:
            from prediction.batch_processor import get_cached_result
            batch_result = get_cached_result(ticker)
        except Exception:
            pass

        llm_data = {}
        if batch_result and "sentiment_score" in batch_result:
            llm_data["batch"] = {
                "llm_score": batch_result["sentiment_score"],
                "key_catalysts": batch_result.get("key_catalysts", []),
                "risk_factors": batch_result.get("risk_factors", []),
                "short_term_outlook": batch_result.get("short_term_outlook", ""),
            }
        else:
            try:
                llm_data = await _llm_sentiment(ticker, tech)
            except Exception as e:
                logger.warning(f"LLM sentiment for {ticker} failed: {e}")

        x_data = None
        try:
            x_data = await x_tasks.get(ticker)
        except Exception as e:
            logger.warning(f"X sentiment for {ticker} failed: {e}")

        llm_score = None
        catalysts = []
        risks = []
        outlook = ""
        for name, resp in llm_data.items():
            if isinstance(resp, dict) and "llm_score" in resp:
                llm_score = resp["llm_score"]
                catalysts = resp.get("key_catalysts", [])
                risks = resp.get("risk_factors", [])
                outlook = resp.get("short_term_outlook", "")

        x_score = None
        if x_data and isinstance(x_data.get("x_sentiment_score"), (int, float)):
            x_score = x_data["x_sentiment_score"]

        if llm_score is not None and x_score is not None:
            final_score = int(round(tech["score"] * 0.45 + llm_score * 0.25 + x_score * 0.30))
        elif x_score is not None:
            final_score = int(round(tech["score"] * 0.55 + x_score * 0.45))
        elif llm_score is not None:
            final_score = int(round(tech["score"] * 0.6 + llm_score * 0.4))
        else:
            final_score = tech["score"]
        final_score = max(5, min(95, final_score))

        if final_score >= 65:
            sentiment = "bullish"
        elif final_score <= 35:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        key_factors = []
        for sig in tech["signals"][:3]:
            key_factors.append(sig["factor"])

        if x_data:
            for narrative in x_data.get("key_narratives", [])[:2]:
                key_factors.append(f"[X] {narrative}")

        if catalysts:
            key_factors.extend(catalysts[:1])
        key_factors = key_factors[:6]

        articles = []

        if x_data:
            for xp in x_data.get("x_posts", [])[:4]:
                articles.append({
                    "title": xp.get("summary", ""),
                    "url": xp.get("url", f"#x-{ticker.lower()}"),
                    "sentiment": "positive" if xp.get("sentiment") == "positive" else "negative" if xp.get("sentiment") == "negative" else "neutral",
                    "source": "x",
                    "topic": xp.get("topic", "other"),
                })

        for sig in tech["signals"][:3]:
            articles.append({
                "title": sig["factor"],
                "url": f"#technical-{ticker.lower()}",
                "sentiment": "positive" if sig["type"] == "bullish" else "negative" if sig["type"] == "bearish" else "neutral",
                "source": "technical",
            })

        if risks:
            for risk in risks[:1]:
                articles.append({
                    "title": risk,
                    "url": f"#risk-{ticker.lower()}",
                    "sentiment": "negative",
                    "source": "llm",
                })
        articles = articles[:8]

        real_q = real_prices.get(ticker, {})
        price = tech["price_action"]["price"]
        chg_pct = tech["price_action"]["change_1d"]

        sources = ["technical_analysis"]
        if llm_score is not None:
            if batch_result and "sentiment_score" in batch_result:
                sources.append("batch_grok4")
            else:
                sources.append("llm")
        if x_data:
            sources.append("x_social")

        confidence = tech["confidence"]
        if x_data:
            confidence = min(98, confidence + 5)

        tech_breakdown = {
            "score": tech["score"],
            "weight": 0.45,
            "weighted_score": round(tech["score"] * 0.45, 1),
            "signals": tech["signals"][:6],
            "indicators": {},
        }
        for ind_name in ["rsi", "macd_hist", "adx", "cci", "stoch_k", "bb_pband"]:
            val = tech["technicals"].get(ind_name)
            if val is not None:
                tech_breakdown["indicators"][ind_name] = val

        social_breakdown = None
        if x_data:
            social_breakdown = {
                "score": x_score,
                "weight": 0.30,
                "weighted_score": round(x_score * 0.30, 1) if x_score else 0,
                "x_sentiment": x_data.get("x_sentiment", "neutral"),
                "post_volume": x_data.get("post_volume", "medium"),
                "posts_analyzed": len(x_data.get("x_posts", [])),
                "key_narratives": x_data.get("key_narratives", [])[:5],
                "notable_news": x_data.get("notable_news", [])[:3],
                "sentiment_distribution": x_data.get("sentiment_distribution", {}),
            }

        llm_breakdown_data = None
        if llm_score is not None:
            source_name = "batch_grok4" if (batch_result and "sentiment_score" in batch_result) else "real_time_llm"
            llm_breakdown_data = {
                "score": llm_score,
                "weight": 0.25,
                "weighted_score": round(llm_score * 0.25, 1),
                "source": source_name,
                "catalysts": catalysts[:5],
                "risks": risks[:5],
                "outlook": outlook,
            }

        result_entry = {
            "ticker": ticker,
            "overall_score": final_score,
            "sentiment": sentiment,
            "confidence": confidence,
            "key_factors": key_factors,
            "top_articles": articles,
            "market_data": {
                "price": price,
                "change": round(price * chg_pct / 100, 2),
                "change_pct": round(chg_pct, 2),
                "volume": real_q.get("volume", 0),
            },
            "technicals": tech["technicals"],
            "price_action": tech["price_action"],
            "source": "+".join(sources),
            "llm_outlook": outlook if outlook else None,
            "breakdown": {
                "technical": tech_breakdown,
                "social": social_breakdown,
                "llm": llm_breakdown_data,
            },
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }

        if x_data:
            result_entry["x_social"] = {
                "x_sentiment_score": x_score,
                "x_sentiment": x_data.get("x_sentiment", "neutral"),
                "key_narratives": x_data.get("key_narratives", []),
                "notable_news": x_data.get("notable_news", []),
                "post_volume": x_data.get("post_volume", "medium"),
                "posts_analyzed": len(x_data.get("x_posts", [])),
            }

        results[ticker] = result_entry

    return results
