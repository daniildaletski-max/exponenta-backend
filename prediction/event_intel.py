import asyncio
import hashlib
import json
import logging
import os
import random
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

XAI_API_KEY = os.environ.get("XAI_API_KEY", "")

_event_cache: Dict[str, tuple] = {}
EVENT_CACHE_TTL = 300

_live_feed_cache: Optional[tuple] = None
LIVE_FEED_CACHE_TTL = 180

EVENT_CATEGORIES = [
    "Earnings Surprise",
    "FDA Approval",
    "M&A",
    "Activist Investor",
    "Macro Policy",
    "Guidance Change",
    "Analyst Upgrade/Downgrade",
    "Product Launch",
    "Legal/Regulatory",
    "Management Change",
]


def _cache_key(ticker: str) -> str:
    return f"event_intel:{ticker}"


def _get_cached(key: str):
    if key in _event_cache:
        ts, data = _event_cache[key]
        if time.time() - ts < EVENT_CACHE_TTL:
            return data
    return None


def _set_cached(key: str, data):
    _event_cache[key] = (time.time(), data)


def _extract_json(content: str) -> Any:
    import re
    content = content.strip()
    if content.startswith("```"):
        parts = content.split("```")
        if len(parts) >= 2:
            block = parts[1]
            if block.startswith("json"):
                block = block[4:]
            return json.loads(block.strip())
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[\s\S]*\}', content)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    match = re.search(r'\[[\s\S]*\]', content)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _historical_reaction_patterns(ticker: str) -> List[Dict[str, Any]]:
    try:
        from prediction.features import fetch_ohlcv
        import numpy as np

        df = fetch_ohlcv(ticker, days=365)
        if df is None or len(df) < 60:
            return []

        closes = df["close"].values
        returns = []
        for i in range(1, len(closes)):
            returns.append((closes[i] - closes[i - 1]) / closes[i - 1] * 100)

        if not returns:
            return []

        returns_arr = np.array(returns)
        std_dev = np.std(returns_arr)
        mean_ret = np.mean(returns_arr)

        big_moves = []
        for i, r in enumerate(returns):
            if abs(r) > std_dev * 2:
                next_day = returns[i + 1] if i + 1 < len(returns) else 0
                next_5d = sum(returns[i + 1:i + 6]) if i + 5 < len(returns) else sum(returns[i + 1:])
                big_moves.append({
                    "date": str(df["date"].iloc[i + 1])[:10] if i + 1 < len(df) else "N/A",
                    "move_pct": round(r, 2),
                    "direction": "positive" if r > 0 else "negative",
                    "next_day_pct": round(next_day, 2),
                    "next_5d_pct": round(next_5d, 2),
                })

        patterns = []
        if big_moves:
            pos_moves = [m for m in big_moves if m["direction"] == "positive"]
            neg_moves = [m for m in big_moves if m["direction"] == "negative"]

            if pos_moves:
                avg_next_day = sum(m["next_day_pct"] for m in pos_moves) / len(pos_moves)
                avg_next_5d = sum(m["next_5d_pct"] for m in pos_moves) / len(pos_moves)
                patterns.append({
                    "pattern": "After large positive moves",
                    "occurrences": len(pos_moves),
                    "avg_move": round(sum(m["move_pct"] for m in pos_moves) / len(pos_moves), 2),
                    "avg_next_day": round(avg_next_day, 2),
                    "avg_next_5d": round(avg_next_5d, 2),
                    "continuation_rate": round(len([m for m in pos_moves if m["next_day_pct"] > 0]) / len(pos_moves) * 100, 1),
                })

            if neg_moves:
                avg_next_day = sum(m["next_day_pct"] for m in neg_moves) / len(neg_moves)
                avg_next_5d = sum(m["next_5d_pct"] for m in neg_moves) / len(neg_moves)
                patterns.append({
                    "pattern": "After large negative moves",
                    "occurrences": len(neg_moves),
                    "avg_move": round(sum(m["move_pct"] for m in neg_moves) / len(neg_moves), 2),
                    "avg_next_day": round(avg_next_day, 2),
                    "avg_next_5d": round(avg_next_5d, 2),
                    "reversal_rate": round(len([m for m in neg_moves if m["next_day_pct"] > 0]) / len(neg_moves) * 100, 1),
                })

        earnings_pattern = _earnings_reaction_pattern(ticker)
        if earnings_pattern:
            patterns.append(earnings_pattern)

        return patterns
    except Exception as e:
        logger.warning(f"Historical patterns error for {ticker}: {e}")
        return []


def _earnings_reaction_pattern(ticker: str) -> Optional[Dict[str, Any]]:
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        cal = t.calendar
        if cal is None:
            return None

        hist = t.history(period="2y")
        if hist.empty or len(hist) < 60:
            return None

        try:
            earnings_dates = t.earnings_dates
            if earnings_dates is not None and len(earnings_dates) > 0:
                reactions = []
                for ed in earnings_dates.index[:8]:
                    ed_str = ed.strftime("%Y-%m-%d")
                    try:
                        idx = hist.index.get_indexer([ed], method="nearest")[0]
                        if idx > 0 and idx < len(hist) - 1:
                            day_before = float(hist["Close"].iloc[idx - 1])
                            day_of = float(hist["Close"].iloc[idx])
                            reaction = (day_of - day_before) / day_before * 100
                            reactions.append(round(reaction, 2))
                    except Exception:
                        continue

                if reactions:
                    beats = len([r for r in reactions if r > 0])
                    return {
                        "pattern": "Earnings reactions",
                        "occurrences": len(reactions),
                        "reactions": reactions[:4],
                        "avg_reaction": round(sum(reactions) / len(reactions), 2),
                        "positive_rate": round(beats / len(reactions) * 100, 1),
                        "description": f"Last {len(reactions)} earnings: avg {sum(reactions)/len(reactions):+.1f}%, positive {beats}/{len(reactions)} times",
                    }
        except Exception:
            pass

        return None
    except Exception:
        return None


def _cascading_impacts(ticker: str) -> List[Dict[str, Any]]:
    sector_map = {
        "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMD"],
        "Automotive": ["TSLA", "F", "GM", "RIVN"],
        "Consumer": ["AMZN", "WMT", "COST", "TGT"],
        "Finance": ["JPM", "GS", "BAC", "V", "MA"],
        "Entertainment": ["DIS", "NFLX", "CMCSA"],
        "Industrial": ["BA", "CAT", "HON", "GE"],
    }

    from data.market_data import TICKER_SECTORS
    ticker_sector = TICKER_SECTORS.get(ticker, "Other")

    related = []
    if ticker_sector in sector_map:
        peers = [t for t in sector_map[ticker_sector] if t != ticker]
        for peer in peers[:4]:
            try:
                from prediction.features import fetch_ohlcv
                import numpy as np

                df_main = fetch_ohlcv(ticker, days=90)
                df_peer = fetch_ohlcv(peer, days=90)

                if df_main is not None and df_peer is not None and len(df_main) > 20 and len(df_peer) > 20:
                    ret_main = np.diff(df_main["close"].values[-30:]) / df_main["close"].values[-31:-1]
                    ret_peer = np.diff(df_peer["close"].values[-30:]) / df_peer["close"].values[-31:-1]
                    min_len = min(len(ret_main), len(ret_peer))
                    if min_len > 5:
                        corr = np.corrcoef(ret_main[:min_len], ret_peer[:min_len])[0, 1]
                        related.append({
                            "ticker": peer,
                            "sector": ticker_sector,
                            "correlation": round(float(corr), 3),
                            "impact_likelihood": "high" if abs(corr) > 0.7 else "medium" if abs(corr) > 0.4 else "low",
                        })
            except Exception:
                related.append({
                    "ticker": peer,
                    "sector": ticker_sector,
                    "correlation": 0.5,
                    "impact_likelihood": "medium",
                })

    return sorted(related, key=lambda x: abs(x["correlation"]), reverse=True)


async def _fetch_events_real(ticker: str) -> List[Dict[str, Any]]:
    from prediction.news_feed import fetch_news_for_events
    try:
        raw_news = await fetch_news_for_events([ticker], max_items=15)
    except Exception as e:
        logger.warning(f"Real news fetch failed for {ticker}: {e}")
        raw_news = []

    if not raw_news:
        return _fallback_events(ticker)

    CATEGORY_KEYWORDS = {
        "Earnings Surprise": ["earnings", "eps", "revenue", "quarterly", "q1", "q2", "q3", "q4", "profit", "income statement"],
        "FDA Approval": ["fda", "drug", "approval", "clinical", "trial", "pharma"],
        "M&A": ["merger", "acquisition", "acquire", "deal", "buyout", "takeover"],
        "Activist Investor": ["activist", "stake", "board seat", "proxy"],
        "Macro Policy": ["fed", "inflation", "interest rate", "gdp", "tariff", "policy", "treasury"],
        "Guidance Change": ["guidance", "outlook", "forecast", "raise", "lower", "revise"],
        "Analyst Upgrade/Downgrade": ["upgrade", "downgrade", "price target", "analyst", "rating", "overweight", "underweight"],
        "Product Launch": ["launch", "product", "release", "unveil", "announce", "patent", "innovation"],
        "Legal/Regulatory": ["sec", "lawsuit", "regulation", "fine", "investigation", "settlement", "compliance"],
        "Management Change": ["ceo", "cfo", "executive", "resign", "appoint", "hire", "management"],
    }

    BULLISH_WORDS = ["surge", "soar", "rally", "beat", "upgrade", "gain", "record",
                     "outperform", "growth", "profit", "strong", "exceeds", "raise",
                     "buy", "breakout", "upside", "positive", "bullish"]
    BEARISH_WORDS = ["crash", "plunge", "drop", "miss", "downgrade", "loss", "decline",
                     "underperform", "weak", "warning", "cut", "sell", "fall",
                     "layoff", "recall", "investigation", "lawsuit", "bearish"]

    events = []
    for item in raw_news:
        headline = item.get("headline", "")
        summary = item.get("summary", "")
        if not headline:
            continue

        text = (headline + " " + summary).lower()

        category = "Macro Policy"
        for cat, keywords in CATEGORY_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                category = cat
                break

        bull = sum(1 for w in BULLISH_WORDS if w in text)
        bear = sum(1 for w in BEARISH_WORDS if w in text)
        if bull > bear:
            direction = "positive"
        elif bear > bull:
            direction = "negative"
        else:
            direction = "neutral"

        high_impact_words = ["earnings", "fda", "merger", "acquisition", "bankruptcy",
                             "sec", "ceo", "guidance", "recall", "investigation"]
        if any(w in text for w in high_impact_words):
            magnitude = min(10, 7 + bull + bear)
        elif any(w in text for w in ["upgrade", "downgrade", "target", "revenue"]):
            magnitude = min(9, 5 + bull + bear)
        else:
            magnitude = min(7, 3 + bull + bear)

        hours_ago = item.get("hours_ago", 1)
        if hours_ago <= 2:
            time_horizon = "intraday"
        elif hours_ago <= 48:
            time_horizon = "swing"
        else:
            time_horizon = "long-term"

        event = {
            "id": hashlib.md5(f"{ticker}:{headline}:{time.time()}".encode()).hexdigest()[:12],
            "ticker": ticker,
            "title": headline,
            "category": category,
            "impact_magnitude": min(10, max(1, magnitude)),
            "direction": direction,
            "time_horizon": time_horizon,
            "summary": summary,
            "source": item.get("source", "Market Data"),
            "url": item.get("url", ""),
            "hours_ago": max(0, hours_ago),
            "timestamp": item.get("timestamp", datetime.now(timezone.utc).isoformat()),
        }
        events.append(event)

    return events[:10] if events else _fallback_events(ticker)


def _fallback_events(ticker: str) -> List[Dict[str, Any]]:
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.fast_info
        price = float(info.get("lastPrice", 0) or info.get("last_price", 0) or 0)
        prev = float(info.get("previousClose", 0) or info.get("previous_close", 0) or price)
        change_pct = ((price - prev) / prev * 100) if prev > 0 else 0
    except Exception:
        price = 0
        change_pct = 0

    events = []

    if abs(change_pct) > 3:
        events.append({
            "id": hashlib.md5(f"{ticker}:move:{time.time()}".encode()).hexdigest()[:12],
            "ticker": ticker,
            "title": f"{ticker} {'surges' if change_pct > 0 else 'drops'} {abs(change_pct):.1f}%",
            "category": "Macro Policy",
            "impact_magnitude": min(10, max(3, int(abs(change_pct)))),
            "direction": "positive" if change_pct > 0 else "negative",
            "time_horizon": "intraday",
            "summary": f"{ticker} moved {change_pct:+.1f}% in today's session, representing a significant price action event.",
            "source": "Price Action",
            "url": "",
            "hours_ago": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    try:
        from prediction.news_feed import _fallback_news
        news = _fallback_news([ticker])
        for item in news[:3]:
            cat = "Macro Policy"
            direction = item.get("sentiment", "neutral")
            if direction == "bullish":
                direction = "positive"
            elif direction == "bearish":
                direction = "negative"

            impact_map = {"high": 8, "medium": 5, "low": 3}
            magnitude = impact_map.get(item.get("impact", "medium"), 5)

            events.append({
                "id": item.get("id", hashlib.md5(f"{ticker}:news:{time.time()}:{random.random()}".encode()).hexdigest()[:12]),
                "ticker": ticker,
                "title": item.get("headline", "Market update"),
                "category": cat,
                "impact_magnitude": magnitude,
                "direction": direction,
                "time_horizon": "swing",
                "summary": item.get("summary", ""),
                "source": item.get("source", "Market Data"),
                "url": item.get("url", ""),
                "hours_ago": item.get("hours_ago", 1),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
    except Exception:
        pass

    if not events:
        events.append({
            "id": hashlib.md5(f"{ticker}:monitoring:{time.time()}".encode()).hexdigest()[:12],
            "ticker": ticker,
            "title": f"{ticker} — Monitoring for events",
            "category": "Macro Policy",
            "impact_magnitude": 2,
            "direction": "neutral",
            "time_horizon": "swing",
            "summary": f"No major market-moving events detected for {ticker} in the last 48 hours. Normal trading conditions.",
            "source": "Event Monitor",
            "url": "",
            "hours_ago": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    return events


async def get_event_intelligence(ticker: str) -> Dict[str, Any]:
    ticker = ticker.upper()
    cache_key = _cache_key(ticker)
    cached = _get_cached(cache_key)
    if cached:
        return cached

    events = await _fetch_events_real(ticker)
    patterns = _historical_reaction_patterns(ticker)

    try:
        cascading = _cascading_impacts(ticker)
    except Exception:
        cascading = []

    critical_count = len([e for e in events if e.get("impact_magnitude", 0) >= 7])
    positive_count = len([e for e in events if e.get("direction") == "positive"])
    negative_count = len([e for e in events if e.get("direction") == "negative"])

    if positive_count > negative_count:
        overall_sentiment = "positive"
    elif negative_count > positive_count:
        overall_sentiment = "negative"
    else:
        overall_sentiment = "neutral"

    avg_magnitude = round(sum(e.get("impact_magnitude", 5) for e in events) / max(len(events), 1), 1)

    result = {
        "ticker": ticker,
        "events": events,
        "event_count": len(events),
        "critical_count": critical_count,
        "overall_sentiment": overall_sentiment,
        "avg_impact_magnitude": avg_magnitude,
        "historical_patterns": patterns,
        "cascading_impacts": cascading,
        "categories": list(set(e.get("category", "") for e in events)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    _set_cached(cache_key, result)
    return result


async def get_live_event_feed(tickers: Optional[List[str]] = None) -> Dict[str, Any]:
    global _live_feed_cache

    if _live_feed_cache is not None:
        ts, data = _live_feed_cache
        if time.time() - ts < LIVE_FEED_CACHE_TTL:
            return data

    if not tickers:
        tickers = ["AAPL", "TSLA", "NVDA", "GOOGL", "MSFT", "AMZN", "META"]

    all_events = []
    tasks = [_fetch_events_real(t) for t in tickers[:7]]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning(f"Event fetch failed for {tickers[i]}: {result}")
            continue
        if isinstance(result, list):
            all_events.extend(result)

    all_events.sort(key=lambda e: e.get("hours_ago", 999))

    critical = [e for e in all_events if e.get("impact_magnitude", 0) >= 7]
    positive = len([e for e in all_events if e.get("direction") == "positive"])
    negative = len([e for e in all_events if e.get("direction") == "negative"])

    category_counts: Dict[str, int] = {}
    for e in all_events:
        cat = e.get("category", "Other")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    result = {
        "events": all_events[:30],
        "total_events": len(all_events),
        "critical_events": len(critical),
        "positive_events": positive,
        "negative_events": negative,
        "neutral_events": len(all_events) - positive - negative,
        "category_distribution": category_counts,
        "tickers_monitored": tickers,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    _live_feed_cache = (time.time(), result)
    return result
