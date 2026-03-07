import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

import httpx
from prediction.api_utils import async_resilient_get

from prediction.cache_manager import SmartCache

logger = logging.getLogger(__name__)

XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")

_news_cache = SmartCache("news_feed", max_size=500, default_ttl=300)
NEWS_CACHE_TTL = 300

_earnings_cache = SmartCache("earnings", max_size=500, default_ttl=3600)
EARNINGS_CACHE_TTL = 3600

_earnings_surprises_cache = SmartCache("earnings_surprises", max_size=500, default_ttl=3600)
EARNINGS_SURPRISES_CACHE_TTL = 3600


async def _fetch_fmp_company_news(tickers: List[str], limit: int = 50) -> List[Dict[str, Any]]:
    if not FMP_API_KEY:
        return []
    try:
        ticker_str = ",".join(tickers[:5])
        url = f"https://financialmodelingprep.com/stable/stock-news"
        params = {"tickers": ticker_str, "limit": str(limit), "apikey": FMP_API_KEY}
        resp = await async_resilient_get(url, params=params, timeout=15, source="fmp")
        if resp is None:
            return []
        if resp.status_code in (402, 403):
            logger.info(f"FMP stock-news returned {resp.status_code} — not available on current plan")
            return []
        if resp.status_code != 200:
            return []
        items = resp.json()
        if not isinstance(items, list):
            return []
        results = []
        for item in items:
            pub_date = item.get("publishedDate", "")
            hours_ago = 1
            if pub_date:
                try:
                    dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                    hours_ago = max(0, int((datetime.now(timezone.utc) - dt).total_seconds() / 3600))
                except Exception:
                    pass
            results.append({
                "headline": item.get("title", ""),
                "summary": item.get("text", "")[:300],
                "source": item.get("site", "FMP"),
                "url": item.get("url", ""),
                "image": item.get("image", ""),
                "tickers": [item.get("symbol", "")] if item.get("symbol") else [],
                "hours_ago": hours_ago,
                "timestamp": pub_date,
                "provider": "fmp",
            })
        return results
    except Exception as e:
        logger.warning(f"FMP company news error: {e}")
        return []


async def _fetch_fmp_general_news(page: int = 0) -> List[Dict[str, Any]]:
    if not FMP_API_KEY:
        return []
    try:
        url = f"https://financialmodelingprep.com/stable/news"
        params = {"page": str(page), "apikey": FMP_API_KEY}
        resp = await async_resilient_get(url, params=params, timeout=15, source="fmp")
        if resp is None:
            return []
        if resp.status_code in (402, 403):
            return []
        if resp.status_code != 200:
            return []
        items = resp.json()
        if not isinstance(items, list):
            return []
        results = []
        for item in items:
            pub_date = item.get("publishedDate", "")
            hours_ago = 1
            if pub_date:
                try:
                    dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                    hours_ago = max(0, int((datetime.now(timezone.utc) - dt).total_seconds() / 3600))
                except Exception:
                    pass
            results.append({
                "headline": item.get("title", ""),
                "summary": item.get("text", "")[:300],
                "source": item.get("site", item.get("source", "FMP")),
                "url": item.get("url", ""),
                "image": item.get("image", ""),
                "tickers": [],
                "hours_ago": hours_ago,
                "timestamp": pub_date,
                "provider": "fmp",
            })
        return results
    except Exception as e:
        logger.warning(f"FMP general news error: {e}")
        return []


async def _fetch_finnhub_general_news() -> List[Dict[str, Any]]:
    if not FINNHUB_API_KEY:
        return []
    try:
        url = "https://finnhub.io/api/v1/news"
        params = {"category": "general", "token": FINNHUB_API_KEY}
        resp = await async_resilient_get(url, params=params, timeout=15, source="finnhub")
        if resp is None or resp.status_code != 200:
            return []
        items = resp.json()
        if not isinstance(items, list):
            return []
        results = []
        for item in items:
            ts = item.get("datetime", 0)
            hours_ago = max(0, int((time.time() - ts) / 3600)) if ts else 1
            results.append({
                "headline": item.get("headline", ""),
                "summary": item.get("summary", "")[:300],
                "source": item.get("source", "Finnhub"),
                "url": item.get("url", ""),
                "image": item.get("image", ""),
                "tickers": [item.get("related", "")] if item.get("related") else [],
                "hours_ago": hours_ago,
                "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else "",
                "provider": "finnhub",
            })
        return results
    except Exception as e:
        logger.warning(f"Finnhub general news error: {e}")
        return []


async def _fetch_finnhub_company_news(ticker: str) -> List[Dict[str, Any]]:
    if not FINNHUB_API_KEY:
        return []
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        url = "https://finnhub.io/api/v1/company-news"
        params = {"symbol": ticker, "from": from_date, "to": today, "token": FINNHUB_API_KEY}
        resp = await async_resilient_get(url, params=params, timeout=15, source="finnhub")
        if resp is None or resp.status_code != 200:
            return []
        items = resp.json()
        if not isinstance(items, list):
            return []
        results = []
        for item in items[:20]:
            ts = item.get("datetime", 0)
            hours_ago = max(0, int((time.time() - ts) / 3600)) if ts else 1
            results.append({
                "headline": item.get("headline", ""),
                "summary": item.get("summary", "")[:300],
                "source": item.get("source", "Finnhub"),
                "url": item.get("url", ""),
                "image": item.get("image", ""),
                "tickers": [ticker],
                "hours_ago": hours_ago,
                "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else "",
                "provider": "finnhub",
            })
        return results
    except Exception as e:
        logger.warning(f"Finnhub company news error for {ticker}: {e}")
        return []


def _deduplicate_news(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for item in items:
        headline = item.get("headline", "").strip().lower()
        if not headline:
            continue
        norm = re.sub(r'[^a-z0-9\s]', '', headline)[:80]
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(item)
    return unique


def _classify_sentiment(headline: str, summary: str) -> str:
    text = (headline + " " + summary).lower()
    bullish_words = ["surge", "soar", "rally", "beat", "upgrade", "bullish", "gain", "record high",
                     "outperform", "positive", "growth", "profit", "strong", "exceeds", "raise",
                     "buy", "breakout", "momentum", "upside"]
    bearish_words = ["crash", "plunge", "drop", "miss", "downgrade", "bearish", "loss", "decline",
                     "underperform", "negative", "weak", "warning", "cut", "sell", "fall",
                     "layoff", "recall", "investigation", "lawsuit", "debt"]
    bull_count = sum(1 for w in bullish_words if w in text)
    bear_count = sum(1 for w in bearish_words if w in text)
    if bull_count > bear_count:
        return "bullish"
    elif bear_count > bull_count:
        return "bearish"
    return "neutral"


def _classify_impact(headline: str) -> str:
    text = headline.lower()
    high_impact = ["earnings", "fda", "merger", "acquisition", "bankruptcy", "sec",
                   "ceo", "guidance", "recall", "investigation", "antitrust"]
    for word in high_impact:
        if word in text:
            return "high"
    medium_impact = ["upgrade", "downgrade", "analyst", "target", "revenue",
                     "partnership", "launch", "expand"]
    for word in medium_impact:
        if word in text:
            return "medium"
    return "low"


def _classify_category(headline: str) -> str:
    text = headline.lower()
    categories = {
        "earnings": ["earnings", "eps", "revenue", "quarterly", "q1", "q2", "q3", "q4", "profit", "income"],
        "analyst": ["upgrade", "downgrade", "target", "analyst", "rating", "price target"],
        "merger": ["merger", "acquisition", "acquire", "deal", "buyout", "takeover"],
        "regulatory": ["sec", "fda", "regulation", "antitrust", "investigation", "lawsuit", "compliance"],
        "product": ["launch", "product", "release", "unveil", "announce", "patent"],
        "macro": ["fed", "inflation", "interest rate", "gdp", "jobs", "unemployment", "tariff"],
        "sector": ["industry", "sector", "market", "trend"],
    }
    for cat, words in categories.items():
        if any(w in text for w in words):
            return cat
    return "market"


async def fetch_financial_news(tickers: List[str], max_items: int = 20) -> List[Dict[str, Any]]:
    cache_key = ",".join(sorted(tickers))
    now = time.time()
    cached_data = _news_cache.get(cache_key, NEWS_CACHE_TTL)
    if cached_data is not None:
        return cached_data

    tasks = []
    if FMP_API_KEY:
        tasks.append(_fetch_fmp_company_news(tickers))
    if FINNHUB_API_KEY:
        for t in tickers[:3]:
            tasks.append(_fetch_finnhub_company_news(t))
        tasks.append(_fetch_finnhub_general_news())
    if FMP_API_KEY:
        tasks.append(_fetch_fmp_general_news())

    all_news = []
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                all_news.extend(result)

    if not all_news:
        all_news = _fallback_news(tickers)
        _news_cache.set(cache_key, all_news)
        return all_news

    all_news = _deduplicate_news(all_news)

    processed = []
    for i, item in enumerate(all_news):
        headline = item.get("headline", "")
        summary = item.get("summary", "")
        if not headline:
            continue
        processed.append({
            "id": f"news_{int(now)}_{i}",
            "headline": headline,
            "summary": summary,
            "source": item.get("source", "Market Data"),
            "url": item.get("url", ""),
            "image": item.get("image", ""),
            "sentiment": _classify_sentiment(headline, summary),
            "impact": _classify_impact(headline),
            "tickers": item.get("tickers", []),
            "category": _classify_category(headline),
            "hours_ago": item.get("hours_ago", 1),
            "timestamp": item.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "provider": item.get("provider", ""),
        })

    processed.sort(key=lambda x: x.get("hours_ago", 999))
    processed = processed[:max_items]

    _news_cache.set(cache_key, processed)
    return processed


async def _fetch_fmp_earnings_calendar() -> List[Dict[str, Any]]:
    if not FMP_API_KEY:
        return []
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        future = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        url = "https://financialmodelingprep.com/stable/earning-calendar"
        params = {"from": today, "to": future, "apikey": FMP_API_KEY}
        resp = await async_resilient_get(url, params=params, timeout=15, source="fmp")
        if resp is None:
            return []
        if resp.status_code in (402, 403):
            return []
        if resp.status_code != 200:
            return []
        items = resp.json()
        if not isinstance(items, list):
            return []
        return items
    except Exception as e:
        logger.warning(f"FMP earnings calendar error: {e}")
        return []


async def _fetch_fmp_earnings_surprises(ticker: str) -> List[Dict[str, Any]]:
    if not FMP_API_KEY:
        return []
    try:
        url = "https://financialmodelingprep.com/stable/earnings-surprises"
        params = {"symbol": ticker, "apikey": FMP_API_KEY}
        resp = await async_resilient_get(url, params=params, timeout=15, source="fmp")
        if resp is None:
            return []
        if resp.status_code in (402, 403):
            return []
        if resp.status_code != 200:
            return []
        items = resp.json()
        if not isinstance(items, list):
            return []
        return items
    except Exception as e:
        logger.warning(f"FMP earnings surprises error for {ticker}: {e}")
        return []


async def _fetch_finnhub_earnings_calendar() -> List[Dict[str, Any]]:
    if not FINNHUB_API_KEY:
        return []
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        future = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        url = "https://finnhub.io/api/v1/calendar/earnings"
        params = {"from": today, "to": future, "token": FINNHUB_API_KEY}
        resp = await async_resilient_get(url, params=params, timeout=15, source="finnhub")
        if resp is None or resp.status_code != 200:
            return []
        data = resp.json()
        return data.get("earningsCalendar", [])
    except Exception as e:
        logger.warning(f"Finnhub earnings calendar error: {e}")
        return []


async def _fetch_finnhub_earnings(ticker: str) -> List[Dict[str, Any]]:
    if not FINNHUB_API_KEY:
        return []
    try:
        url = "https://finnhub.io/api/v1/stock/earnings"
        params = {"symbol": ticker, "token": FINNHUB_API_KEY}
        resp = await async_resilient_get(url, params=params, timeout=15, source="finnhub")
        if resp is None or resp.status_code != 200:
            return []
        items = resp.json()
        if not isinstance(items, list):
            return []
        return items
    except Exception as e:
        logger.warning(f"Finnhub earnings error for {ticker}: {e}")
        return []


async def get_earnings_calendar(tickers: List[str]) -> List[Dict[str, Any]]:
    cache_key = ",".join(sorted(tickers))
    cached_data = _earnings_cache.get(cache_key, EARNINGS_CACHE_TTL)
    if cached_data is not None:
        return cached_data

    try:
        earnings = await _async_get_earnings_calendar(tickers)
    except Exception as e:
        logger.error(f"Earnings calendar error: {e}")
        earnings = _fallback_earnings_calendar(tickers)

    _earnings_cache.set(cache_key, earnings)
    return earnings


def get_earnings_calendar_sync(tickers: List[str]) -> List[Dict[str, Any]]:
    cache_key = ",".join(sorted(tickers))
    cached_data = _earnings_cache.get(cache_key, EARNINGS_CACHE_TTL)
    if cached_data is not None:
        return cached_data

    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                def _run():
                    new_loop = asyncio.new_event_loop()
                    try:
                        return new_loop.run_until_complete(_async_get_earnings_calendar(tickers))
                    finally:
                        new_loop.close()
                earnings = pool.submit(_run).result(timeout=30)
        else:
            earnings = asyncio.run(_async_get_earnings_calendar(tickers))
    except Exception as e:
        logger.error(f"Earnings calendar error: {e}")
        earnings = _fallback_earnings_calendar(tickers)

    _earnings_cache.set(cache_key, earnings)
    return earnings


async def _async_get_earnings_calendar(tickers: List[str]) -> List[Dict[str, Any]]:
    ticker_set = set(t.upper() for t in tickers)
    earnings = []
    seen_tickers = set()

    tasks = []
    if FMP_API_KEY:
        tasks.append(_fetch_fmp_earnings_calendar())
    if FINNHUB_API_KEY:
        tasks.append(_fetch_finnhub_earnings_calendar())

    surprise_tasks = []
    if FMP_API_KEY:
        for t in tickers[:5]:
            surprise_tasks.append(_fetch_fmp_earnings_surprises(t))
    if FINNHUB_API_KEY:
        for t in tickers[:5]:
            surprise_tasks.append(_fetch_finnhub_earnings(t))

    all_tasks = tasks + surprise_tasks
    if not all_tasks:
        return _fallback_earnings_calendar(tickers)

    results = await asyncio.gather(*all_tasks, return_exceptions=True)

    calendar_results = results[:len(tasks)]
    surprise_results = results[len(tasks):]

    surprises_by_ticker: Dict[str, List[Dict]] = {}

    si = 0
    if FMP_API_KEY:
        for t in tickers[:5]:
            if si < len(surprise_results) and isinstance(surprise_results[si], list):
                surprises_by_ticker[t.upper()] = surprise_results[si][:4]
            si += 1
    if FINNHUB_API_KEY:
        for t in tickers[:5]:
            if si < len(surprise_results) and isinstance(surprise_results[si], list):
                existing = surprises_by_ticker.get(t.upper(), [])
                if not existing:
                    finnhub_surprises = []
                    for item in surprise_results[si][:4]:
                        finnhub_surprises.append({
                            "date": item.get("period", ""),
                            "actualEarningResult": item.get("actual"),
                            "estimatedEarning": item.get("estimate"),
                        })
                    surprises_by_ticker[t.upper()] = finnhub_surprises
            si += 1

    for result in calendar_results:
        if isinstance(result, Exception) or not isinstance(result, list):
            continue
        for item in result:
            symbol = item.get("symbol", item.get("ticker", "")).upper()
            if symbol not in ticker_set or symbol in seen_tickers:
                continue
            seen_tickers.add(symbol)

            date_str = item.get("date", item.get("reportDate", ""))
            if not date_str:
                continue
            try:
                days_until = (datetime.strptime(date_str[:10], "%Y-%m-%d") - datetime.now()).days
            except Exception:
                days_until = 0

            eps_est = item.get("epsEstimate", item.get("epsEstimated", item.get("estimate")))
            rev_est = item.get("revenueEstimate", item.get("revenueEstimated"))

            surprise_history = []
            if symbol in surprises_by_ticker:
                for s in surprises_by_ticker[symbol]:
                    actual = s.get("actualEarningResult", s.get("actual"))
                    estimate = s.get("estimatedEarning", s.get("estimate"))
                    if actual is not None and estimate is not None:
                        try:
                            actual_f = float(actual)
                            estimate_f = float(estimate)
                            surprise_pct = round(((actual_f - estimate_f) / abs(estimate_f)) * 100, 1) if estimate_f != 0 else 0
                            surprise_history.append({
                                "date": s.get("date", s.get("period", "")),
                                "actual": actual_f,
                                "estimate": estimate_f,
                                "surprise_pct": surprise_pct,
                                "beat": actual_f > estimate_f,
                            })
                        except (ValueError, TypeError):
                            pass

            earnings.append({
                "ticker": symbol,
                "earnings_date": date_str[:10],
                "days_until": max(0, days_until),
                "eps_estimate": eps_est,
                "revenue_estimate": rev_est,
                "is_upcoming": 0 <= days_until <= 30,
                "surprise_history": surprise_history,
                "beat_rate": round(sum(1 for s in surprise_history if s.get("beat", False)) / len(surprise_history) * 100, 1) if surprise_history else None,
            })

    for t in tickers:
        t_upper = t.upper()
        if t_upper not in seen_tickers:
            surprise_history = []
            if t_upper in surprises_by_ticker:
                for s in surprises_by_ticker[t_upper]:
                    actual = s.get("actualEarningResult", s.get("actual"))
                    estimate = s.get("estimatedEarning", s.get("estimate"))
                    if actual is not None and estimate is not None:
                        try:
                            actual_f = float(actual)
                            estimate_f = float(estimate)
                            surprise_pct = round(((actual_f - estimate_f) / abs(estimate_f)) * 100, 1) if estimate_f != 0 else 0
                            surprise_history.append({
                                "date": s.get("date", s.get("period", "")),
                                "actual": actual_f,
                                "estimate": estimate_f,
                                "surprise_pct": surprise_pct,
                                "beat": actual_f > estimate_f,
                            })
                        except (ValueError, TypeError):
                            pass
            if surprise_history:
                earnings.append({
                    "ticker": t_upper,
                    "earnings_date": None,
                    "days_until": None,
                    "eps_estimate": None,
                    "revenue_estimate": None,
                    "is_upcoming": False,
                    "surprise_history": surprise_history,
                    "beat_rate": round(sum(1 for s in surprise_history if s.get("beat", False)) / len(surprise_history) * 100, 1),
                })

    if not earnings:
        return _fallback_earnings_calendar(tickers)

    earnings.sort(key=lambda x: x.get("days_until") if x.get("days_until") is not None else 999)
    return earnings


def _fallback_earnings_calendar(tickers: List[str]) -> List[Dict[str, Any]]:
    try:
        import yfinance as yf
        earnings = []
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                cal = t.calendar
                if cal is not None and not (hasattr(cal, 'empty') and cal.empty):
                    if isinstance(cal, dict):
                        ed = cal.get("Earnings Date")
                        if ed:
                            if isinstance(ed, list) and len(ed) > 0:
                                next_date = ed[0]
                            else:
                                next_date = ed
                            if hasattr(next_date, 'isoformat'):
                                date_str = next_date.strftime("%Y-%m-%d")
                            else:
                                date_str = str(next_date)[:10]
                            days_until = (datetime.strptime(date_str, "%Y-%m-%d") - datetime.now()).days
                            earnings.append({
                                "ticker": ticker,
                                "earnings_date": date_str,
                                "days_until": max(0, days_until),
                                "eps_estimate": cal.get("EPS Estimate"),
                                "revenue_estimate": cal.get("Revenue Estimate"),
                                "is_upcoming": 0 <= days_until <= 30,
                                "surprise_history": [],
                                "beat_rate": None,
                            })
            except Exception:
                continue
        earnings.sort(key=lambda x: x.get("days_until", 999))
        return earnings
    except Exception:
        return []


def _fallback_news(tickers: List[str]) -> List[Dict[str, Any]]:
    import yfinance as yf

    news_items = []
    seen_titles = set()
    for ticker in tickers[:5]:
        try:
            t = yf.Ticker(ticker)
            news = t.news
            if news:
                for item in news[:5]:
                    title = item.get("title", "")
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        pub_time = item.get("providerPublishTime", int(time.time()))
                        hours_ago = max(1, int((time.time() - pub_time) / 3600))
                        news_items.append({
                            "id": f"yf_{hash(title) % 100000}",
                            "headline": title,
                            "summary": item.get("title", ""),
                            "source": item.get("publisher", "Yahoo Finance"),
                            "url": item.get("link", ""),
                            "image": "",
                            "sentiment": _classify_sentiment(title, ""),
                            "impact": _classify_impact(title),
                            "tickers": [ticker],
                            "category": _classify_category(title),
                            "hours_ago": hours_ago,
                            "timestamp": datetime.fromtimestamp(pub_time, tz=timezone.utc).isoformat(),
                            "provider": "yfinance",
                        })
        except Exception as e:
            logger.debug(f"yfinance news fallback failed for {ticker}: {e}")

    news_items.sort(key=lambda x: x.get("hours_ago", 999))
    return news_items[:20]


async def fetch_news_for_events(tickers: List[str], max_items: int = 10) -> List[Dict[str, Any]]:
    tasks = []
    if FMP_API_KEY:
        tasks.append(_fetch_fmp_company_news(tickers, limit=20))
    if FINNHUB_API_KEY:
        for t in tickers[:3]:
            tasks.append(_fetch_finnhub_company_news(t))

    all_news = []
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                all_news.extend(result)

    all_news = _deduplicate_news(all_news)
    all_news.sort(key=lambda x: x.get("hours_ago", 999))
    return all_news[:max_items]


try:
    import pandas as pd
except ImportError:
    pd = None
