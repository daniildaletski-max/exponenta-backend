import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)

PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

_alert_cache: Dict[str, Any] = {}
_ALERT_CACHE_TTL = 120


def _make_alert(
    alert_type: str,
    ticker: str,
    priority: str,
    title: str,
    message: str,
    source: str = "",
    action_url: str = "",
    data: Optional[dict] = None,
) -> dict:
    return {
        "id": str(uuid.uuid4())[:12],
        "type": alert_type,
        "ticker": ticker,
        "priority": priority,
        "title": title,
        "message": message,
        "source": source,
        "action_url": action_url,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": data or {},
    }


def _detect_insider_cluster_alerts(tickers: List[str]) -> List[dict]:
    alerts = []
    try:
        from prediction.flow_tracker import get_smart_money_flow
        for ticker in tickers:
            try:
                flow = get_smart_money_flow(ticker)
                clusters = flow.get("insider_clusters", [])
                if not clusters:
                    continue
                for cluster in clusters:
                    num_insiders = cluster.get("num_insiders", 0)
                    if num_insiders >= 3:
                        priority = "critical"
                    elif num_insiders >= 2:
                        priority = "high"
                    else:
                        continue

                    total_value = cluster.get("total_value", 0)
                    names = cluster.get("insider_names", [])
                    has_csuite = cluster.get("has_c_suite", False)
                    start = cluster.get("start_date", "")
                    end = cluster.get("end_date", "")

                    value_str = f"${total_value / 1_000_000:.1f}M" if total_value >= 1_000_000 else f"${total_value:,.0f}"
                    csuite_tag = " (includes C-Suite)" if has_csuite else ""

                    alerts.append(_make_alert(
                        alert_type="insider_cluster",
                        ticker=ticker,
                        priority=priority,
                        title=f"{ticker} insider cluster: {num_insiders} insiders buying",
                        message=f"{num_insiders} insiders purchased {value_str} worth of {ticker} between {start} and {end}{csuite_tag}. Insiders: {', '.join(names[:3])}",
                        source="SEC Form 4",
                        action_url=f"/market?ticker={ticker}&tab=smart-money",
                        data={
                            "num_insiders": num_insiders,
                            "total_value": total_value,
                            "has_c_suite": has_csuite,
                            "insider_names": names[:5],
                            "start_date": start,
                            "end_date": end,
                        },
                    ))
            except Exception as e:
                logger.debug(f"Insider cluster alert error for {ticker}: {e}")
    except Exception as e:
        logger.warning(f"Insider cluster alerts failed: {e}")
    return alerts


def _detect_unusual_options_alerts(tickers: List[str]) -> List[dict]:
    alerts = []
    try:
        from prediction.options_flow import get_options_flow
        for ticker in tickers:
            try:
                flow = get_options_flow(ticker)
                unusual = flow.get("unusual_activity", [])
                if not unusual:
                    continue

                for activity in unusual[:3]:
                    vol = activity.get("volume", 0)
                    oi = activity.get("open_interest", 1)
                    vol_oi = activity.get("vol_oi_ratio", 0)
                    premium = activity.get("premium_total", 0)
                    sentiment = activity.get("sentiment", "neutral")
                    trade_type = activity.get("trade_type", "standard")
                    strike = activity.get("strike", 0)
                    expiry = activity.get("expiry", "")
                    contract_type = activity.get("type", "call")

                    if vol_oi < 10.0 and premium < 500_000:
                        continue

                    if vol_oi >= 10.0 or premium >= 5_000_000:
                        priority = "critical"
                    elif vol_oi >= 5.0 or premium >= 1_000_000:
                        priority = "high"
                    else:
                        priority = "medium"

                    premium_str = f"${premium / 1_000_000:.1f}M" if premium >= 1_000_000 else f"${premium:,.0f}"
                    direction = "Bullish" if sentiment == "bullish" else "Bearish"

                    alerts.append(_make_alert(
                        alert_type="unusual_options",
                        ticker=ticker,
                        priority=priority,
                        title=f"{ticker} unusual {contract_type} activity ({trade_type})",
                        message=f"{direction} {trade_type} on {ticker} ${strike} {contract_type} exp {expiry}. Vol/OI: {vol_oi:.1f}x, Premium: {premium_str}, {vol:,} contracts traded",
                        source="Options Desk",
                        action_url=f"/market?ticker={ticker}&tab=options",
                        data={
                            "volume": vol,
                            "open_interest": oi,
                            "vol_oi_ratio": vol_oi,
                            "premium": premium,
                            "sentiment": sentiment,
                            "trade_type": trade_type,
                            "strike": strike,
                            "expiry": expiry,
                            "contract_type": contract_type,
                        },
                    ))
            except Exception as e:
                logger.debug(f"Options alert error for {ticker}: {e}")
    except Exception as e:
        logger.warning(f"Unusual options alerts failed: {e}")
    return alerts


def _detect_earnings_surprise_alerts(tickers: List[str]) -> List[dict]:
    alerts = []
    try:
        from prediction.news_feed import get_earnings_calendar_sync
        earnings = get_earnings_calendar_sync(tickers)
        for item in earnings:
            ticker = item.get("ticker", "")
            surprises = item.get("surprise_history", [])
            if not surprises:
                continue

            latest = surprises[0]
            surprise_pct = latest.get("surprise_pct", 0)
            actual = latest.get("actual")
            estimate = latest.get("estimate")
            beat = latest.get("beat", False)

            if abs(surprise_pct) < 10:
                continue

            if abs(surprise_pct) > 30:
                priority = "critical"
            elif abs(surprise_pct) > 15:
                priority = "high"
            else:
                priority = "medium"

            direction = "beat" if beat else "missed"
            alerts.append(_make_alert(
                alert_type="earnings_surprise",
                ticker=ticker,
                priority=priority,
                title=f"{ticker} earnings {direction} by {abs(surprise_pct):.1f}%",
                message=f"{ticker} reported EPS ${actual:.2f} vs estimate ${estimate:.2f} ({surprise_pct:+.1f}% surprise)",
                source="Earnings Report",
                action_url=f"/market?ticker={ticker}&tab=fundamentals",
                data={
                    "actual": actual,
                    "estimate": estimate,
                    "surprise_pct": surprise_pct,
                    "beat": beat,
                    "date": latest.get("date", ""),
                },
            ))

            days_until = item.get("days_until")
            if days_until is not None and 0 <= days_until <= 5:
                ep = "critical" if days_until <= 1 else "high" if days_until <= 3 else "medium"
                ed = item.get("earnings_date", "")
                alerts.append(_make_alert(
                    alert_type="earnings_approaching",
                    ticker=ticker,
                    priority=ep,
                    title=f"{ticker} earnings in {days_until} day{'s' if days_until != 1 else ''}",
                    message=f"{ticker} reports earnings on {ed}. Historical beat rate: {item.get('beat_rate', 'N/A')}%",
                    source="Earnings Calendar",
                    action_url=f"/market?ticker={ticker}&tab=fundamentals",
                    data={
                        "days_until": days_until,
                        "earnings_date": ed,
                        "beat_rate": item.get("beat_rate"),
                    },
                ))
    except Exception as e:
        logger.warning(f"Earnings surprise alerts failed: {e}")
    return alerts


def _detect_price_breakout_alerts(tickers: List[str]) -> List[dict]:
    alerts = []
    try:
        from prediction.features import fetch_ohlcv
        import numpy as np

        for ticker in tickers:
            try:
                df = fetch_ohlcv(ticker, days=365)
                if df is None or len(df) < 10:
                    continue

                close = float(df["close"].iloc[-1])
                prev_close = float(df["close"].iloc[-2])
                close_vals = df["close"].values
                high_52w = float(np.nanmax(close_vals))  # type: ignore
                low_52w = float(np.nanmin(close_vals))  # type: ignore

                if high_52w > 0 and close >= high_52w:
                    alerts.append(_make_alert(
                        alert_type="breakout_52w_high",
                        ticker=ticker,
                        priority="critical",
                        title=f"{ticker} breaking 52-week high!",
                        message=f"{ticker} at ${close:.2f} has broken its 52-week high of ${high_52w:.2f}. Strong bullish momentum signal.",
                        source="Price Action",
                        action_url=f"/market?ticker={ticker}",
                        data={"price": round(close, 2), "high_52w": round(high_52w, 2)},
                    ))
                elif high_52w > 0 and (high_52w - close) / high_52w <= 0.02:
                    alerts.append(_make_alert(
                        alert_type="near_52w_high",
                        ticker=ticker,
                        priority="high",
                        title=f"{ticker} near 52-week high",
                        message=f"{ticker} at ${close:.2f}, within 2% of 52-week high ${high_52w:.2f}",
                        source="Price Action",
                        action_url=f"/market?ticker={ticker}",
                        data={"price": round(close, 2), "high_52w": round(high_52w, 2)},
                    ))

                if low_52w > 0 and close <= low_52w:
                    alerts.append(_make_alert(
                        alert_type="breakout_52w_low",
                        ticker=ticker,
                        priority="critical",
                        title=f"{ticker} breaking 52-week low!",
                        message=f"{ticker} at ${close:.2f} has broken its 52-week low of ${low_52w:.2f}. Strong bearish signal.",
                        source="Price Action",
                        action_url=f"/market?ticker={ticker}",
                        data={"price": round(close, 2), "low_52w": round(low_52w, 2)},
                    ))
                elif low_52w > 0 and (close - low_52w) / low_52w <= 0.02:
                    alerts.append(_make_alert(
                        alert_type="near_52w_low",
                        ticker=ticker,
                        priority="high",
                        title=f"{ticker} near 52-week low",
                        message=f"{ticker} at ${close:.2f}, within 2% of 52-week low ${low_52w:.2f}",
                        source="Price Action",
                        action_url=f"/market?ticker={ticker}",
                        data={"price": round(close, 2), "low_52w": round(low_52w, 2)},
                    ))

                change_pct = (close - prev_close) / prev_close * 100 if prev_close > 0 else 0
                if abs(change_pct) > 3:
                    direction = "surged" if change_pct > 0 else "dropped"
                    pri = "critical" if abs(change_pct) > 7 else "high" if abs(change_pct) > 5 else "medium"
                    alerts.append(_make_alert(
                        alert_type="price_move",
                        ticker=ticker,
                        priority=pri,
                        title=f"{ticker} {direction} {abs(change_pct):.1f}%",
                        message=f"{ticker} moved {change_pct:+.2f}% from ${prev_close:.2f} to ${close:.2f}",
                        source="Price Action",
                        action_url=f"/market?ticker={ticker}",
                        data={"change_pct": round(change_pct, 2), "price": round(close, 2)},
                    ))
            except Exception as e:
                logger.debug(f"Price breakout alert error for {ticker}: {e}")
    except Exception as e:
        logger.warning(f"Price breakout alerts failed: {e}")
    return alerts


def _detect_congressional_trade_alerts(tickers: List[str]) -> List[dict]:
    alerts = []
    try:
        from prediction.flow_tracker import _fetch_congressional_trades
        for ticker in tickers:
            try:
                trades = _fetch_congressional_trades(ticker)
                if not trades:
                    continue
                for trade in trades[:2]:
                    name = trade.get("name", "Unknown")
                    chamber = trade.get("chamber", "")
                    tx_type = trade.get("transaction_type", "")
                    tx_date = trade.get("transaction_date", "")
                    amount_from = trade.get("amount_from", 0)
                    amount_to = trade.get("amount_to", 0)

                    if amount_to and amount_to > 0:
                        amount_str = f"${amount_from:,} - ${amount_to:,}" if amount_from else f"up to ${amount_to:,}"
                    elif amount_from and amount_from > 0:
                        amount_str = f"${amount_from:,}+"
                    else:
                        amount_str = "undisclosed amount"

                    alerts.append(_make_alert(
                        alert_type="congressional_trade",
                        ticker=ticker,
                        priority="high",
                        title=f"Congressional trade: {name} {tx_type.lower()} {ticker}",
                        message=f"{chamber} member {name} {tx_type.lower()} {ticker} ({amount_str}) on {tx_date}",
                        source="Congressional Trading (Finnhub)",
                        action_url=f"/market?ticker={ticker}&tab=smart-money",
                        data={
                            "name": name,
                            "chamber": chamber,
                            "transaction_type": tx_type,
                            "date": tx_date,
                            "amount_from": amount_from,
                            "amount_to": amount_to,
                        },
                    ))
            except Exception as e:
                logger.debug(f"Congressional trade alert error for {ticker}: {e}")
    except Exception as e:
        logger.warning(f"Congressional trade alerts failed: {e}")
    return alerts


def _detect_analyst_alerts(tickers: List[str]) -> List[dict]:
    alerts = []
    try:
        from prediction.news_feed import fetch_financial_news
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    news = pool.submit(_sync_fetch_news, tickers).result(timeout=15)
            else:
                news = loop.run_until_complete(fetch_financial_news(tickers, max_items=30))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                news = loop.run_until_complete(fetch_financial_news(tickers, max_items=30))
            finally:
                loop.close()

        analyst_keywords = ["upgrade", "downgrade", "price target", "initiates", "coverage",
                            "overweight", "underweight", "outperform", "underperform", "buy rating",
                            "sell rating", "hold rating", "raises target", "lowers target"]

        for item in news:
            headline = (item.get("headline", "") or "").lower()
            if not any(kw in headline for kw in analyst_keywords):
                continue

            matched_tickers = item.get("tickers", [])
            ticker = matched_tickers[0] if matched_tickers else ""
            if not ticker:
                continue

            is_upgrade = any(w in headline for w in ["upgrade", "overweight", "outperform", "raises target", "buy rating"])
            is_downgrade = any(w in headline for w in ["downgrade", "underweight", "underperform", "lowers target", "sell rating"])

            if is_upgrade:
                action = "upgraded"
                priority = "high"
            elif is_downgrade:
                action = "downgraded"
                priority = "high"
            else:
                action = "analyst coverage"
                priority = "medium"

            alerts.append(_make_alert(
                alert_type="analyst_action",
                ticker=ticker,
                priority=priority,
                title=f"{ticker} {action}",
                message=item.get("headline", ""),
                source=item.get("source", "Analyst"),
                action_url=f"/market?ticker={ticker}&tab=fundamentals",
                data={
                    "headline": item.get("headline", ""),
                    "source": item.get("source", ""),
                    "url": item.get("url", ""),
                    "action": action,
                },
            ))
    except Exception as e:
        logger.warning(f"Analyst alerts failed: {e}")
    return alerts


def _sync_fetch_news(tickers: List[str]):
    import asyncio
    from prediction.news_feed import fetch_financial_news
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(fetch_financial_news(tickers, max_items=30))
    finally:
        loop.close()


def _detect_score_threshold_alerts(tickers: List[str]) -> List[dict]:
    alerts = []
    try:
        from prediction.composite_intel import _score_cache
        for ticker in tickers:
            cached = _score_cache.get(ticker.upper())
            if not cached:
                continue
            score = cached.get("composite_score", 50)
            direction = cached.get("direction", "neutral")
            label = cached.get("signal_label", "Neutral")

            if score >= 75:
                alerts.append(_make_alert(
                    alert_type="score_bullish",
                    ticker=ticker,
                    priority="high" if score >= 85 else "medium",
                    title=f"{ticker} Exponenta Score: {score:.0f} ({label})",
                    message=f"{ticker} composite intelligence score crossed bullish threshold at {score:.0f}/100. Direction: {direction}",
                    source="Exponenta Score",
                    action_url=f"/market?ticker={ticker}",
                    data={"score": score, "direction": direction, "label": label},
                ))
            elif score <= 25:
                alerts.append(_make_alert(
                    alert_type="score_bearish",
                    ticker=ticker,
                    priority="high" if score <= 15 else "medium",
                    title=f"{ticker} Exponenta Score: {score:.0f} ({label})",
                    message=f"{ticker} composite intelligence score crossed bearish threshold at {score:.0f}/100. Direction: {direction}",
                    source="Exponenta Score",
                    action_url=f"/market?ticker={ticker}",
                    data={"score": score, "direction": direction, "label": label},
                ))
    except Exception as e:
        logger.warning(f"Score threshold alerts failed: {e}")
    return alerts


def _detect_volume_alerts(tickers: List[str]) -> List[dict]:
    alerts = []
    try:
        from prediction.features import fetch_ohlcv
        import numpy as np

        for ticker in tickers:
            try:
                df = fetch_ohlcv(ticker, days=60)
                if df is None or len(df) < 21:
                    continue

                current_vol = float(df["volume"].iloc[-1])
                avg_vol_20 = float(df["volume"].iloc[-21:-1].mean())

                if avg_vol_20 > 0 and current_vol > 3 * avg_vol_20:
                    ratio = current_vol / avg_vol_20
                    priority = "critical" if ratio > 8 else "high" if ratio > 5 else "medium"
                    alerts.append(_make_alert(
                        alert_type="unusual_volume",
                        ticker=ticker,
                        priority=priority,
                        title=f"{ticker} unusual volume ({ratio:.1f}x avg)",
                        message=f"{ticker} trading {ratio:.1f}x its 20-day average volume ({current_vol:,.0f} vs avg {avg_vol_20:,.0f})",
                        source="Volume Analysis",
                        action_url=f"/market?ticker={ticker}",
                        data={"current_volume": current_vol, "avg_volume": round(avg_vol_20), "ratio": round(ratio, 2)},
                    ))
            except Exception as e:
                logger.debug(f"Volume alert error for {ticker}: {e}")
    except Exception as e:
        logger.warning(f"Volume alerts failed: {e}")
    return alerts


def get_all_alerts(tickers: List[str], risk_tolerance: str = "moderate") -> dict:
    cache_key = f"alerts_{'_'.join(sorted(tickers))}_{risk_tolerance}"
    cached = _alert_cache.get(cache_key)
    if cached and (time.time() - cached.get("_ts", 0)) < _ALERT_CACHE_TTL:
        return {k: v for k, v in cached.items() if k != "_ts"}

    all_alerts: List[dict] = []

    try:
        all_alerts.extend(_detect_price_breakout_alerts(tickers))
    except Exception as e:
        logger.error(f"Price breakout alerts failed: {e}")

    try:
        all_alerts.extend(_detect_volume_alerts(tickers))
    except Exception as e:
        logger.error(f"Volume alerts failed: {e}")

    try:
        all_alerts.extend(_detect_insider_cluster_alerts(tickers))
    except Exception as e:
        logger.error(f"Insider cluster alerts failed: {e}")

    try:
        all_alerts.extend(_detect_unusual_options_alerts(tickers))
    except Exception as e:
        logger.error(f"Options alerts failed: {e}")

    try:
        all_alerts.extend(_detect_earnings_surprise_alerts(tickers))
    except Exception as e:
        logger.error(f"Earnings alerts failed: {e}")

    try:
        all_alerts.extend(_detect_congressional_trade_alerts(tickers))
    except Exception as e:
        logger.error(f"Congressional trade alerts failed: {e}")

    try:
        all_alerts.extend(_detect_analyst_alerts(tickers))
    except Exception as e:
        logger.error(f"Analyst alerts failed: {e}")

    try:
        all_alerts.extend(_detect_score_threshold_alerts(tickers))
    except Exception as e:
        logger.error(f"Score threshold alerts failed: {e}")

    all_alerts.sort(key=lambda a: PRIORITY_ORDER.get(a["priority"], 3))

    critical_count = sum(1 for a in all_alerts if a["priority"] == "critical")
    high_count = sum(1 for a in all_alerts if a["priority"] == "high")

    alert_types = {}
    for a in all_alerts:
        t = a["type"]
        alert_types[t] = alert_types.get(t, 0) + 1

    result = {
        "alerts": all_alerts,
        "count": len(all_alerts),
        "critical_count": critical_count,
        "high_count": high_count,
        "alert_types": alert_types,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    _alert_cache[cache_key] = {**result, "_ts": time.time()}
    return result
