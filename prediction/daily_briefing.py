import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
import numpy as np

logger = logging.getLogger(__name__)

XAI_API_KEY = os.environ.get("XAI_API_KEY", "")

_briefing_cache: Dict[str, tuple] = {}
BRIEFING_CACHE_TTL = 1800


def _get_portfolio_summary() -> Dict[str, Any]:
    try:
        portfolio_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "user_portfolio.json")
        if not os.path.exists(portfolio_file):
            return {"has_portfolio": False, "holdings": [], "total_value": 0}

        with open(portfolio_file, "r") as f:
            holdings = json.load(f)

        if not holdings:
            return {"has_portfolio": False, "holdings": [], "total_value": 0}

        import data.market_data as market_data

        enriched = []
        total_value = 0
        total_pnl = 0
        for h in holdings:
            q = market_data._fetch_quote(h["ticker"])
            current_price = q["price"] if q else h["avg_price"]
            market_value = current_price * h["quantity"]
            cost_basis = h["avg_price"] * h["quantity"]
            pnl = market_value - cost_basis
            pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0
            change_pct = q["change_pct"] if q else 0
            total_value += market_value
            total_pnl += pnl
            enriched.append({
                "ticker": h["ticker"],
                "quantity": h["quantity"],
                "avg_price": h["avg_price"],
                "current_price": round(current_price, 2),
                "market_value": round(market_value, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "change_pct": round(change_pct, 2),
            })

        for e in enriched:
            e["weight"] = round(e["market_value"] / total_value, 4) if total_value > 0 else 0

        top_movers = sorted(enriched, key=lambda x: abs(x["change_pct"]), reverse=True)[:3]
        best_performer = max(enriched, key=lambda x: x["pnl_pct"]) if enriched else None
        worst_performer = min(enriched, key=lambda x: x["pnl_pct"]) if enriched else None

        return {
            "has_portfolio": True,
            "holdings": enriched,
            "total_value": round(total_value, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl / (total_value - total_pnl) * 100, 2) if (total_value - total_pnl) > 0 else 0,
            "num_holdings": len(enriched),
            "top_movers": top_movers,
            "best_performer": best_performer,
            "worst_performer": worst_performer,
        }
    except Exception as e:
        logger.error(f"Portfolio summary error: {e}")
        return {"has_portfolio": False, "holdings": [], "total_value": 0}


def _get_sector_rotation() -> Dict[str, Any]:
    try:
        from prediction.market_intelligence import MarketIntelligence
        mi = MarketIntelligence()
        return mi.sector_rotation_analysis()
    except Exception as e:
        logger.error(f"Sector rotation error: {e}")
        return {"sectors": [], "rotation_signal": "unknown", "leaders": [], "laggards": []}


def _get_active_alerts(tickers: List[str], risk_tolerance: str) -> Dict[str, Any]:
    try:
        from prediction.smart_alerts import get_all_alerts
        return get_all_alerts(tickers, risk_tolerance=risk_tolerance)
    except Exception as e:
        logger.error(f"Alerts error: {e}")
        return {"alerts": [], "count": 0, "critical_count": 0}


def _get_earnings_upcoming(tickers: List[str]) -> List[Dict[str, Any]]:
    try:
        from prediction.news_feed import get_earnings_calendar_sync
        earnings = get_earnings_calendar_sync(tickers)
        return [e for e in earnings if e.get("is_upcoming", False)]
    except Exception as e:
        logger.error(f"Earnings error: {e}")
        return []


def _get_market_regime(tickers: List[str]) -> Dict[str, Any]:
    try:
        from prediction.market_intelligence import MarketIntelligence
        mi = MarketIntelligence()
        breadth = mi.market_breadth(tickers)
        volatility = mi.volatility_regime()
        return {
            "breadth": breadth,
            "volatility": volatility,
        }
    except Exception as e:
        logger.error(f"Market regime error: {e}")
        return {
            "breadth": {"breadth_score": 50, "interpretation": "Unknown"},
            "volatility": {"regime": "unknown", "volatility_pct": 0, "trend": "stable"},
        }


def _build_briefing_context(
    portfolio: Dict[str, Any],
    sector_rotation: Dict[str, Any],
    alerts: Dict[str, Any],
    earnings: List[Dict[str, Any]],
    market_regime: Dict[str, Any],
) -> str:
    parts = []

    parts.append("=== PORTFOLIO STATUS ===")
    if portfolio.get("has_portfolio"):
        parts.append(f"Total Value: ${portfolio['total_value']:,.2f}")
        parts.append(f"Total P&L: ${portfolio.get('total_pnl', 0):,.2f} ({portfolio.get('total_pnl_pct', 0):+.2f}%)")
        parts.append(f"Number of Holdings: {portfolio.get('num_holdings', 0)}")
        if portfolio.get("top_movers"):
            parts.append("Today's Top Movers in Portfolio:")
            for m in portfolio["top_movers"]:
                parts.append(f"  {m['ticker']}: {m['change_pct']:+.2f}% (${m['current_price']:.2f})")
        if portfolio.get("best_performer"):
            bp = portfolio["best_performer"]
            parts.append(f"Best Performer: {bp['ticker']} ({bp['pnl_pct']:+.2f}%)")
        if portfolio.get("worst_performer"):
            wp = portfolio["worst_performer"]
            parts.append(f"Worst Performer: {wp['ticker']} ({wp['pnl_pct']:+.2f}%)")
    else:
        parts.append("No portfolio configured")

    parts.append("\n=== SECTOR ROTATION ===")
    parts.append(f"Signal: {sector_rotation.get('rotation_signal', 'unknown')}")
    if sector_rotation.get("leaders"):
        parts.append("Leaders: " + ", ".join(f"{s['name']} ({s['return_1m']:+.1f}%)" for s in sector_rotation["leaders"]))
    if sector_rotation.get("laggards"):
        parts.append("Laggards: " + ", ".join(f"{s['name']} ({s['return_1m']:+.1f}%)" for s in sector_rotation["laggards"]))

    parts.append("\n=== MARKET REGIME ===")
    breadth = market_regime.get("breadth", {})
    vol = market_regime.get("volatility", {})
    parts.append(f"Breadth Score: {breadth.get('breadth_score', 50)}/100 — {breadth.get('interpretation', 'N/A')}")
    parts.append(f"Volatility: {vol.get('regime', 'unknown')} ({vol.get('volatility_pct', 0):.1f}% annualized, trend: {vol.get('trend', 'stable')})")

    parts.append("\n=== ACTIVE ALERTS ===")
    alert_list = alerts.get("alerts", [])
    if alert_list:
        parts.append(f"Total: {alerts.get('count', 0)} alerts, {alerts.get('critical_count', 0)} critical")
        for a in alert_list[:8]:
            parts.append(f"  [{a['priority'].upper()}] {a['title']}: {a['message']}")
    else:
        parts.append("No active alerts")

    parts.append("\n=== UPCOMING EARNINGS ===")
    if earnings:
        for e in earnings[:5]:
            parts.append(f"  {e['ticker']}: {e['earnings_date']} ({e['days_until']} days)")
    else:
        parts.append("No upcoming earnings in watchlist")

    return "\n".join(parts)


async def _generate_ai_briefing(context: str) -> Dict[str, Any]:
    if not XAI_API_KEY:
        return _generate_fallback_briefing(context)

    try:
        prompt = (
            "You are an elite portfolio intelligence analyst. Based on the data below, generate a comprehensive "
            "daily intelligence briefing. Return a JSON object with these exact keys:\n"
            "{\n"
            '  "executive_summary": "2-3 sentence overview of portfolio status and market conditions",\n'
            '  "portfolio_highlights": ["list of 2-4 key portfolio observations"],\n'
            '  "market_signals": ["list of 2-4 market signals or trends to watch"],\n'
            '  "risk_warnings": ["list of 1-3 risk factors or warnings, empty if none"],\n'
            '  "opportunities": ["list of 1-3 actionable opportunities"],\n'
            '  "action_items": [\n'
            '    {"action": "what to do", "priority": "high|medium|low", "rationale": "why"}\n'
            "  ]\n"
            "}\n"
            "Be specific, data-driven, and actionable. Reference actual tickers and numbers. "
            "Return ONLY the JSON object, no other text.\n\n"
            f"DATA:\n{context}"
        )

        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "grok-3-fast",
                    "messages": [
                        {"role": "system", "content": "You are a senior portfolio analyst generating a daily intelligence briefing. Return only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.4,
                },
            )

        if resp.status_code != 200:
            logger.warning(f"Briefing AI call returned {resp.status_code}")
            return _generate_fallback_briefing(context)

        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            return _generate_fallback_briefing(context)

        briefing = json.loads(json_match.group())

        required_keys = ["executive_summary", "portfolio_highlights", "market_signals", "risk_warnings", "opportunities", "action_items"]
        for key in required_keys:
            if key not in briefing:
                briefing[key] = [] if key != "executive_summary" else "Briefing data available but incomplete."

        return briefing

    except Exception as e:
        logger.error(f"AI briefing generation error: {e}")
        return _generate_fallback_briefing(context)


def _generate_fallback_briefing(context: str) -> Dict[str, Any]:
    portfolio_highlights = []
    market_signals = []
    risk_warnings = []
    opportunities = []
    action_items = []
    executive_summary = "Market analysis is available. Review the details below for your portfolio status and key signals."

    lines = context.split("\n")
    for line in lines:
        line = line.strip()
        if not line or line.startswith("==="):
            continue

        if "Total Value:" in line:
            portfolio_highlights.append(line)
        elif "Total P&L:" in line:
            portfolio_highlights.append(line)
        elif "Best Performer:" in line:
            portfolio_highlights.append(line)
        elif "Worst Performer:" in line:
            portfolio_highlights.append(line)

        if "Signal:" in line and "rotation" not in line.lower():
            market_signals.append(line)
        elif "Breadth Score:" in line:
            market_signals.append(line)
        elif "Volatility:" in line:
            market_signals.append(line)
        elif "Leaders:" in line:
            market_signals.append(f"Sector leaders: {line.replace('Leaders: ', '')}")

        if "[CRITICAL]" in line or "[HIGH]" in line:
            risk_warnings.append(line.split("] ", 1)[-1] if "] " in line else line)

        if "STRONG_BUY" in line or "opportunity" in line.lower():
            opportunities.append(line.split("] ", 1)[-1] if "] " in line else line)

    if not portfolio_highlights:
        portfolio_highlights = ["Portfolio data is being processed"]
    if not market_signals:
        market_signals = ["Market conditions are being analyzed"]

    for w in risk_warnings[:2]:
        action_items.append({
            "action": f"Review: {w[:80]}",
            "priority": "high",
            "rationale": "Alert triggered requiring attention",
        })

    if not action_items:
        action_items.append({
            "action": "Review portfolio allocation and rebalance if needed",
            "priority": "medium",
            "rationale": "Regular portfolio maintenance",
        })

    return {
        "executive_summary": executive_summary,
        "portfolio_highlights": portfolio_highlights[:4],
        "market_signals": market_signals[:4],
        "risk_warnings": risk_warnings[:3],
        "opportunities": opportunities[:3],
        "action_items": action_items[:4],
    }


async def generate_daily_briefing() -> Dict[str, Any]:
    now = time.time()
    cache_key = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")
    if cache_key in _briefing_cache:
        cached_time, cached_data = _briefing_cache[cache_key]
        if now - cached_time < BRIEFING_CACHE_TTL:
            return cached_data

    from user_settings import load_settings
    settings = load_settings()
    watchlist = settings.get("watchlist", ["AAPL", "TSLA", "NVDA", "GOOGL", "MSFT", "AMZN"])
    risk_tolerance = settings.get("risk_tolerance", "moderate")

    portfolio = _get_portfolio_summary()
    sector_rotation = _get_sector_rotation()
    alerts = _get_active_alerts(watchlist, risk_tolerance)
    earnings = _get_earnings_upcoming(watchlist)
    market_regime = _get_market_regime(watchlist)

    context = _build_briefing_context(portfolio, sector_rotation, alerts, earnings, market_regime)

    ai_briefing = await _generate_ai_briefing(context)

    result = {
        "briefing": ai_briefing,
        "raw_data": {
            "portfolio_summary": {
                "has_portfolio": portfolio.get("has_portfolio", False),
                "total_value": portfolio.get("total_value", 0),
                "total_pnl": portfolio.get("total_pnl", 0),
                "total_pnl_pct": portfolio.get("total_pnl_pct", 0),
                "num_holdings": portfolio.get("num_holdings", 0),
                "top_movers": portfolio.get("top_movers", []),
            },
            "sector_rotation": {
                "signal": sector_rotation.get("rotation_signal", "unknown"),
                "leaders": sector_rotation.get("leaders", []),
                "laggards": sector_rotation.get("laggards", []),
            },
            "alerts_summary": {
                "total": alerts.get("count", 0),
                "critical": alerts.get("critical_count", 0),
                "top_alerts": alerts.get("alerts", [])[:5],
            },
            "upcoming_earnings": earnings[:5],
            "market_regime": {
                "breadth_score": market_regime.get("breadth", {}).get("breadth_score", 50),
                "breadth_interpretation": market_regime.get("breadth", {}).get("interpretation", "N/A"),
                "volatility_regime": market_regime.get("volatility", {}).get("regime", "unknown"),
                "volatility_pct": market_regime.get("volatility", {}).get("volatility_pct", 0),
                "volatility_trend": market_regime.get("volatility", {}).get("trend", "stable"),
            },
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "watchlist": watchlist,
    }

    _briefing_cache[cache_key] = (now, result)
    return result
