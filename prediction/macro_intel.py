import asyncio
import hashlib
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from prediction.api_utils import resilient_get

from prediction.cache_manager import SmartCache

logger = logging.getLogger(__name__)

ALPHA_VANTAGE_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "")
FMP_API_KEY = os.environ.get("FMP_API_KEY", "")

AV_BASE = "https://www.alphavantage.co/query"
FMP_BASE = "https://financialmodelingprep.com/stable"

_macro_cache = SmartCache("macro_intel", max_size=500, default_ttl=86400)
MACRO_CACHE_TTL = 86400
UNAVAILABLE_CACHE_TTL = 300
ECON_CALENDAR_CACHE_TTL = 3600


def _cache_get(key: str, ttl: int = MACRO_CACHE_TTL):
    return _macro_cache.get(key, ttl)


def _cache_set(key: str, data):
    _macro_cache.set(key, data)


_av_last_call = 0.0

async def _av_fetch(function: str, extra_params: dict = None) -> Optional[dict]:
    global _av_last_call
    if not ALPHA_VANTAGE_KEY:
        return None
    elapsed = time.time() - _av_last_call
    if elapsed < 1.2:
        await asyncio.sleep(1.2 - elapsed)
    params = {"function": function, "apikey": ALPHA_VANTAGE_KEY}
    if extra_params:
        params.update(extra_params)
    try:
        _av_last_call = time.time()
        resp = await asyncio.to_thread(resilient_get, AV_BASE, params, 2, 15, "alpha_vantage")
        if resp and resp.status_code == 200:
            data = resp.json()
            if "Error Message" in data or "Note" in data:
                logger.warning(f"AV API issue for {function}: {data.get('Note', data.get('Error Message', ''))}")
                return None
            if "Information" in data:
                logger.info(f"AV rate limited for {function}")
                return None
            return data
    except Exception as e:
        logger.error(f"AV fetch error {function}: {e}")
    return None


async def _fmp_fetch(endpoint: str, params: dict = None) -> Optional[Any]:
    if not FMP_API_KEY:
        return None
    url = f"{FMP_BASE}/{endpoint}"
    query = {"apikey": FMP_API_KEY}
    if params:
        query.update(params)
    try:
        resp = await asyncio.to_thread(resilient_get, url, query, 2, 15, "fmp")
        if resp is None:
            return None
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict) and "Error Message" in data:
                logger.warning(f"FMP error for {endpoint}: {data.get('Error Message', '')[:100]}")
                return None
            return data
        if resp.status_code in (402, 403):
            logger.info(f"FMP {endpoint} returned {resp.status_code} — not available on current plan")
            return None
    except Exception as e:
        logger.error(f"FMP fetch error {endpoint}: {e}")
    return None


def _parse_av_series(data: dict, key: str = "data") -> List[Dict]:
    if not data:
        return []
    entries = data.get(key, [])
    result = []
    for entry in entries[:60]:
        val = entry.get("value")
        date = entry.get("date", "")
        if val and val != "." and date:
            try:
                result.append({"date": date, "value": float(val)})
            except (ValueError, TypeError):
                pass
    return result


async def fetch_gdp() -> Dict:
    cached = _cache_get("gdp")
    if cached:
        return cached

    data = await _av_fetch("REAL_GDP", {"interval": "quarterly"})
    series = _parse_av_series(data)
    if not series:
        result = _unavailable_indicator("Real GDP", "Billion USD", "Quarterly")
        _macro_cache.set("gdp", result, ttl=UNAVAILABLE_CACHE_TTL)
        return result

    current = series[0]["value"] if series else 0
    prev = series[1]["value"] if len(series) > 1 else current
    change = round(current - prev, 2)
    trend = "growing" if change > 0 else "contracting" if change < 0 else "flat"

    result = {
        "name": "Real GDP",
        "current": current,
        "previous": prev,
        "change": change,
        "unit": "Billion USD",
        "trend": trend,
        "frequency": "Quarterly",
        "history": series[:12],
        "status": "live",
        "data_freshness": datetime.now(timezone.utc).isoformat(),
    }
    _cache_set("gdp", result)
    return result


async def fetch_cpi() -> Dict:
    cached = _cache_get("cpi")
    if cached:
        return cached

    data = await _av_fetch("CPI", {"interval": "monthly"})
    series = _parse_av_series(data)
    if not series:
        result = _unavailable_indicator("CPI", "Index", "Monthly")
        result["change_mom"] = 0
        result["change_yoy"] = 0
        _macro_cache.set("cpi", result, ttl=UNAVAILABLE_CACHE_TTL)
        return result

    current = series[0]["value"] if series else 0
    prev = series[1]["value"] if len(series) > 1 else current
    yoy_prev = series[12]["value"] if len(series) > 12 else current
    change_mom = round(((current / prev) - 1) * 100, 2) if prev else 0
    change_yoy = round(((current / yoy_prev) - 1) * 100, 2) if yoy_prev else 0

    result = {
        "name": "CPI",
        "current": current,
        "previous": prev,
        "change_mom": change_mom,
        "change_yoy": change_yoy,
        "unit": "Index",
        "trend": "rising" if change_yoy > 2.5 else "stable" if change_yoy > 1.5 else "falling",
        "frequency": "Monthly",
        "history": series[:24],
        "status": "live",
        "data_freshness": datetime.now(timezone.utc).isoformat(),
    }
    _cache_set("cpi", result)
    return result


async def fetch_inflation() -> Dict:
    cached = _cache_get("inflation")
    if cached:
        return cached

    data = await _av_fetch("INFLATION")
    series = _parse_av_series(data)
    if not series:
        result = _unavailable_indicator("Inflation Rate", "%", "Annual")
        _macro_cache.set("inflation", result, ttl=UNAVAILABLE_CACHE_TTL)
        return result

    current = series[0]["value"] if series else 0
    prev = series[1]["value"] if len(series) > 1 else current

    result = {
        "name": "Inflation Rate",
        "current": current,
        "previous": prev,
        "change": round(current - prev, 2),
        "unit": "%",
        "trend": "rising" if current > prev else "moderating" if current < prev else "stable",
        "frequency": "Annual",
        "history": series[:10],
        "status": "live",
        "data_freshness": datetime.now(timezone.utc).isoformat(),
    }
    _cache_set("inflation", result)
    return result


async def fetch_fed_rate() -> Dict:
    cached = _cache_get("fed_rate")
    if cached:
        return cached

    data = await _av_fetch("FEDERAL_FUNDS_RATE", {"interval": "monthly"})
    series = _parse_av_series(data)
    if not series:
        try:
            yf_3m = await _yf_treasury_yield("3month")
            if yf_3m and len(yf_3m) >= 2:
                series = yf_3m
        except Exception:
            pass
    if not series:
        result = _unavailable_indicator("Federal Funds Rate", "%", "Monthly")
        _macro_cache.set("fed_rate", result, ttl=UNAVAILABLE_CACHE_TTL)
        return result

    current = series[0]["value"] if series else 0
    prev = series[1]["value"] if len(series) > 1 else current
    prev_6m = series[6]["value"] if len(series) > 6 else current

    if current < prev:
        trend = "cutting"
    elif current > prev:
        trend = "hiking"
    else:
        trend = "holding"

    result = {
        "name": "Federal Funds Rate",
        "current": current,
        "previous": prev,
        "change": round(current - prev, 2),
        "six_month_change": round(current - prev_6m, 2),
        "unit": "%",
        "trend": trend,
        "frequency": "Monthly",
        "history": series[:36],
        "status": "live",
        "data_freshness": datetime.now(timezone.utc).isoformat(),
    }
    _cache_set("fed_rate", result)
    return result


async def fetch_unemployment() -> Dict:
    cached = _cache_get("unemployment")
    if cached:
        return cached

    data = await _av_fetch("UNEMPLOYMENT")
    series = _parse_av_series(data)
    if not series:
        result = _unavailable_indicator("Unemployment Rate", "%", "Monthly")
        _macro_cache.set("unemployment", result, ttl=UNAVAILABLE_CACHE_TTL)
        return result

    current = series[0]["value"] if series else 0
    prev = series[1]["value"] if len(series) > 1 else current

    result = {
        "name": "Unemployment Rate",
        "current": current,
        "previous": prev,
        "change": round(current - prev, 2),
        "unit": "%",
        "trend": "rising" if current > prev + 0.2 else "falling" if current < prev - 0.2 else "stable",
        "frequency": "Monthly",
        "history": series[:24],
        "status": "live",
        "data_freshness": datetime.now(timezone.utc).isoformat(),
    }
    _cache_set("unemployment", result)
    return result


_TREASURY_YF_MAP = {
    "3month": "^IRX",
    "2year": "2YY=F",
    "5year": "^FVX",
    "10year": "^TNX",
    "30year": "^TYX",
}


async def _yf_treasury_yield(maturity: str) -> Optional[Dict]:
    yf_symbol = _TREASURY_YF_MAP.get(maturity)
    if not yf_symbol:
        return None
    try:
        import yfinance as yf
        t = await asyncio.to_thread(lambda: yf.Ticker(yf_symbol))
        hist = await asyncio.to_thread(lambda: t.history(period="3mo"))
        if hist is None or hist.empty:
            return None
        values = []
        for idx in hist.index:
            val = float(hist.loc[idx, "Close"])
            if yf_symbol == "^IRX":
                val = val / 100.0
            values.append({"date": idx.strftime("%Y-%m-%d"), "value": round(val, 3)})
        values.reverse()
        return values if values else None
    except Exception as e:
        logger.warning(f"yfinance treasury {maturity} failed: {e}")
        return None


async def fetch_treasury_yield(maturity: str = "10year") -> Dict:
    cached = _cache_get(f"treasury_{maturity}")
    if cached:
        return cached

    series = []
    data = await _av_fetch("TREASURY_YIELD", {"interval": "monthly", "maturity": maturity})
    series = _parse_av_series(data)

    if not series:
        yf_series = await _yf_treasury_yield(maturity)
        if yf_series:
            series = yf_series

    if not series:
        result = _unavailable_indicator(f"Treasury Yield ({maturity})", "%", "Monthly")
        result["maturity"] = maturity
        _macro_cache.set(f"treasury_{maturity}", result, ttl=UNAVAILABLE_CACHE_TTL)
        return result

    current = series[0]["value"] if series else 0
    prev = series[1]["value"] if len(series) > 1 else current

    result = {
        "name": f"Treasury Yield ({maturity})",
        "current": current,
        "previous": prev,
        "change": round(current - prev, 2),
        "unit": "%",
        "trend": "rising" if current > prev + 0.1 else "falling" if current < prev - 0.1 else "stable",
        "maturity": maturity,
        "frequency": "Monthly",
        "history": series[:36],
        "status": "live",
        "data_freshness": datetime.now(timezone.utc).isoformat(),
    }
    _cache_set(f"treasury_{maturity}", result)
    return result


async def fetch_economic_calendar() -> List[Dict]:
    cached = _cache_get("econ_calendar", ECON_CALENDAR_CACHE_TTL)
    if cached:
        return cached

    events = await _fmp_fetch("economic_calendar", {
        "from": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "to": (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d"),
    })

    if not events or not isinstance(events, list):
        _cache_set("econ_calendar", [])
        return []

    result = []
    for ev in events[:30]:
        result.append({
            "date": ev.get("date", ""),
            "event": ev.get("event", ev.get("name", "")),
            "country": ev.get("country", "US"),
            "impact": ev.get("impact", "Medium"),
            "actual": ev.get("actual"),
            "estimate": ev.get("estimate"),
            "previous": ev.get("previous"),
        })

    _cache_set("econ_calendar", result)
    return result


async def calculate_yield_curve() -> Dict:
    y2 = await fetch_treasury_yield("2year")
    y10 = await fetch_treasury_yield("10year")
    y5 = await fetch_treasury_yield("5year")
    y30 = await fetch_treasury_yield("30year")

    spread_2_10 = round(y10["current"] - y2["current"], 3)
    inverted = spread_2_10 < 0

    y2_hist = y2.get("history", [])
    y10_hist = y10.get("history", [])
    spread_history = []
    for i in range(min(len(y2_hist), len(y10_hist), 24)):
        spread_history.append({
            "date": y10_hist[i]["date"],
            "spread": round(y10_hist[i]["value"] - y2_hist[i]["value"], 3),
        })

    if inverted and spread_2_10 < -0.5:
        signal = "STRONG_RECESSION_WARNING"
    elif inverted:
        signal = "RECESSION_WARNING"
    elif spread_2_10 < 0.3:
        signal = "FLATTENING"
    elif spread_2_10 > 1.0:
        signal = "STEEPENING"
    else:
        signal = "NORMAL"

    return {
        "yield_2y": y2["current"],
        "yield_5y": y5["current"],
        "yield_10y": y10["current"],
        "yield_30y": y30["current"],
        "spread_2_10": spread_2_10,
        "inverted": inverted,
        "signal": signal,
        "spread_history": spread_history,
    }


async def calculate_recession_probability() -> Dict:
    unemployment = await fetch_unemployment()
    yield_curve = await calculate_yield_curve()
    fed_rate = await fetch_fed_rate()
    inflation = await fetch_inflation()

    score = 0.0
    factors = []

    if yield_curve["inverted"]:
        inversion_severity = min(abs(yield_curve["spread_2_10"]) * 30, 35)
        score += inversion_severity
        factors.append({"factor": "Yield Curve Inverted", "impact": round(inversion_severity, 1), "detail": f"2Y-10Y spread: {yield_curve['spread_2_10']}%"})
    elif yield_curve["spread_2_10"] < 0.3:
        score += 10
        factors.append({"factor": "Yield Curve Flattening", "impact": 10, "detail": f"Spread narrowing to {yield_curve['spread_2_10']}%"})

    unemp = unemployment.get("current", 0)
    unemp_change = unemployment.get("change", 0)
    if unemp_change > 0.5:
        contrib = min(unemp_change * 20, 25)
        score += contrib
        factors.append({"factor": "Rising Unemployment", "impact": round(contrib, 1), "detail": f"Rate: {unemp}%, change: +{unemp_change}%"})
    elif unemp > 5.0:
        score += 15
        factors.append({"factor": "High Unemployment", "impact": 15, "detail": f"Rate: {unemp}%"})

    fed = fed_rate.get("current", 0)
    if fed > 5.0:
        contrib = min((fed - 4.0) * 8, 20)
        score += contrib
        factors.append({"factor": "Restrictive Monetary Policy", "impact": round(contrib, 1), "detail": f"Fed rate: {fed}%"})

    inf = inflation.get("current", 0)
    if inf > 5.0:
        score += 10
        factors.append({"factor": "High Inflation", "impact": 10, "detail": f"Inflation: {inf}%"})

    probability = min(round(score, 1), 100)

    if probability >= 60:
        risk_level = "HIGH"
    elif probability >= 35:
        risk_level = "ELEVATED"
    elif probability >= 15:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    return {
        "probability": probability,
        "risk_level": risk_level,
        "factors": factors,
    }


async def classify_macro_regime() -> Dict:
    gdp = await fetch_gdp()
    unemployment = await fetch_unemployment()
    inflation = await fetch_inflation()
    fed_rate = await fetch_fed_rate()
    yield_curve = await calculate_yield_curve()
    recession = await calculate_recession_probability()

    gdp_growth = gdp.get("change", 0)
    unemp = unemployment.get("current", 0)
    unemp_trend = unemployment.get("trend", "stable")
    inf = inflation.get("current", 0)
    fed_trend = fed_rate.get("trend", "holding")

    if recession["probability"] >= 50:
        regime = "RECESSION"
        confidence = min(recession["probability"], 90)
        description = "Economic contraction with deteriorating fundamentals"
    elif gdp_growth < 0 and unemp_trend == "rising":
        regime = "RECESSION"
        confidence = 70
        description = "GDP contracting with rising unemployment"
    elif gdp_growth > 0 and unemp < 4.5 and inf > 3.0 and fed_trend in ("hiking", "holding"):
        regime = "LATE_CYCLE"
        confidence = 65
        description = "Economy growing but with inflation pressure and tight policy"
    elif gdp_growth > 0 and unemp_trend == "falling" and inf < 3.0:
        regime = "EXPANSION"
        confidence = 70
        description = "Healthy growth with controlled inflation"
    elif gdp_growth > 0 and unemp_trend == "falling" and fed_trend == "cutting":
        regime = "RECOVERY"
        confidence = 65
        description = "Economy recovering with supportive monetary policy"
    elif gdp_growth > 0:
        regime = "EXPANSION"
        confidence = 55
        description = "Economy is growing"
    else:
        regime = "LATE_CYCLE"
        confidence = 50
        description = "Mixed signals across economic indicators"

    regime_colors = {
        "EXPANSION": "positive",
        "LATE_CYCLE": "warning",
        "RECESSION": "negative",
        "RECOVERY": "info",
    }

    return {
        "regime": regime,
        "confidence": confidence,
        "description": description,
        "color": regime_colors.get(regime, "warning"),
        "key_drivers": [
            f"GDP: {'Growing' if gdp_growth > 0 else 'Contracting'} ({gdp_growth:+.1f}B)",
            f"Unemployment: {unemp}% ({unemp_trend})",
            f"Inflation: {inf}%",
            f"Fed Policy: {fed_trend.title()}",
            f"Yield Curve: {'Inverted' if yield_curve['inverted'] else 'Normal'} ({yield_curve['spread_2_10']:+.2f}%)",
        ],
    }


async def fetch_vix() -> Dict:
    cached = _cache_get("vix")
    if cached:
        return cached
    try:
        import yfinance as yf
        t = await asyncio.to_thread(lambda: yf.Ticker("^VIX"))
        hist = await asyncio.to_thread(lambda: t.history(period="3mo"))
        if hist is not None and not hist.empty:
            current = round(float(hist["Close"].iloc[-1]), 2)
            prev = round(float(hist["Close"].iloc[-2]), 2) if len(hist) > 1 else current
            change = round(current - prev, 2)
            sparkline = [round(float(v), 2) for v in hist["Close"].values[-20:]]
            if current >= 30:
                trend = "elevated"
            elif current >= 20:
                trend = "normal"
            else:
                trend = "low"
            result = {
                "name": "VIX",
                "current": current,
                "previous": prev,
                "change": change,
                "unit": "",
                "trend": trend,
                "frequency": "Daily",
                "history": [{"date": idx.strftime("%Y-%m-%d"), "value": round(float(hist.loc[idx, "Close"]), 2)} for idx in hist.index[-60:]],
                "sparkline": sparkline,
                "status": "live",
            }
            _cache_set("vix", result)
            return result
    except Exception as e:
        logger.warning(f"VIX fetch failed: {e}")
    result = _unavailable_indicator("VIX", "", "Daily")
    _macro_cache.set("vix", result, ttl=UNAVAILABLE_CACHE_TTL)
    return result


async def get_macro_dashboard() -> Dict:
    gdp = await fetch_gdp()
    cpi = await fetch_cpi()
    inflation = await fetch_inflation()
    fed_rate = await fetch_fed_rate()
    unemployment = await fetch_unemployment()
    yield_curve = await calculate_yield_curve()
    recession = await calculate_recession_probability()
    regime = await classify_macro_regime()
    calendar = await fetch_economic_calendar()
    vix = await fetch_vix()

    y10 = await fetch_treasury_yield("10year")

    indicators = []

    if yield_curve.get("yield_10y", 0) != 0:
        indicators.append({
            "id": "treasury_10y",
            "name": "10Y Treasury",
            "value": yield_curve.get("yield_10y", 0),
            "change": round(yield_curve.get("yield_10y", 0) - y10.get("previous", 0), 2),
            "unit": "%",
            "trend": y10.get("trend", "stable"),
            "sparkline": [h["value"] for h in reversed(y10.get("history", [])[:12])],
        })

    if vix.get("current", 0) != 0:
        indicators.append({
            "id": "vix",
            "name": "VIX",
            "value": vix.get("current", 0),
            "change": vix.get("change", 0),
            "unit": "",
            "trend": vix.get("trend", "normal"),
            "sparkline": vix.get("sparkline", []),
        })

    if fed_rate.get("current", 0) != 0:
        indicators.append({
            "id": "fed_rate",
            "name": "Fed Funds Rate",
            "value": fed_rate.get("current", 0),
            "change": fed_rate.get("change", 0),
            "unit": "%",
            "trend": fed_rate.get("trend", "holding"),
            "sparkline": [h["value"] for h in reversed(fed_rate.get("history", [])[:12])],
        })

    if gdp.get("current", 0) != 0:
        indicators.append({
            "id": "gdp",
            "name": "Real GDP",
            "value": gdp.get("current", 0),
            "change": gdp.get("change", 0),
            "unit": "B USD",
            "trend": gdp.get("trend", "flat"),
            "sparkline": [h["value"] for h in reversed(gdp.get("history", [])[:8])],
        })

    if cpi.get("change_yoy", 0) != 0:
        indicators.append({
            "id": "cpi",
            "name": "CPI (YoY)",
            "value": cpi.get("change_yoy", 0),
            "change": cpi.get("change_mom", 0),
            "unit": "%",
            "trend": cpi.get("trend", "stable"),
            "sparkline": [h["value"] for h in reversed(cpi.get("history", [])[:12])],
        })

    if inflation.get("current", 0) != 0:
        indicators.append({
            "id": "inflation",
            "name": "Inflation",
            "value": inflation.get("current", 0),
            "change": inflation.get("change", 0),
            "unit": "%",
            "trend": inflation.get("trend", "stable"),
            "sparkline": [h["value"] for h in reversed(inflation.get("history", [])[:8])],
        })

    if unemployment.get("current", 0) != 0:
        indicators.append({
            "id": "unemployment",
            "name": "Unemployment",
            "value": unemployment.get("current", 0),
            "change": unemployment.get("change", 0),
            "unit": "%",
            "trend": unemployment.get("trend", "stable"),
            "sparkline": [h["value"] for h in reversed(unemployment.get("history", [])[:12])],
        })

    if not indicators:
        indicators = [
            {"id": "treasury_10y", "name": "10Y Treasury", "value": 0, "change": 0, "unit": "%", "trend": "unavailable", "sparkline": []},
            {"id": "vix", "name": "VIX", "value": 0, "change": 0, "unit": "", "trend": "unavailable", "sparkline": []},
            {"id": "fed_rate", "name": "Fed Funds Rate", "value": 0, "change": 0, "unit": "%", "trend": "unavailable", "sparkline": []},
        ]

    return {
        "indicators": indicators,
        "yield_curve": yield_curve,
        "recession": recession,
        "regime": regime,
        "economic_calendar": calendar[:15],
        "data_freshness": datetime.now(timezone.utc).isoformat(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _unavailable_indicator(name: str, unit: str = "", frequency: str = "") -> Dict:
    return {
        "name": name,
        "current": 0,
        "previous": 0,
        "change": 0,
        "unit": unit,
        "trend": "unavailable",
        "frequency": frequency,
        "history": [],
        "status": "unavailable",
        "data_freshness": datetime.now(timezone.utc).isoformat(),
    }
