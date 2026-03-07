import os
import sys
import math
import logging
import random
import time
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx
from prediction.api_utils import resilient_get

from prediction.cache_manager import SmartCache

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_options_cache = SmartCache("options_flow", max_size=500, default_ttl=300)
_OPTIONS_CACHE_TTL = 300


def _cache_get(key: str):
    return _options_cache.get(key, _OPTIONS_CACHE_TTL)


def _cache_set(key: str, data: Any, ttl: int = _OPTIONS_CACHE_TTL):
    _options_cache.set(key, data)


def _fmp_request(endpoint: str, params: Optional[Dict] = None, timeout: float = 10.0) -> Optional[Any]:
    api_key = os.environ.get("FMP_API_KEY")
    if not api_key:
        return None
    url = f"https://financialmodelingprep.com/stable{endpoint}"
    if params is None:
        params = {}
    params["apikey"] = api_key
    try:
        resp = resilient_get(url, params=params, timeout=timeout, source="fmp")
        if resp is None:
            return None
        if resp.status_code in (402, 403):
            logger.info(f"FMP {endpoint} returned {resp.status_code} — not available on current plan")
            return None
        if resp.status_code >= 400:
            logger.warning(f"FMP {endpoint} returned {resp.status_code}")
            return None
        data = resp.json()
        if isinstance(data, dict) and "Error Message" in data:
            return None
        return data
    except Exception as e:
        logger.warning(f"FMP options request failed for {endpoint}: {e}")
        return None


def _fetch_yfinance_chain(ticker: str, current_price: float) -> List[Dict]:
    import numpy as np

    def _safe_num(val, default=0):
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return default
        try:
            f = float(val)
            if np.isnan(f) or np.isinf(f):
                return default
            return f
        except (ValueError, TypeError):
            return default

    cache_key = f"yf_opts:{ticker}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    try:
        import yfinance as yf
        yticker = yf.Ticker(ticker)
        expirations = yticker.options
        if not expirations:
            return []

        contracts = []
        now = datetime.now()
        max_exps = min(6, len(expirations))

        for exp_str in expirations[:max_exps]:
            try:
                chain = yticker.option_chain(exp_str)
            except Exception:
                continue

            try:
                exp_dt = datetime.strptime(exp_str, "%Y-%m-%d")
                dte = max(0, (exp_dt - now).days)
            except Exception:
                dte = 30

            for df, ctype in [(chain.calls, "call"), (chain.puts, "put")]:
                if df is None or df.empty:
                    continue
                for _, row in df.iterrows():
                    strike = _safe_num(row.get("strike", 0))
                    if strike <= 0:
                        continue
                    bid = _safe_num(row.get("bid", 0))
                    ask = _safe_num(row.get("ask", 0))
                    mid = round((bid + ask) / 2, 2) if bid > 0 or ask > 0 else _safe_num(row.get("lastPrice", 0))
                    vol = int(_safe_num(row.get("volume", 0)))
                    oi = int(_safe_num(row.get("openInterest", 0)))
                    iv = _safe_num(row.get("impliedVolatility", 0))
                    itm = bool(row.get("inTheMoney", False))

                    t = max(dte, 1) / 365.0
                    d1 = 0
                    if iv > 0 and t > 0 and strike > 0:
                        try:
                            d1 = (math.log(current_price / strike) + (0.05 + 0.5 * iv ** 2) * t) / (iv * math.sqrt(t))
                        except Exception:
                            pass
                    nd1 = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
                    delta = round(nd1 if ctype == "call" else nd1 - 1, 4)
                    gamma = round(math.exp(-d1 ** 2 / 2) / (current_price * iv * math.sqrt(2 * math.pi * t)), 6) if iv > 0 and t > 0 else 0
                    theta = round(-current_price * iv * math.exp(-d1 ** 2 / 2) / (2 * math.sqrt(2 * math.pi * t)) / 365, 4) if iv > 0 and t > 0 else 0
                    vega = round(current_price * math.sqrt(t) * math.exp(-d1 ** 2 / 2) / math.sqrt(2 * math.pi) / 100, 4) if t > 0 else 0

                    contracts.append({
                        "contract_ticker": str(row.get("contractSymbol", f"O:{ticker}{exp_str.replace('-','')}{ctype[0].upper()}{int(strike*1000):08d}")),
                        "underlying_ticker": ticker,
                        "contract_type": ctype,
                        "expiration_date": exp_str,
                        "strike_price": strike,
                        "bid": round(bid, 2),
                        "ask": round(ask, 2),
                        "mid_price": mid,
                        "volume": vol,
                        "open_interest": oi,
                        "implied_volatility": round(iv, 4),
                        "delta": delta,
                        "gamma": gamma,
                        "theta": theta,
                        "vega": vega,
                        "in_the_money": itm,
                        "days_to_expiry": dte,
                    })

        if contracts:
            _cache_set(cache_key, contracts)
        return contracts
    except Exception as e:
        logger.warning(f"yfinance options chain error for {ticker}: {e}")
        return []


def _fetch_polygon_options_snapshot(ticker: str) -> Optional[List[Dict]]:
    try:
        from data.polygon_client import _get_api_key, _request
        api_key = _get_api_key()
        if not api_key:
            return None

        cache_key = f"opt_snap:{ticker}"
        cached = _cache_get(cache_key)
        if cached:
            return cached

        url = f"https://api.polygon.io/v3/snapshot/options/{ticker.upper()}"
        params = {"apiKey": api_key, "limit": "250"}
        try:
            resp = resilient_get(url, params=params, timeout=15.0, source="polygon")
            if resp is None or resp.status_code != 200:
                return None
            data = resp.json()
        except Exception as e:
            logger.warning(f"Polygon options snapshot failed: {e}")
            return None

        results = data.get("results", [])
        if not results:
            return None

        contracts = []
        for r in results:
            details = r.get("details", {})
            greeks = r.get("greeks", {})
            day = r.get("day", {})
            last_quote = r.get("last_quote", {})

            ct = details.get("contract_type", "").lower()
            if ct not in ("call", "put"):
                continue

            strike = float(details.get("strike_price", 0))
            exp = details.get("expiration_date", "")

            bid = float(last_quote.get("bid", 0) or 0)
            ask = float(last_quote.get("ask", 0) or 0)
            mid = round((bid + ask) / 2, 2) if bid > 0 or ask > 0 else 0

            vol = int(day.get("volume", 0) or 0)
            oi = int(r.get("open_interest", 0) or 0)
            iv = float(r.get("implied_volatility", 0) or 0)

            exp_dt = None
            try:
                exp_dt = datetime.strptime(exp, "%Y-%m-%d")
            except Exception:
                pass
            dte = max(0, (exp_dt - datetime.now()).days) if exp_dt else 0

            from_price = r.get("underlying_asset", {}).get("price", 0)
            itm = False
            if from_price and strike:
                if ct == "call":
                    itm = from_price > strike
                else:
                    itm = from_price < strike

            contracts.append({
                "contract_ticker": details.get("ticker", r.get("ticker", "")),
                "underlying_ticker": ticker.upper(),
                "contract_type": ct,
                "expiration_date": exp,
                "strike_price": strike,
                "bid": round(bid, 2),
                "ask": round(ask, 2),
                "mid_price": mid,
                "volume": vol,
                "open_interest": oi,
                "implied_volatility": round(iv, 4),
                "delta": round(float(greeks.get("delta", 0) or 0), 4),
                "gamma": round(float(greeks.get("gamma", 0) or 0), 6),
                "theta": round(float(greeks.get("theta", 0) or 0), 4),
                "vega": round(float(greeks.get("vega", 0) or 0), 4),
                "in_the_money": itm,
                "days_to_expiry": dte,
            })

        if contracts:
            _cache_set(cache_key, contracts)
        return contracts if contracts else None

    except Exception as e:
        logger.warning(f"Polygon options snapshot error for {ticker}: {e}")
        return None


def _fetch_polygon_chain(ticker: str) -> Optional[List[Dict]]:
    snapshot = _fetch_polygon_options_snapshot(ticker)
    if snapshot and len(snapshot) > 10:
        has_pricing = any(
            c.get("volume") is not None or c.get("bid") is not None for c in snapshot[:5]
        )
        if has_pricing:
            return snapshot

    try:
        from data.polygon_client import is_polygon_available, get_options_chain, get_snapshot
        if not is_polygon_available():
            return None

        snap = get_snapshot(ticker)
        if not snap:
            return None

        chain = get_options_chain(ticker, limit=250)
        if not chain:
            return None

        return chain
    except Exception as e:
        logger.warning(f"Polygon options fetch failed for {ticker}: {e}")
        return None


def _fetch_fmp_options(ticker: str) -> Optional[List[Dict]]:
    cache_key = f"fmp_opts:{ticker}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    data = _fmp_request(f"/stock-options", {"symbol": ticker.upper()})
    if not data or not isinstance(data, list):
        return None

    contracts = []
    for item in data:
        ct = (item.get("type", "") or item.get("putCall", "")).lower()
        if ct not in ("call", "put"):
            continue

        strike = float(item.get("strike", 0) or item.get("strikePrice", 0))
        exp = item.get("expiration", "") or item.get("expirationDate", "")
        bid = float(item.get("bid", 0) or 0)
        ask = float(item.get("ask", 0) or 0)

        contracts.append({
            "contract_ticker": f"O:{ticker}{exp.replace('-', '')}{ct[0].upper()}{int(strike * 1000):08d}",
            "underlying_ticker": ticker.upper(),
            "contract_type": ct,
            "expiration_date": exp,
            "strike_price": strike,
            "bid": round(bid, 2),
            "ask": round(ask, 2),
            "mid_price": round((bid + ask) / 2, 2),
            "volume": int(item.get("volume", 0) or 0),
            "open_interest": int(item.get("openInterest", 0) or 0),
            "implied_volatility": round(float(item.get("impliedVolatility", 0) or 0), 4),
            "delta": round(float(item.get("delta", 0) or 0), 4),
            "gamma": round(float(item.get("gamma", 0) or 0), 6),
            "theta": round(float(item.get("theta", 0) or 0), 4),
            "vega": round(float(item.get("vega", 0) or 0), 4),
            "in_the_money": bool(item.get("inTheMoney", False)),
            "days_to_expiry": int(item.get("daysToExpiration", 0) or 0),
        })

    if contracts:
        _cache_set(cache_key, contracts)
    return contracts if contracts else None


_last_known_prices: Dict[str, float] = {}

def _get_current_price(ticker: str) -> float:
    ticker = ticker.upper()
    try:
        import data.market_data as market_data
        q = market_data._fetch_quote(ticker)
        if q and q.get("price", 0) > 0:
            _last_known_prices[ticker] = q["price"]
            return q["price"]
    except Exception:
        pass
    try:
        import yfinance as yf
        yticker = yf.Ticker(ticker)
        info = yticker.info or {}
        price = float(info.get("regularMarketPrice", 0) or info.get("currentPrice", 0) or 0)
        if price > 0:
            _last_known_prices[ticker] = price
            return price
    except Exception:
        pass
    if ticker in _last_known_prices:
        logger.warning(f"Using last-known cached price for {ticker}: {_last_known_prices[ticker]}")
        return _last_known_prices[ticker]
    logger.warning(f"No price available for {ticker}, returning 0")
    return 0.0


def compute_max_pain(contracts: List[Dict], current_price: float) -> Dict:
    strikes = sorted(set(c["strike_price"] for c in contracts))
    if not strikes:
        return {"max_pain_strike": current_price, "distance_pct": 0}

    min_pain = float("inf")
    max_pain_strike = current_price

    for test_strike in strikes:
        total_pain = 0
        for c in contracts:
            oi = c.get("open_interest", 0)
            if oi <= 0:
                continue
            if c["contract_type"] == "call":
                pain = max(0, test_strike - c["strike_price"]) * oi * 100
            else:
                pain = max(0, c["strike_price"] - test_strike) * oi * 100
            total_pain += pain

        if total_pain < min_pain:
            min_pain = total_pain
            max_pain_strike = test_strike

    distance_pct = round((max_pain_strike - current_price) / current_price * 100, 2) if current_price > 0 else 0

    return {
        "max_pain_strike": max_pain_strike,
        "distance_pct": distance_pct,
        "total_oi_pain": round(min_pain, 0),
    }


def detect_unusual_activity(contracts: List[Dict], current_price: float = 0) -> List[Dict]:
    unusual = []
    for c in contracts:
        vol = c.get("volume", 0)
        oi = c.get("open_interest", 1)
        if oi == 0:
            oi = 1

        vol_oi_ratio = vol / oi
        mid = c.get("mid_price", 0)
        premium = mid * vol * 100

        is_unusual = False
        reason = []
        trade_type = "standard"

        if vol_oi_ratio > 5.0 and vol > 500:
            is_unusual = True
            reason.append(f"Vol/OI {vol_oi_ratio:.1f}x (>5x)")
            trade_type = "sweep"
        elif vol_oi_ratio > 3.0 and vol > 500:
            is_unusual = True
            reason.append(f"Vol/OI {vol_oi_ratio:.1f}x")

        if vol > 10000:
            is_unusual = True
            reason.append(f"Block trade {vol:,} contracts")
            trade_type = "block"
        elif vol > 5000 and vol_oi_ratio > 1.5:
            is_unusual = True
            reason.append(f"High volume {vol:,}")

        if premium > 1_000_000:
            is_unusual = True
            reason.append(f"Premium ${premium / 1_000_000:.1f}M")
            if trade_type == "standard":
                trade_type = "block"

        if c.get("implied_volatility", 0) > 0.8:
            is_unusual = True
            reason.append(f"IV {c['implied_volatility']*100:.0f}%")
        elif c.get("implied_volatility", 0) > 0.6:
            reason.append(f"Elevated IV {c['implied_volatility']*100:.0f}%")

        if is_unusual:
            strike = c["strike_price"]
            ct = c["contract_type"]
            direction = "bullish" if ct == "call" else "bearish"

            if current_price > 0:
                otm_pct = abs(strike - current_price) / current_price
                if ct == "call" and strike < current_price:
                    direction = "bullish"
                elif ct == "put" and strike > current_price:
                    direction = "bearish"
                elif ct == "call" and otm_pct > 0.1:
                    reason.append(f"Deep OTM call ({otm_pct*100:.0f}%)")
                elif ct == "put" and otm_pct > 0.1:
                    reason.append(f"Deep OTM put ({otm_pct*100:.0f}%)")

            unusual.append({
                "contract": c.get("contract_ticker", ""),
                "type": ct,
                "strike": strike,
                "expiry": c.get("expiration_date", ""),
                "volume": vol,
                "open_interest": oi if oi > 1 else 0,
                "vol_oi_ratio": round(vol_oi_ratio, 2),
                "iv": c.get("implied_volatility", 0),
                "mid_price": mid,
                "premium_total": round(premium, 0),
                "reasons": reason,
                "sentiment": direction,
                "trade_type": trade_type,
                "days_to_expiry": c.get("days_to_expiry", 0),
            })

    unusual.sort(key=lambda x: x.get("premium_total", 0), reverse=True)
    return unusual[:20]


def compute_oi_analysis(contracts: List[Dict], current_price: float) -> Dict:
    call_oi = sum(c.get("open_interest", 0) for c in contracts if c["contract_type"] == "call")
    put_oi = sum(c.get("open_interest", 0) for c in contracts if c["contract_type"] == "put")
    total_oi = call_oi + put_oi

    call_vol = sum(c.get("volume", 0) for c in contracts if c["contract_type"] == "call")
    put_vol = sum(c.get("volume", 0) for c in contracts if c["contract_type"] == "put")
    total_vol = call_vol + put_vol

    strikes = sorted(set(c["strike_price"] for c in contracts))
    oi_by_strike = []
    for s in strikes:
        s_call_oi = sum(c.get("open_interest", 0) for c in contracts if c["contract_type"] == "call" and c["strike_price"] == s)
        s_put_oi = sum(c.get("open_interest", 0) for c in contracts if c["contract_type"] == "put" and c["strike_price"] == s)
        if s_call_oi > 0 or s_put_oi > 0:
            oi_by_strike.append({
                "strike": s,
                "call_oi": s_call_oi,
                "put_oi": s_put_oi,
                "net_oi": s_call_oi - s_put_oi,
                "relative_to_spot": round((s - current_price) / current_price * 100, 2),
            })

    top_call_strikes = sorted([s for s in oi_by_strike], key=lambda x: x["call_oi"], reverse=True)[:5]
    top_put_strikes = sorted([s for s in oi_by_strike], key=lambda x: x["put_oi"], reverse=True)[:5]

    return {
        "call_oi": call_oi,
        "put_oi": put_oi,
        "total_oi": total_oi,
        "call_volume": call_vol,
        "put_volume": put_vol,
        "total_volume": total_vol,
        "oi_put_call_ratio": round(put_oi / max(call_oi, 1), 3),
        "vol_put_call_ratio": round(put_vol / max(call_vol, 1), 3),
        "oi_by_strike": oi_by_strike,
        "top_call_strikes": top_call_strikes,
        "top_put_strikes": top_put_strikes,
    }


def compute_net_premium_flow(contracts: List[Dict]) -> Dict:
    call_premium = 0.0
    put_premium = 0.0
    flow_by_expiry: Dict[str, Dict] = {}

    for c in contracts:
        vol = c.get("volume", 0)
        mid = c.get("mid_price", 0)
        premium = mid * vol * 100

        exp = c.get("expiration_date", "unknown")
        if exp not in flow_by_expiry:
            flow_by_expiry[exp] = {"expiry": exp, "call_premium": 0, "put_premium": 0}

        if c["contract_type"] == "call":
            call_premium += premium
            flow_by_expiry[exp]["call_premium"] += premium
        else:
            put_premium += premium
            flow_by_expiry[exp]["put_premium"] += premium

    total_premium = call_premium + put_premium
    net_premium = call_premium - put_premium

    if net_premium > total_premium * 0.2:
        flow_bias = "bullish"
    elif net_premium < -total_premium * 0.2:
        flow_bias = "bearish"
    else:
        flow_bias = "neutral"

    flow_timeline = []
    for exp, data in sorted(flow_by_expiry.items()):
        net = data["call_premium"] - data["put_premium"]
        flow_timeline.append({
            "expiry": exp,
            "call_premium": round(data["call_premium"], 0),
            "put_premium": round(data["put_premium"], 0),
            "net_premium": round(net, 0),
            "bias": "bullish" if net > 0 else "bearish",
        })

    return {
        "call_premium": round(call_premium, 0),
        "put_premium": round(put_premium, 0),
        "total_premium": round(total_premium, 0),
        "net_premium": round(net_premium, 0),
        "flow_bias": flow_bias,
        "premium_put_call_ratio": round(put_premium / max(call_premium, 1), 3),
        "flow_by_expiry": flow_timeline,
    }


def compute_iv_metrics(contracts: List[Dict]) -> Dict:
    ivs = [c.get("implied_volatility", 0) for c in contracts if c.get("implied_volatility", 0) > 0]
    if not ivs:
        return {
            "current_iv": 0,
            "iv_rank": 0,
            "iv_percentile": 0,
            "iv_high": 0,
            "iv_low": 0,
            "iv_mean": 0,
            "iv_skew": 0,
            "iv_term_structure": [],
        }

    atm_ivs = []
    for c in contracts:
        if c.get("implied_volatility", 0) > 0 and abs(c.get("delta", 0)) > 0.35 and abs(c.get("delta", 0)) < 0.65:
            atm_ivs.append(c["implied_volatility"])

    current_iv = sum(atm_ivs) / len(atm_ivs) if atm_ivs else sum(ivs) / len(ivs)
    iv_high = max(ivs)
    iv_low = min(ivs)
    iv_mean = sum(ivs) / len(ivs)

    iv_range = iv_high - iv_low
    iv_rank = round(((current_iv - iv_low) / iv_range * 100) if iv_range > 0 else 50, 1)
    sorted_ivs = sorted(ivs)
    below_count = sum(1 for iv in sorted_ivs if iv <= current_iv)
    iv_percentile = round(below_count / len(sorted_ivs) * 100, 1)

    call_atm_ivs = [c["implied_volatility"] for c in contracts
                    if c["contract_type"] == "call" and c.get("implied_volatility", 0) > 0
                    and abs(c.get("delta", 0)) > 0.35 and abs(c.get("delta", 0)) < 0.65]
    put_atm_ivs = [c["implied_volatility"] for c in contracts
                   if c["contract_type"] == "put" and c.get("implied_volatility", 0) > 0
                   and abs(c.get("delta", 0)) > 0.35 and abs(c.get("delta", 0)) < 0.65]

    call_atm_avg = sum(call_atm_ivs) / len(call_atm_ivs) if call_atm_ivs else current_iv
    put_atm_avg = sum(put_atm_ivs) / len(put_atm_ivs) if put_atm_ivs else current_iv
    iv_skew = round(put_atm_avg - call_atm_avg, 4)

    term_structure: Dict[str, List[float]] = {}
    for c in contracts:
        exp = c.get("expiration_date", "")
        iv = c.get("implied_volatility", 0)
        if exp and iv > 0 and abs(c.get("delta", 0)) > 0.3 and abs(c.get("delta", 0)) < 0.7:
            if exp not in term_structure:
                term_structure[exp] = []
            term_structure[exp].append(iv)

    iv_term = []
    for exp in sorted(term_structure.keys()):
        avg = sum(term_structure[exp]) / len(term_structure[exp])
        dte = 0
        try:
            dte = max(0, (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days)
        except Exception:
            pass
        iv_term.append({
            "expiry": exp,
            "days_to_expiry": dte,
            "avg_iv": round(avg, 4),
        })

    return {
        "current_iv": round(current_iv, 4),
        "iv_rank": min(100, max(0, iv_rank)),
        "iv_percentile": min(100, max(0, iv_percentile)),
        "iv_high": round(iv_high, 4),
        "iv_low": round(iv_low, 4),
        "iv_mean": round(iv_mean, 4),
        "iv_skew": iv_skew,
        "iv_term_structure": iv_term,
    }


def compute_iv_surface(contracts: List[Dict], current_price: float) -> List[Dict]:
    surface_data: Dict[str, Dict[float, Dict]] = {}

    for c in contracts:
        exp = c.get("expiration_date", "")
        strike = c["strike_price"]
        iv = c.get("implied_volatility", 0)
        if not exp or iv <= 0:
            continue

        if exp not in surface_data:
            surface_data[exp] = {}
        key = strike
        if key not in surface_data[exp] or c["contract_type"] == "call":
            dte = 0
            try:
                dte = max(0, (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days)
            except Exception:
                pass
            surface_data[exp][key] = {
                "expiry": exp,
                "days_to_expiry": dte,
                "strike": strike,
                "moneyness": round((strike / current_price - 1) * 100, 2) if current_price > 0 else 0,
                "iv": round(iv, 4),
                "contract_type": c["contract_type"],
            }

    result = []
    for exp in sorted(surface_data.keys()):
        for strike in sorted(surface_data[exp].keys()):
            result.append(surface_data[exp][strike])

    return result[:200]


def get_options_flow(ticker: str) -> Dict[str, Any]:
    ticker = ticker.upper()
    current_price = _get_current_price(ticker)

    polygon_chain = _fetch_polygon_chain(ticker)
    has_pricing = polygon_chain and len(polygon_chain) > 10 and any(
        c.get("volume") is not None or c.get("bid") is not None for c in polygon_chain[:5]
    )

    if has_pricing and polygon_chain:
        for c in polygon_chain:
            c.setdefault("bid", 0)
            c.setdefault("ask", 0)
            c.setdefault("mid_price", round((c.get("bid", 0) + c.get("ask", 0)) / 2, 2))
            c.setdefault("volume", 0)
            c.setdefault("open_interest", 0)
            c.setdefault("implied_volatility", 0)
            c.setdefault("delta", 0)
            c.setdefault("gamma", 0)
            c.setdefault("theta", 0)
            c.setdefault("vega", 0)
            c.setdefault("in_the_money", False)
            c.setdefault("days_to_expiry", 0)
            c.setdefault("contract_ticker", c.get("ticker", ""))
        contracts = polygon_chain
        data_source = "polygon"
    else:
        yf_chain = _fetch_yfinance_chain(ticker, current_price)
        if yf_chain and len(yf_chain) > 5:
            contracts = yf_chain
            data_source = "yfinance"
        else:
            fmp_chain = _fetch_fmp_options(ticker)
            if fmp_chain and len(fmp_chain) > 5:
                contracts = fmp_chain
                data_source = "fmp"
            else:
                contracts = []
                data_source = "unavailable"

    calls = [c for c in contracts if c["contract_type"] == "call"]
    puts = [c for c in contracts if c["contract_type"] == "put"]

    call_volume = sum(c.get("volume", 0) for c in calls)
    put_volume = sum(c.get("volume", 0) for c in puts)
    total_volume = call_volume + put_volume

    call_oi = sum(c.get("open_interest", 0) for c in calls)
    put_oi = sum(c.get("open_interest", 0) for c in puts)

    pcr_volume = round(put_volume / max(call_volume, 1), 3)
    pcr_oi = round(put_oi / max(call_oi, 1), 3)

    if pcr_volume < 0.7:
        pcr_sentiment = "bullish"
    elif pcr_volume > 1.2:
        pcr_sentiment = "bearish"
    else:
        pcr_sentiment = "neutral"

    max_pain = compute_max_pain(contracts, current_price)
    unusual = detect_unusual_activity(contracts, current_price)
    oi_analysis = compute_oi_analysis(contracts, current_price)
    net_premium = compute_net_premium_flow(contracts)
    iv_metrics = compute_iv_metrics(contracts)
    iv_surface = compute_iv_surface(contracts, current_price)

    expirations = sorted(set(c.get("expiration_date", "") for c in contracts))
    nearest_exp = expirations[0] if expirations else None

    nearest_contracts = [c for c in contracts if c.get("expiration_date") == nearest_exp] if nearest_exp else contracts[:50]

    nearest_calls = sorted(
        [c for c in nearest_contracts if c["contract_type"] == "call"],
        key=lambda x: x["strike_price"]
    )
    nearest_puts = sorted(
        [c for c in nearest_contracts if c["contract_type"] == "put"],
        key=lambda x: x["strike_price"]
    )

    avg_iv_calls = sum(c.get("implied_volatility", 0) for c in calls) / max(len(calls), 1)
    avg_iv_puts = sum(c.get("implied_volatility", 0) for c in puts) / max(len(puts), 1)

    bullish_signals = sum(1 for u in unusual if u["sentiment"] == "bullish")
    bearish_signals = sum(1 for u in unusual if u["sentiment"] == "bearish")

    sweep_count = sum(1 for u in unusual if u.get("trade_type") == "sweep")
    block_count = sum(1 for u in unusual if u.get("trade_type") == "block")

    if bullish_signals > bearish_signals * 1.5 and pcr_sentiment != "bearish":
        flow_sentiment = "bullish"
    elif bearish_signals > bullish_signals * 1.5 and pcr_sentiment != "bullish":
        flow_sentiment = "bearish"
    elif net_premium.get("flow_bias") == "bullish" and pcr_sentiment != "bearish":
        flow_sentiment = "bullish"
    elif net_premium.get("flow_bias") == "bearish" and pcr_sentiment != "bullish":
        flow_sentiment = "bearish"
    else:
        flow_sentiment = "neutral"

    return {
        "ticker": ticker,
        "current_price": round(current_price, 2),
        "data_source": data_source,
        "put_call_ratio": {
            "volume": pcr_volume,
            "open_interest": pcr_oi,
            "sentiment": pcr_sentiment,
        },
        "max_pain": max_pain,
        "unusual_activity": unusual,
        "oi_analysis": oi_analysis,
        "net_premium_flow": net_premium,
        "iv_metrics": iv_metrics,
        "iv_surface": iv_surface,
        "chain": {
            "calls": nearest_calls[:20],
            "puts": nearest_puts[:20],
            "expiration": nearest_exp,
            "expirations_available": expirations,
        },
        "summary": {
            "total_contracts": len(contracts),
            "total_volume": total_volume,
            "call_volume": call_volume,
            "put_volume": put_volume,
            "call_oi": call_oi,
            "put_oi": put_oi,
            "avg_iv_calls": round(avg_iv_calls, 4),
            "avg_iv_puts": round(avg_iv_puts, 4),
            "unusual_count": len(unusual),
            "bullish_signals": bullish_signals,
            "bearish_signals": bearish_signals,
            "sweep_count": sweep_count,
            "block_count": block_count,
            "flow_sentiment": flow_sentiment,
            "net_premium": net_premium.get("net_premium", 0),
            "total_premium": net_premium.get("total_premium", 0),
            "iv_rank": iv_metrics.get("iv_rank", 0),
            "iv_percentile": iv_metrics.get("iv_percentile", 0),
        },
        "data_freshness": datetime.now(timezone.utc).isoformat(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
