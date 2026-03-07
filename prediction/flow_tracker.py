import os
import sys
import math
import hashlib
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from prediction.api_utils import resilient_get

from prediction.cache_manager import SmartCache

logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")
FMP_BASE = "https://financialmodelingprep.com/stable"
FINNHUB_BASE = "https://finnhub.io/api/v1"

_cache = SmartCache("flow_tracker", max_size=500, default_ttl=300)
INSIDER_TTL = 300
INSTITUTIONAL_TTL = 3600
CONGRESS_TTL = 3600


def _cache_get(key: str, ttl: int) -> Optional[Any]:
    return _cache.get(key, ttl)


def _cache_set(key: str, data: Any):
    _cache.set(key, data)


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


def _fmp_request(path: str, params: Optional[Dict] = None) -> Optional[Any]:
    if not FMP_API_KEY:
        return None
    try:
        p = params or {}
        p["apikey"] = FMP_API_KEY
        resp = resilient_get(f"{FMP_BASE}{path}", params=p, timeout=10, source="fmp")
        if resp is None:
            return None
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict) and ("Error Message" in data or "Legacy Endpoint" in str(data.get("Error Message", ""))):
                logger.warning(f"FMP legacy endpoint error for {path}")
                return None
            return data
        if resp.status_code in (402, 403):
            logger.info(f"FMP {path} returned {resp.status_code} — endpoint not available on current plan")
            return None
        logger.warning(f"FMP {path} returned {resp.status_code}")
    except Exception as e:
        logger.warning(f"FMP request failed {path}: {e}")
    return None


def _finnhub_request(path: str, params: Optional[Dict] = None) -> Optional[Any]:
    if not FINNHUB_API_KEY:
        return None
    try:
        p = params or {}
        p["token"] = FINNHUB_API_KEY
        resp = resilient_get(f"{FINNHUB_BASE}{path}", params=p, timeout=10, source="finnhub")
        if resp is None:
            return None
        if resp.status_code == 200:
            return resp.json()
        logger.warning(f"Finnhub {path} returned {resp.status_code}")
    except Exception as e:
        logger.warning(f"Finnhub request failed {path}: {e}")
    return None


def _fetch_insider_transactions(ticker: str, current_price: float) -> List[Dict]:
    cache_key = f"insider_tx_{ticker}"
    cached = _cache_get(cache_key, INSIDER_TTL)
    if cached is not None:
        return cached

    transactions = []
    now = datetime.now(timezone.utc)

    fmp_data = _fmp_request("/insider-trading", {"symbol": ticker, "page": "0"})
    if fmp_data and isinstance(fmp_data, list):
        for item in fmp_data[:30]:
            tx_date_str = item.get("transactionDate", item.get("filingDate", ""))
            if not tx_date_str:
                continue
            try:
                tx_date = datetime.strptime(tx_date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            days_ago = max(0, (now - tx_date).days)

            tx_type_raw = (item.get("transactionType", "") or "").strip()
            if "purchase" in tx_type_raw.lower() or tx_type_raw in ("P", "P-Purchase"):
                tx_type = "Purchase"
            elif "sale" in tx_type_raw.lower() or tx_type_raw in ("S", "S-Sale"):
                tx_type = "Sale"
            elif tx_type_raw in ("A", "A-Award"):
                tx_type = "Award"
            else:
                tx_type = tx_type_raw or "Other"

            shares = abs(int(item.get("securitiesTransacted", 0) or 0))
            price = float(item.get("price", 0) or 0)
            if price <= 0:
                price = current_price
            total_value = round(shares * price, 2)

            owner_name = item.get("reportingName", "") or item.get("ownerName", "") or "Unknown"
            title = item.get("typeOfOwner", "") or ""

            role_category = "Officer"
            title_lower = title.lower()
            if any(t in title_lower for t in ["ceo", "chief executive"]):
                role_category = "C-Suite"
                if not title or title == title_lower:
                    title = "CEO"
            elif any(t in title_lower for t in ["cfo", "chief financial"]):
                role_category = "C-Suite"
                if not title or title == title_lower:
                    title = "CFO"
            elif any(t in title_lower for t in ["coo", "cto", "chief"]):
                role_category = "C-Suite"
            elif "director" in title_lower:
                role_category = "Director"
            elif "vp" in title_lower or "vice president" in title_lower:
                role_category = "VP"
            elif "10%" in title_lower or "owner" in title_lower:
                role_category = "Major Holder"
            elif "board" in title_lower:
                role_category = "Board"

            weight_map = {"C-Suite": 3.0, "Board": 1.5, "Director": 1.2, "VP": 1.8, "Major Holder": 2.0, "Officer": 1.3}

            shares_after = abs(int(item.get("securitiesOwned", 0) or 0))
            pct_of_holdings = round(shares / max(shares_after, 1) * 100, 2) if shares_after > 0 else 0

            transactions.append({
                "date": tx_date_str[:10],
                "insider_name": owner_name,
                "title": title or role_category,
                "role_category": role_category,
                "transaction_type": tx_type,
                "shares": shares,
                "price": round(price, 2),
                "total_value": total_value,
                "shares_owned_after": shares_after,
                "pct_of_holdings": pct_of_holdings,
                "insider_weight": weight_map.get(role_category, 1.0),
                "filing_type": "Form 4",
                "days_ago": days_ago,
                "source": "SEC Form 4 (FMP)",
            })

    finnhub_data = _finnhub_request("/stock/insider-transactions", {"symbol": ticker})
    if finnhub_data and isinstance(finnhub_data, dict):
        fh_txs = finnhub_data.get("data", [])
        existing_dates_names = {(t["date"], t["insider_name"]) for t in transactions}
        for item in fh_txs[:20]:
            tx_date_str = item.get("transactionDate", "")
            name = item.get("name", "Unknown")
            if not tx_date_str:
                continue
            if (tx_date_str[:10], name) in existing_dates_names:
                continue
            try:
                tx_date = datetime.strptime(tx_date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            days_ago = max(0, (now - tx_date).days)

            tx_code = item.get("transactionCode", "")
            if tx_code == "P":
                tx_type = "Purchase"
            elif tx_code == "S":
                tx_type = "Sale"
            elif tx_code == "A":
                tx_type = "Award"
            else:
                tx_type = tx_code or "Other"

            shares = abs(int(item.get("share", 0) or 0))
            change = float(item.get("change", 0) or 0)
            price_val = float(item.get("transactionPrice", 0) or 0)
            if price_val <= 0:
                price_val = current_price
            total_value = round(shares * price_val, 2)

            transactions.append({
                "date": tx_date_str[:10],
                "insider_name": name,
                "title": item.get("filingType", "Officer"),
                "role_category": "Officer",
                "transaction_type": tx_type,
                "shares": shares,
                "price": round(price_val, 2),
                "total_value": total_value,
                "shares_owned_after": abs(int(item.get("share", 0) or 0)),
                "pct_of_holdings": 0,
                "insider_weight": 1.0,
                "filing_type": "Form 4",
                "days_ago": days_ago,
                "source": "SEC (Finnhub)",
            })

    transactions.sort(key=lambda x: x["days_ago"])
    _cache_set(cache_key, transactions)
    return transactions


def _fetch_insider_sentiment(ticker: str) -> Dict:
    cache_key = f"insider_sentiment_{ticker}"
    cached = _cache_get(cache_key, INSIDER_TTL)
    if cached is not None:
        return cached

    result = {"mspr": 0, "change": 0, "available": False}
    data = _finnhub_request("/stock/insider-sentiment", {"symbol": ticker, "from": "2024-01-01"})
    if data and isinstance(data, dict):
        sentiments = data.get("data", [])
        if sentiments:
            latest = sentiments[-1] if sentiments else {}
            result = {
                "mspr": round(float(latest.get("mspr", 0) or 0), 4),
                "change": round(float(latest.get("change", 0) or 0), 2),
                "available": True,
            }

    _cache_set(cache_key, result)
    return result


def _fetch_congressional_trades(ticker: str) -> List[Dict]:
    cache_key = f"congress_{ticker}"
    cached = _cache_get(cache_key, CONGRESS_TTL)
    if cached is not None:
        return cached

    trades = []
    data = _finnhub_request("/stock/congressional-trading", {"symbol": ticker})
    if data and isinstance(data, dict):
        items = data.get("data", [])
        now = datetime.now(timezone.utc)
        for item in items[:20]:
            tx_date = item.get("transactionDate", "")
            trades.append({
                "name": item.get("name", "Unknown"),
                "chamber": item.get("chamber", "Unknown"),
                "transaction_type": item.get("transactionType", "Unknown"),
                "transaction_date": tx_date,
                "disclosure_date": item.get("disclosureDate", ""),
                "amount_range": item.get("amountFrom", ""),
                "amount_from": item.get("amountFrom", 0),
                "amount_to": item.get("amountTo", 0),
                "asset_description": item.get("assetDescription", ticker),
                "source": "Congressional Trading (Finnhub)",
            })

    _cache_set(cache_key, trades)
    return trades


def _fetch_institutional_holders(ticker: str) -> Dict:
    cache_key = f"institutional_{ticker}"
    cached = _cache_get(cache_key, INSTITUTIONAL_TTL)
    if cached is not None:
        return cached

    result = {
        "institutional_ownership_pct": 0,
        "change_qoq_pct": 0,
        "sentiment": "unknown",
        "num_institutional_holders": 0,
        "new_positions": 0,
        "closed_positions": 0,
        "increased_positions": 0,
        "decreased_positions": 0,
        "top_holders": [],
    }

    try:
        import yfinance as yf
        yticker = yf.Ticker(ticker)
        holders_df = yticker.institutional_holders
        if holders_df is not None and not holders_df.empty:
            holders_list = []
            for _, row in holders_df.iterrows():
                shares = int(row.get("Shares", 0) or 0)
                pct = float(row.get("% Out", 0) or 0)
                date_reported = ""
                if row.get("Date Reported") is not None:
                    try:
                        date_reported = row["Date Reported"].strftime("%Y-%m-%d")
                    except Exception:
                        date_reported = str(row.get("Date Reported", ""))[:10]
                holders_list.append({
                    "name": str(row.get("Holder", "Unknown")),
                    "shares": shares,
                    "change": 0,
                    "pct_ownership": round(pct * 100, 2) if pct < 1 else round(pct, 2),
                    "change_pct": 0,
                    "action": "Held",
                    "date_reported": date_reported,
                })

            holders_list.sort(key=lambda x: x["shares"], reverse=True)
            top_holders = holders_list[:10]
            total_ownership = sum(h["pct_ownership"] for h in top_holders)

            info = yticker.info or {}
            inst_pct = float(info.get("heldPercentInstitutions", 0) or 0) * 100

            sentiment = "stable"
            if inst_pct > 80:
                sentiment = "strong_accumulation"
            elif inst_pct > 60:
                sentiment = "accumulation"
            elif inst_pct < 20:
                sentiment = "distribution"

            result = {
                "institutional_ownership_pct": round(inst_pct, 1) if inst_pct > 0 else round(total_ownership, 1),
                "change_qoq_pct": 0,
                "sentiment": sentiment,
                "num_institutional_holders": len(holders_df),
                "new_positions": 0,
                "closed_positions": 0,
                "increased_positions": 0,
                "decreased_positions": 0,
                "top_holders": top_holders,
            }
    except Exception as e:
        logger.warning(f"yfinance institutional holders error for {ticker}: {e}")

    _cache_set(cache_key, result)
    return result


def _derive_institutional_flow_signal(institutional: Dict) -> Dict:
    change_qoq = institutional.get("change_qoq_pct", 0)
    increased = institutional.get("increased_positions", 0)
    decreased = institutional.get("decreased_positions", 0)
    new_pos = institutional.get("new_positions", 0)

    net_flow_ratio = (increased + new_pos) / max(increased + new_pos + decreased, 1)

    if net_flow_ratio > 0.65 and change_qoq > 1:
        sentiment = "accumulation"
        score = min(100, 60 + change_qoq * 5 + net_flow_ratio * 20)
    elif net_flow_ratio > 0.55:
        sentiment = "slight_accumulation"
        score = 55 + net_flow_ratio * 15
    elif net_flow_ratio < 0.35 and change_qoq < -1:
        sentiment = "distribution"
        score = max(0, 40 + change_qoq * 5 - (1 - net_flow_ratio) * 20)
    elif net_flow_ratio < 0.45:
        sentiment = "slight_distribution"
        score = 45 - (1 - net_flow_ratio) * 15
    else:
        sentiment = "neutral"
        score = 50

    return {
        "score": round(min(100, max(0, score)), 1),
        "sentiment": sentiment,
        "net_flow_ratio": round(net_flow_ratio, 3),
        "institutional_change_qoq": change_qoq,
    }


def _detect_insider_clusters(transactions: List[Dict], window_days: int = 14) -> List[Dict]:
    buys = [t for t in transactions if t["transaction_type"] == "Purchase"]
    if len(buys) < 2:
        return []

    clusters = []
    used = set()

    for i, tx in enumerate(buys):
        if i in used:
            continue
        cluster_members = [tx]
        cluster_indices = {i}

        for j, other in enumerate(buys):
            if j in used or j == i:
                continue
            if abs(tx["days_ago"] - other["days_ago"]) <= window_days:
                cluster_members.append(other)
                cluster_indices.add(j)

        if len(cluster_members) >= 2:
            used.update(cluster_indices)
            total_value = sum(m["total_value"] for m in cluster_members)
            unique_insiders = list(set(m["insider_name"] for m in cluster_members))
            has_csuite = any(m["role_category"] == "C-Suite" for m in cluster_members)

            if len(unique_insiders) >= 3:
                strength = "Strong"
            elif has_csuite:
                strength = "Moderate-Strong"
            else:
                strength = "Moderate"

            start_date = min(m["date"] for m in cluster_members)
            end_date = max(m["date"] for m in cluster_members)

            clusters.append({
                "start_date": start_date,
                "end_date": end_date,
                "num_insiders": len(unique_insiders),
                "insider_names": unique_insiders,
                "total_shares": sum(m["shares"] for m in cluster_members),
                "total_value": round(total_value, 2),
                "has_c_suite": has_csuite,
                "strength": strength,
                "transactions": cluster_members,
            })

    clusters.sort(key=lambda x: x["total_value"], reverse=True)
    return clusters


def _compute_insider_score(transactions: List[Dict], clusters: List[Dict], insider_sentiment: Optional[Dict] = None) -> Dict:
    if not transactions:
        return {"score": 50, "breakdown": {}, "signal": "neutral"}

    buys = [t for t in transactions if t["transaction_type"] == "Purchase"]
    sells = [t for t in transactions if t["transaction_type"] == "Sale"]

    buy_value = sum(t["total_value"] for t in buys)
    sell_value = sum(t["total_value"] for t in sells)
    total_value = buy_value + sell_value

    if total_value > 0:
        buy_ratio = buy_value / total_value
    else:
        buy_ratio = 0.5

    buy_sell_score = buy_ratio * 100

    csuite_buys = [t for t in buys if t["role_category"] == "C-Suite"]
    csuite_value = sum(t["total_value"] for t in csuite_buys)
    seniority_score = min(100, (len(csuite_buys) * 25) + (csuite_value / max(total_value, 1) * 50))

    recent_buys = [t for t in buys if t["days_ago"] <= 14]
    recency_score = min(100, len(recent_buys) * 30)

    cluster_score = 0
    if clusters:
        cluster_score = min(100, sum(
            30 * (1.5 if c["has_c_suite"] else 1.0) * (c["num_insiders"] / 2)
            for c in clusters
        ))

    size_scores = []
    for t in buys:
        if t["pct_of_holdings"] > 20:
            size_scores.append(100)
        elif t["pct_of_holdings"] > 10:
            size_scores.append(75)
        elif t["pct_of_holdings"] > 5:
            size_scores.append(50)
        else:
            size_scores.append(25)
    size_score = sum(size_scores) / len(size_scores) if size_scores else 50

    mspr_score = 50
    if insider_sentiment and insider_sentiment.get("available"):
        mspr = insider_sentiment.get("mspr", 0)
        mspr_score = min(100, max(0, 50 + mspr * 100))

    weighted = (
        buy_sell_score * 0.20 +
        seniority_score * 0.20 +
        recency_score * 0.15 +
        cluster_score * 0.15 +
        size_score * 0.15 +
        mspr_score * 0.15
    )
    final_score = round(min(100, max(0, weighted)), 1)

    if final_score >= 70:
        signal = "bullish"
    elif final_score >= 55:
        signal = "slightly_bullish"
    elif final_score <= 30:
        signal = "bearish"
    elif final_score <= 45:
        signal = "slightly_bearish"
    else:
        signal = "neutral"

    return {
        "score": final_score,
        "signal": signal,
        "breakdown": {
            "buy_sell_ratio": round(buy_sell_score, 1),
            "seniority": round(seniority_score, 1),
            "recency": round(recency_score, 1),
            "cluster_activity": round(cluster_score, 1),
            "transaction_size": round(size_score, 1),
            "mspr_sentiment": round(mspr_score, 1),
        },
    }


def _compute_options_flow_sentiment(ticker: str) -> Dict:
    try:
        from prediction.options_flow import get_options_flow
        flow = get_options_flow(ticker)
        summary = flow.get("summary", {})
        pcr = flow.get("put_call_ratio", {})
        pcr_vol = pcr.get("volume", 1.0)

        if pcr_vol < 0.5:
            score = 85
        elif pcr_vol < 0.7:
            score = 70
        elif pcr_vol < 1.0:
            score = 55
        elif pcr_vol < 1.3:
            score = 40
        else:
            score = 25

        bullish_signals = summary.get("bullish_signals", 0)
        bearish_signals = summary.get("bearish_signals", 0)
        if bullish_signals > bearish_signals:
            score = min(100, score + 10)
        elif bearish_signals > bullish_signals:
            score = max(0, score - 10)

        return {
            "score": score,
            "put_call_ratio": pcr_vol,
            "flow_sentiment": summary.get("flow_sentiment", "neutral"),
            "unusual_count": summary.get("unusual_count", 0),
            "bullish_signals": bullish_signals,
            "bearish_signals": bearish_signals,
        }
    except Exception as e:
        logger.warning(f"Options flow fetch failed for {ticker}: {e}")
        return {
            "score": 50,
            "put_call_ratio": 1.0,
            "flow_sentiment": "neutral",
            "unusual_count": 0,
            "bullish_signals": 0,
            "bearish_signals": 0,
        }


def get_smart_money_flow(ticker: str) -> Dict[str, Any]:
    ticker = ticker.upper()
    current_price = _get_current_price(ticker)

    transactions = _fetch_insider_transactions(ticker, current_price)
    clusters = _detect_insider_clusters(transactions)
    insider_sentiment = _fetch_insider_sentiment(ticker)
    insider_score_data = _compute_insider_score(transactions, clusters, insider_sentiment)

    options_sentiment = _compute_options_flow_sentiment(ticker)
    institutional = _fetch_institutional_holders(ticker)
    institutional_flow = _derive_institutional_flow_signal(institutional)
    congressional_trades = _fetch_congressional_trades(ticker)

    insider_weight = 0.40
    options_weight = 0.30
    institutional_weight = 0.30

    smart_money_score = round(
        insider_score_data["score"] * insider_weight +
        options_sentiment["score"] * options_weight +
        institutional_flow["score"] * institutional_weight,
        1
    )
    smart_money_score = min(100, max(0, smart_money_score))

    if smart_money_score >= 75:
        overall_signal = "strong_bullish"
    elif smart_money_score >= 60:
        overall_signal = "bullish"
    elif smart_money_score >= 45:
        overall_signal = "neutral"
    elif smart_money_score >= 30:
        overall_signal = "bearish"
    else:
        overall_signal = "strong_bearish"

    buys = [t for t in transactions if t["transaction_type"] == "Purchase"]
    sells = [t for t in transactions if t["transaction_type"] == "Sale"]
    buy_value = sum(t["total_value"] for t in buys)
    sell_value = sum(t["total_value"] for t in sells)

    return {
        "ticker": ticker,
        "current_price": round(current_price, 2),
        "smart_money_score": smart_money_score,
        "overall_signal": overall_signal,
        "score_breakdown": {
            "insider_activity": {
                "score": insider_score_data["score"],
                "weight": insider_weight,
                "signal": insider_score_data["signal"],
                "components": insider_score_data["breakdown"],
            },
            "options_flow": {
                "score": options_sentiment["score"],
                "weight": options_weight,
                "put_call_ratio": options_sentiment["put_call_ratio"],
                "flow_sentiment": options_sentiment["flow_sentiment"],
                "unusual_count": options_sentiment["unusual_count"],
            },
            "institutional_flow": {
                "score": institutional_flow["score"],
                "weight": institutional_weight,
                "sentiment": institutional_flow["sentiment"],
                "net_flow_ratio": institutional_flow["net_flow_ratio"],
                "change_qoq": institutional_flow["institutional_change_qoq"],
            },
        },
        "insider_transactions": transactions[:20],
        "insider_summary": {
            "total_buys": len(buys),
            "total_sells": len(sells),
            "buy_value": round(buy_value, 2),
            "sell_value": round(sell_value, 2),
            "net_value": round(buy_value - sell_value, 2),
            "net_direction": "buying" if buy_value > sell_value else "selling",
        },
        "insider_sentiment": insider_sentiment,
        "insider_clusters": clusters,
        "institutional": institutional,
        "institutional_flow": institutional_flow,
        "congressional_trades": congressional_trades,
        "data_freshness": datetime.now(timezone.utc).isoformat(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
