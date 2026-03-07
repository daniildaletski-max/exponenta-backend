import yfinance as yf
import time
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_heatmap_cache = {}
_heatmap_cache_ttl = 120

HEATMAP_UNIVERSE = {
    "Technology": [
        ("AAPL", "Apple", 3000), ("MSFT", "Microsoft", 2800), ("NVDA", "NVIDIA", 2500),
        ("GOOGL", "Alphabet", 1900), ("META", "Meta", 1200), ("AVGO", "Broadcom", 800),
        ("ORCL", "Oracle", 400), ("CRM", "Salesforce", 300), ("AMD", "AMD", 280),
        ("ADBE", "Adobe", 250), ("CSCO", "Cisco", 220), ("ACN", "Accenture", 210),
        ("INTC", "Intel", 120), ("QCOM", "Qualcomm", 190), ("TXN", "Texas Instruments", 175),
        ("NOW", "ServiceNow", 170), ("IBM", "IBM", 160), ("AMAT", "Applied Materials", 150),
    ],
    "Healthcare": [
        ("UNH", "UnitedHealth", 500), ("JNJ", "Johnson & Johnson", 380),
        ("LLY", "Eli Lilly", 750), ("ABBV", "AbbVie", 310), ("MRK", "Merck", 290),
        ("PFE", "Pfizer", 160), ("TMO", "Thermo Fisher", 210), ("ABT", "Abbott", 190),
        ("DHR", "Danaher", 180), ("AMGN", "Amgen", 150),
    ],
    "Finance": [
        ("JPM", "JPMorgan", 550), ("V", "Visa", 500), ("MA", "Mastercard", 420),
        ("BAC", "Bank of America", 300), ("WFC", "Wells Fargo", 200),
        ("GS", "Goldman Sachs", 150), ("MS", "Morgan Stanley", 140),
        ("BLK", "BlackRock", 130), ("AXP", "American Express", 170),
    ],
    "Consumer Discretionary": [
        ("AMZN", "Amazon", 1900), ("TSLA", "Tesla", 800), ("HD", "Home Depot", 370),
        ("MCD", "McDonald's", 210), ("NKE", "Nike", 120), ("SBUX", "Starbucks", 100),
        ("LOW", "Lowe's", 140), ("TJX", "TJX Companies", 120),
    ],
    "Communication": [
        ("NFLX", "Netflix", 280), ("DIS", "Disney", 170), ("CMCSA", "Comcast", 160),
        ("TMUS", "T-Mobile", 230), ("VZ", "Verizon", 170),
    ],
    "Consumer Staples": [
        ("WMT", "Walmart", 500), ("PG", "Procter & Gamble", 370), ("COST", "Costco", 350),
        ("KO", "Coca-Cola", 260), ("PEP", "PepsiCo", 230),
    ],
    "Energy": [
        ("XOM", "Exxon Mobil", 450), ("CVX", "Chevron", 280), ("COP", "ConocoPhillips", 130),
        ("SLB", "Schlumberger", 70), ("EOG", "EOG Resources", 65),
    ],
    "Industrial": [
        ("GE", "GE Aerospace", 190), ("CAT", "Caterpillar", 170), ("RTX", "RTX Corp", 150),
        ("HON", "Honeywell", 140), ("UPS", "UPS", 110), ("BA", "Boeing", 120),
        ("DE", "Deere", 110), ("LMT", "Lockheed Martin", 120),
    ],
}

PERIOD_MAP = {
    "1D": "1d",
    "1W": "5d",
    "1M": "1mo",
}


def _fetch_change(ticker: str, period_key: str) -> dict | None:
    cache_key = f"hm:{ticker}:{period_key}"
    if cache_key in _heatmap_cache:
        data, ts = _heatmap_cache[cache_key]
        if time.time() - ts < _heatmap_cache_ttl:
            return data

    yf_period = PERIOD_MAP.get(period_key, "1d")
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=yf_period)
        if hist.empty or len(hist) < 1:
            return None

        current = float(hist["Close"].iloc[-1])
        volume = 0

        if period_key == "1D":
            fi = t.fast_info
            prev_close = None
            try:
                prev_close = float(fi["previousClose"]) if "previousClose" in fi else None
            except Exception:
                pass
            if prev_close is None:
                try:
                    prev_close = float(fi["previous_close"]) if "previous_close" in fi else None
                except Exception:
                    pass
            if prev_close is None or prev_close == 0:
                prev_close = float(hist["Open"].iloc[-1]) if len(hist) >= 1 else current
            start = prev_close
            try:
                volume = int(fi.get("lastVolume", 0) or fi.get("last_volume", 0) or 0)
            except Exception:
                volume = int(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else 0
        else:
            start = float(hist["Close"].iloc[0])
            volume = int(hist["Volume"].sum()) if "Volume" in hist.columns else 0

        change_pct = round(((current - start) / start) * 100, 2) if start else 0

        result = {
            "price": round(current, 2),
            "change_pct": change_pct,
            "volume": volume,
        }
        _heatmap_cache[cache_key] = (result, time.time())
        return result
    except Exception as e:
        logger.warning(f"Heatmap fetch failed for {ticker}: {e}")
        return None


def get_market_heatmap(timeframe: str = "1D") -> dict:
    if timeframe not in PERIOD_MAP:
        timeframe = "1D"

    sectors = []
    all_stocks = []
    total_market_cap = 0

    for sector_name, tickers in HEATMAP_UNIVERSE.items():
        sector_stocks = []
        sector_cap = 0

        for ticker, name, approx_cap_b in tickers:
            data = _fetch_change(ticker, timeframe)
            if data is None:
                continue

            stock = {
                "ticker": ticker,
                "name": name,
                "price": data["price"],
                "change_pct": data["change_pct"],
                "market_cap_b": approx_cap_b,
                "volume": data["volume"],
                "sector": sector_name,
            }
            sector_stocks.append(stock)
            sector_cap += approx_cap_b
            total_market_cap += approx_cap_b

        if sector_stocks:
            avg_change = round(
                sum(s["change_pct"] * s["market_cap_b"] for s in sector_stocks) / sector_cap
                if sector_cap > 0 else 0,
                2,
            )
            sectors.append({
                "name": sector_name,
                "stocks": sector_stocks,
                "total_market_cap_b": sector_cap,
                "avg_change_pct": avg_change,
                "stock_count": len(sector_stocks),
            })
            all_stocks.extend(sector_stocks)

    for sector in sectors:
        sector["weight"] = round(sector["total_market_cap_b"] / total_market_cap, 4) if total_market_cap > 0 else 0
        for stock in sector["stocks"]:
            stock["weight_in_sector"] = round(
                stock["market_cap_b"] / sector["total_market_cap_b"], 4
            ) if sector["total_market_cap_b"] > 0 else 0

    sectors.sort(key=lambda x: x["total_market_cap_b"], reverse=True)

    advancing = sum(1 for s in all_stocks if s["change_pct"] > 0)
    declining = sum(1 for s in all_stocks if s["change_pct"] < 0)

    return {
        "sectors": sectors,
        "timeframe": timeframe,
        "total_stocks": len(all_stocks),
        "advancing": advancing,
        "declining": declining,
        "unchanged": len(all_stocks) - advancing - declining,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
