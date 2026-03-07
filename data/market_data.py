import yfinance as yf
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
import time
import random
import logging

logger = logging.getLogger(__name__)

try:
    from data.polygon_client import PolygonClient, POLYGON_API_KEY
    _polygon_available = bool(POLYGON_API_KEY)
except ImportError:
    _polygon_available = False
    POLYGON_API_KEY = ""

TRACKED_TICKERS = ["AAPL", "TSLA", "NVDA", "GOOGL", "MSFT", "AMZN", "META", "VOO"]
INDEX_MAP = {
    "^GSPC": "S&P 500",
    "^IXIC": "NASDAQ",
    "^DJI": "DOW 30",
}

TICKER_NAMES = {
    "AAPL": "Apple Inc.",
    "TSLA": "Tesla Inc.",
    "NVDA": "NVIDIA Corp.",
    "GOOGL": "Alphabet Inc.",
    "MSFT": "Microsoft Corp.",
    "AMZN": "Amazon.com Inc.",
    "META": "Meta Platforms",
    "VOO": "Vanguard S&P 500 ETF",
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "JPM": "JPMorgan Chase",
    "V": "Visa Inc.",
    "AMD": "AMD Inc.",
    "NFLX": "Netflix Inc.",
    "DIS": "Walt Disney",
    "BA": "Boeing Co.",
}

TICKER_SECTORS = {
    "AAPL": "Technology", "TSLA": "Automotive", "NVDA": "Technology",
    "GOOGL": "Technology", "MSFT": "Technology", "AMZN": "Consumer",
    "META": "Technology", "VOO": "ETF", "BTC-USD": "Crypto",
    "ETH-USD": "Crypto", "JPM": "Finance", "V": "Finance",
    "AMD": "Technology", "NFLX": "Communication", "DIS": "Entertainment",
    "BA": "Industrial",
}

_cache = {}
_cache_ttl = 60


def _check_polygon():
    global _polygon_available
    _polygon_available = bool(POLYGON_API_KEY)
    return _polygon_available


def _get_cached(key):
    if key in _cache:
        data, ts = _cache[key]
        if time.time() - ts < _cache_ttl:
            return data
    return None


def _set_cache(key, data):
    _cache[key] = (data, time.time())


def _fetch_quote_polygon(ticker_symbol):
    """Fetch quote via Polygon.io synchronously (runs async client in thread)."""
    try:
        import asyncio

        async def _fetch():
            client = PolygonClient()
            try:
                snap = await client.get_snapshot(ticker_symbol)
                if not snap or snap.price == 0:
                    return None
                return {
                    "ticker": ticker_symbol,
                    "name": TICKER_NAMES.get(ticker_symbol, ticker_symbol),
                    "price": snap.price,
                    "change": snap.change,
                    "change_pct": snap.change_pct,
                    "volume": snap.volume,
                    "sector": TICKER_SECTORS.get(ticker_symbol, "Other"),
                }
            finally:
                await client.close()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in async context — cannot nest; skip polygon
            return None

        result = asyncio.run(_fetch())
        if result:
            _set_cache(f"quote:{ticker_symbol}", result)
        return result
    except Exception as e:
        logger.warning(f"Polygon quote failed for {ticker_symbol}: {e}")
        return None


def _fetch_quote_yfinance(ticker_symbol):
    try:
        t = yf.Ticker(ticker_symbol)
        info = t.fast_info
        price = float(info.get("lastPrice", 0) or info.get("last_price", 0) or 0)
        prev_close = float(info.get("previousClose", 0) or info.get("previous_close", 0) or price)

        if price == 0:
            hist = t.history(period="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
                if len(hist) > 1:
                    prev_close = float(hist["Close"].iloc[-2])

        if price == 0:
            return None

        change = round(price - prev_close, 2)
        change_pct = round((change / prev_close) * 100, 2) if prev_close else 0
        volume = int(info.get("lastVolume", 0) or info.get("last_volume", 0) or 0)

        result = {
            "ticker": ticker_symbol,
            "name": TICKER_NAMES.get(ticker_symbol, ticker_symbol),
            "price": round(price, 2),
            "change": change,
            "change_pct": change_pct,
            "volume": volume,
            "sector": TICKER_SECTORS.get(ticker_symbol, "Other"),
        }
        _set_cache(f"quote:{ticker_symbol}", result)
        return result
    except Exception as e:
        print(f"Error fetching {ticker_symbol}: {e}")
        return None


def _fetch_quote(ticker_symbol):
    cached = _get_cached(f"quote:{ticker_symbol}")
    if cached:
        return cached

    if _check_polygon():
        result = _fetch_quote_polygon(ticker_symbol)
        if result:
            return result

    return _fetch_quote_yfinance(ticker_symbol)


def batch_fetch_quotes(tickers):
    cached_results = {}
    uncached = []
    for t in tickers:
        cached = _get_cached(f"quote:{t}")
        if cached:
            cached_results[t] = cached
        else:
            uncached.append(t)

    results = dict(cached_results)

    if uncached:
        if _check_polygon() and len(uncached) > 1:
            try:
                import asyncio

                async def _batch_fetch():
                    client = PolygonClient()
                    try:
                        return {t: await client.get_snapshot(t) for t in uncached}
                    finally:
                        await client.close()

                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop is None or not loop.is_running():
                    snapshots = asyncio.run(_batch_fetch())
                    still_uncached = []
                    for t in uncached:
                        snap = snapshots.get(t)
                        if snap and snap.price > 0:
                            result = {
                                "ticker": t,
                                "name": TICKER_NAMES.get(t, t),
                                "price": snap.price,
                                "change": snap.change,
                                "change_pct": snap.change_pct,
                                "volume": snap.volume,
                                "sector": TICKER_SECTORS.get(t, "Other"),
                            }
                            _set_cache(f"quote:{t}", result)
                            results[t] = result
                        else:
                            still_uncached.append(t)
                    uncached = still_uncached
            except Exception as e:
                logger.warning(f"Polygon batch snapshots failed: {e}")

        if uncached:
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(_fetch_quote, t): t for t in uncached}
                for future in futures:
                    t = futures[future]
                    try:
                        q = future.result()
                        if q:
                            results[t] = q
                    except Exception as e:
                        logger.warning(f"Parallel fetch failed for {t}: {e}")

    return results


def get_real_quotes(tickers=None):
    if tickers is None:
        tickers = TRACKED_TICKERS

    batch = batch_fetch_quotes(tickers)
    quotes = [batch[t] for t in tickers if t in batch]
    return quotes


def _fetch_single_index(symbol, name):
    cached = _get_cached(f"index:{symbol}")
    if cached:
        return cached

    entry = None

    if _check_polygon():
        try:
            import asyncio

            async def _fetch_idx():
                client = PolygonClient()
                try:
                    snap = await client.get_snapshot(symbol.replace("^", "I:"))
                    return snap
                finally:
                    await client.close()

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is None or not loop.is_running():
                snap = asyncio.run(_fetch_idx())
                if snap and snap.price > 0:
                    entry = {
                        "symbol": symbol.replace("^", ""),
                        "name": name,
                        "value": snap.price,
                        "change": snap.change,
                        "change_pct": snap.change_pct,
                    }
                    _set_cache(f"index:{symbol}", entry)
        except Exception as e:
            logger.warning(f"Polygon index failed for {symbol}: {e}")

    if not entry:
        try:
            t = yf.Ticker(symbol)
            info = t.fast_info
            value = float(info.get("lastPrice", 0) or info.get("last_price", 0) or 0)
            prev = float(info.get("previousClose", 0) or info.get("previous_close", 0) or value)

            if value == 0:
                hist = t.history(period="1d")
                if not hist.empty:
                    value = float(hist["Close"].iloc[-1])

            change = round(value - prev, 2)
            change_pct = round((change / prev) * 100, 2) if prev else 0

            entry = {
                "symbol": symbol.replace("^", ""),
                "name": name,
                "value": round(value, 2),
                "change": change,
                "change_pct": change_pct,
            }
            _set_cache(f"index:{symbol}", entry)
        except Exception as e:
            print(f"Error fetching index {symbol}: {e}")

    return entry


def get_real_indices():
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_fetch_single_index, symbol, name): symbol for symbol, name in INDEX_MAP.items()}
        results = []
        for future in futures:
            try:
                entry = future.result()
                if entry:
                    results.append(entry)
            except Exception as e:
                logger.warning(f"Parallel index fetch failed: {e}")
    return results


def get_real_movers():
    all_tickers = ["AAPL", "TSLA", "NVDA", "GOOGL", "MSFT", "AMZN", "META", "AMD", "NFLX", "JPM", "V", "BA", "DIS"]
    batch = batch_fetch_quotes(all_tickers)
    quotes = [batch[t] for t in all_tickers if t in batch]

    quotes.sort(key=lambda x: x["change_pct"], reverse=True)
    gainers = [q for q in quotes if q["change_pct"] > 0][:4]
    losers = [q for q in quotes if q["change_pct"] < 0][:4]

    if not losers:
        losers = sorted(quotes, key=lambda x: x["change_pct"])[:4]
    if not gainers:
        gainers = sorted(quotes, key=lambda x: x["change_pct"], reverse=True)[:4]

    return {"gainers": gainers, "losers": losers}


def get_ticker_history(ticker, period="1mo"):
    cached = _get_cached(f"history:{ticker}:{period}")
    if cached:
        return cached

    if _check_polygon():
        try:
            import asyncio

            period_to_days = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}
            days = period_to_days.get(period, 30)

            async def _fetch_hist():
                client = PolygonClient()
                try:
                    return await client.get_daily_bars(ticker, lookback_days=days)
                finally:
                    await client.close()

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is None or not loop.is_running():
                hist = asyncio.run(_fetch_hist())
                if hist and len(hist) > 0:
                    _set_cache(f"history:{ticker}:{period}", hist)
                    return hist
        except Exception as e:
            logger.warning(f"Polygon history failed for {ticker}: {e}")

    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period)
        if hist.empty:
            return []
        data = []
        for date, row in hist.iterrows():
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            })
        _set_cache(f"history:{ticker}:{period}", data)
        return data
    except Exception:
        return []


def get_data_provider() -> str:
    if _check_polygon():
        return "polygon"
    return "yfinance"
