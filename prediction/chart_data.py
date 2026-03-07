import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from prediction.features import fetch_ohlcv, compute_features

logger = logging.getLogger(__name__)

PERIOD_DAYS = {
    "1d": 2,
    "5d": 7,
    "1w": 7,
    "1mo": 35,
    "1M": 35,
    "3mo": 100,
    "3M": 100,
    "6mo": 190,
    "6M": 190,
    "1y": 370,
    "1Y": 370,
    "2y": 740,
    "5Y": 1300,
    "5y": 1300,
}


def get_chart_data(ticker: str, period: str = "3mo") -> Dict[str, Any]:
    days = PERIOD_DAYS.get(period, 100)
    df = fetch_ohlcv(ticker.upper(), days=max(days + 200, 400))
    if df is None or len(df) < 10:
        return {"error": f"Insufficient data for {ticker}"}

    df = compute_features(df)

    tail = df.tail(min(len(df), PERIOD_DAYS.get(period, 100)))

    candles: List[Dict[str, Any]] = []
    volumes: List[Dict[str, Any]] = []
    sma20: List[Optional[float]] = []
    sma50: List[Optional[float]] = []
    sma200: List[Optional[float]] = []
    ema20: List[Optional[float]] = []
    ema50: List[Optional[float]] = []
    ema200: List[Optional[float]] = []
    bb_upper: List[Optional[float]] = []
    bb_lower: List[Optional[float]] = []
    rsi_series: List[Optional[float]] = []
    macd_line: List[Optional[float]] = []
    macd_signal: List[Optional[float]] = []
    macd_hist: List[Optional[float]] = []
    dates: List[str] = []

    def _safe(val):
        if val is None:
            return None
        v = float(val)
        return round(v, 2) if not np.isnan(v) else None

    def _to_unix(date_val) -> int:
        try:
            if hasattr(date_val, "timestamp"):
                return int(date_val.timestamp())
            d = str(date_val)[:10]
            return int(datetime.strptime(d, "%Y-%m-%d").timestamp())
        except Exception:
            return 0

    for _, row in tail.iterrows():
        date_str = str(row.get("date", ""))[:10]
        dates.append(date_str)
        ts = _to_unix(row.get("date", date_str))

        candles.append({
            "date": date_str,
            "time": ts,
            "open": round(float(row["open"]), 2),
            "high": round(float(row["high"]), 2),
            "low": round(float(row["low"]), 2),
            "close": round(float(row["close"]), 2),
        })

        volumes.append({
            "date": date_str,
            "time": ts,
            "volume": int(row.get("volume", 0)),
            "value": int(row.get("volume", 0)),
            "color": "green" if row["close"] >= row["open"] else "red",
        })

        sma20.append(_safe(row.get("sma_20")))
        sma50.append(_safe(row.get("sma_50")))
        sma200.append(_safe(row.get("sma_200")))
        ema20.append(_safe(row.get("ema_20")))
        ema50.append(_safe(row.get("ema_50")))
        ema200.append(_safe(row.get("ema_200")))
        bb_upper.append(_safe(row.get("bb_upper")))
        bb_lower.append(_safe(row.get("bb_lower")))
        rsi_series.append(_safe(row.get("rsi")))
        macd_line.append(_safe(row.get("macd")))
        macd_signal.append(_safe(row.get("macd_signal")))
        macd_hist.append(_safe(row.get("macd_hist")))

    latest = tail.iloc[-1]
    prev = tail.iloc[-2] if len(tail) > 1 else latest
    change = round(float(latest["close"] - prev["close"]), 2)
    change_pct = round(float(change / prev["close"] * 100), 2) if prev["close"] > 0 else 0

    period_high = round(float(tail["high"].max()), 2)
    period_low = round(float(tail["low"].min()), 2)
    avg_volume = int(tail["volume"].mean()) if "volume" in tail.columns else 0

    support_levels = _find_support_resistance(tail, "support")
    resistance_levels = _find_support_resistance(tail, "resistance")

    return {
        "ticker": ticker.upper(),
        "period": period,
        "candles": candles,
        "volumes": volumes,
        "dates": dates,
        "overlays": {
            "sma20": sma20,
            "sma50": sma50,
            "sma200": sma200,
            "ema20": ema20,
            "ema50": ema50,
            "ema200": ema200,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
        },
        "oscillators": {
            "rsi": rsi_series,
            "macd": macd_line,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
        },
        "summary": {
            "current_price": round(float(latest["close"]), 2),
            "change": change,
            "change_pct": change_pct,
            "period_high": period_high,
            "period_low": period_low,
            "avg_volume": avg_volume,
            "rsi": round(float(latest.get("rsi", 50)), 1) if not np.isnan(latest.get("rsi", float("nan"))) else None,
            "adx": round(float(latest.get("adx", 0)), 1) if not np.isnan(latest.get("adx", float("nan"))) else None,
            "macd_hist": round(float(latest.get("macd_hist", 0)), 3) if not np.isnan(latest.get("macd_hist", float("nan"))) else None,
        },
        "levels": {
            "support": support_levels,
            "resistance": resistance_levels,
        },
        "data_points": len(candles),
    }


def _find_support_resistance(df, level_type: str, n_levels: int = 3) -> List[float]:
    closes = df["close"].values
    if len(closes) < 20:
        return []

    levels = []
    window = max(5, len(closes) // 10)

    for i in range(window, len(closes) - window):
        if level_type == "support":
            if closes[i] == min(closes[i - window:i + window + 1]):
                levels.append(round(float(closes[i]), 2))
        else:
            if closes[i] == max(closes[i - window:i + window + 1]):
                levels.append(round(float(closes[i]), 2))

    if not levels:
        if level_type == "support":
            levels = [round(float(np.percentile(closes, p)), 2) for p in [10, 25]]
        else:
            levels = [round(float(np.percentile(closes, p)), 2) for p in [75, 90]]

    unique = sorted(set(levels))
    if level_type == "support":
        return unique[:n_levels]
    else:
        return unique[-n_levels:]
