import logging
from datetime import datetime
from typing import Optional

import numpy as np

from prediction.features import (
    build_feature_matrix, GLOBAL_UNIVERSE, ALL_SYMBOLS, get_feature_cols
)
from prediction.engine import get_engine

logger = logging.getLogger(__name__)


def compute_momentum_score(df) -> float:
    if df is None or len(df) < 50:
        return 50.0
    close = df["close"].values
    scores = []
    if len(close) > 5:
        ret5 = (close[-1] - close[-5]) / close[-5]
        scores.append(50 + ret5 * 500)
    if len(close) > 20:
        ret20 = (close[-1] - close[-20]) / close[-20]
        scores.append(50 + ret20 * 200)
    if len(close) > 60:
        ret60 = (close[-1] - close[-60]) / close[-60]
        scores.append(50 + ret60 * 100)
    if "rsi" in df.columns:
        rsi = df["rsi"].iloc[-1]
        if not np.isnan(rsi):
            if rsi < 30:
                scores.append(75)
            elif rsi > 70:
                scores.append(25)
            else:
                scores.append(50 + (rsi - 50))
    return float(np.clip(np.mean(scores), 0, 100)) if scores else 50.0


def compute_liquidity_score(df) -> float:
    if df is None or len(df) < 20:
        return 50.0
    vol = df["volume"].values
    avg_vol = np.mean(vol[-20:])
    recent_vol = np.mean(vol[-5:])
    rel_vol = recent_vol / (avg_vol + 1)
    abs_score = min(avg_vol / 1e6, 100)
    return float(np.clip(abs_score * 0.5 + rel_vol * 25 + 25, 0, 100))


def compute_volatility_score(df) -> float:
    if df is None or len(df) < 20:
        return 50.0
    returns = df["close"].pct_change().dropna().values[-20:]
    if len(returns) < 5:
        return 50.0
    vol = np.std(returns) * np.sqrt(252)
    if 0.15 <= vol <= 0.40:
        return 80.0
    elif vol < 0.15:
        return 40.0 + vol * 200
    else:
        return max(20, 80 - (vol - 0.40) * 100)


def compute_trend_strength(df) -> float:
    if df is None or len(df) < 50:
        return 50.0
    score = 50.0
    if "adx" in df.columns:
        adx = df["adx"].iloc[-1]
        if not np.isnan(adx):
            if adx > 25:
                score += 15
            if adx > 40:
                score += 10
    close = df["close"].iloc[-1]
    for col in ["sma_20", "sma_50", "sma_200"]:
        if col in df.columns:
            ma = df[col].iloc[-1]
            if not np.isnan(ma) and close > ma:
                score += 5
    if "macd_hist" in df.columns:
        macd_h = df["macd_hist"].iloc[-1]
        if not np.isnan(macd_h) and macd_h > 0:
            score += 10
    return float(np.clip(score, 0, 100))


def scan_asset(symbol: str, use_ml: bool = False) -> Optional[dict]:
    try:
        df = build_feature_matrix(symbol, days=730)
        if df is None or len(df) < 50:
            return None

        ml_score = 50.0
        ml_confidence = 50.0
        ml_direction = "neutral"
        ml_probability = 0.5
        shap_factors = []

        if use_ml:
            try:
                engine = get_engine()
                pred = engine.predict(symbol)
                if "error" not in pred:
                    ml_confidence = pred["confidence"]
                    ml_probability = pred["probability"]
                    ml_direction = pred["direction"]
                    ml_score = ml_confidence if pred["direction"] == "bullish" else (100 - ml_confidence)
                    shap_factors = pred.get("shap_factors", [])
            except Exception as e:
                logger.debug(f"ML prediction failed for {symbol}: {e}")

        momentum = compute_momentum_score(df)
        liquidity = compute_liquidity_score(df)
        volatility = compute_volatility_score(df)
        trend_strength = compute_trend_strength(df)

        close = float(df["close"].iloc[-1])
        prev_close = float(df["close"].iloc[-2]) if len(df) > 1 else close
        change_pct = (close - prev_close) / prev_close * 100
        ret_5d = float((close - df["close"].iloc[-5]) / df["close"].iloc[-5] * 100) if len(df) > 5 else 0
        ret_20d = float((close - df["close"].iloc[-20]) / df["close"].iloc[-20] * 100) if len(df) > 20 else 0

        composite = (
            ml_score * 0.35 +
            momentum * 0.20 +
            trend_strength * 0.15 +
            volatility * 0.10 +
            liquidity * 0.10 +
            50 * 0.10
        )
        composite = float(np.clip(composite, 0, 100))

        if composite >= 75:
            signal = "STRONG_BUY"
        elif composite >= 60:
            signal = "BUY"
        elif composite >= 40:
            signal = "HOLD"
        elif composite >= 25:
            signal = "SELL"
        else:
            signal = "STRONG_SELL"

        ann_vol = float(np.std(df["close"].pct_change().dropna().values[-60:]) * np.sqrt(252))
        risk_level = "LOW" if ann_vol < 0.20 else "MEDIUM" if ann_vol < 0.40 else "HIGH"

        technicals = {}
        for ind in ["rsi", "adx", "macd_hist"]:
            if ind in df.columns:
                val = df[ind].iloc[-1]
                technicals[ind] = round(float(val), 2) if not np.isnan(val) else None

        return {
            "symbol": symbol,
            "price": round(close, 2),
            "change_pct": round(change_pct, 2),
            "ret_5d": round(ret_5d, 2),
            "ret_20d": round(ret_20d, 2),
            "signal": signal,
            "composite_score": round(composite, 1),
            "risk_level": risk_level,
            "confidence": round(ml_confidence, 1),
            "scores": {
                "ml_signal": round(ml_score, 1),
                "momentum": round(momentum, 1),
                "trend_strength": round(trend_strength, 1),
                "volatility": round(volatility, 1),
                "liquidity": round(liquidity, 1),
            },
            "ml_prediction": {
                "direction": ml_direction,
                "probability": round(ml_probability, 3),
                "confidence": round(ml_confidence, 1),
            },
            "technicals": technicals,
            "shap_factors": shap_factors[:5],
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Scan failed for {symbol}: {e}")
        return None


SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "GOOGL": "Technology",
    "META": "Technology", "AMD": "Technology", "AVGO": "Technology", "CRM": "Technology",
    "ORCL": "Technology", "ADBE": "Technology", "INTC": "Technology", "QCOM": "Technology",
    "TSLA": "Consumer Discretionary", "AMZN": "Consumer Discretionary", "NFLX": "Communication",
    "DIS": "Communication", "JPM": "Financials", "V": "Financials", "MA": "Financials",
    "GS": "Financials", "BAC": "Financials", "WMT": "Consumer Staples", "PG": "Consumer Staples",
    "KO": "Consumer Staples", "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
    "LLY": "Healthcare", "XOM": "Energy", "CVX": "Energy", "BA": "Industrials",
    "CAT": "Industrials", "BTC-USD": "Crypto", "ETH-USD": "Crypto", "SOL-USD": "Crypto",
    "SPY": "ETF", "QQQ": "ETF", "VOO": "ETF", "IWM": "ETF",
}

CAP_MAP = {
    "AAPL": "Mega Cap", "MSFT": "Mega Cap", "NVDA": "Mega Cap", "GOOGL": "Mega Cap",
    "AMZN": "Mega Cap", "META": "Mega Cap", "TSLA": "Large Cap", "JPM": "Large Cap",
    "V": "Large Cap", "AMD": "Large Cap", "NFLX": "Large Cap",
}


def scan_universe(
    universe_keys: list = None,
    max_assets: int = 20,
    use_ml: bool = False,
    min_score: float = 0,
    signal_filter: str = None,
    sector_filter: str = None,
) -> dict:
    if universe_keys is None:
        universe_keys = ["us_mega_cap"]

    symbols = []
    for key in universe_keys:
        symbols.extend(GLOBAL_UNIVERSE.get(key, []))
    symbols = list(set(symbols))[:max_assets]

    results = []
    for sym in symbols:
        result = scan_asset(sym, use_ml=use_ml)
        if result is not None:
            result["sector"] = SECTOR_MAP.get(sym, "Other")
            result["market_cap_category"] = CAP_MAP.get(sym, "Large Cap")

            rsi = result["technicals"].get("rsi")
            macd_h = result["technicals"].get("macd_hist")
            adx = result["technicals"].get("adx")
            setup_parts = []
            if rsi is not None:
                if rsi < 30:
                    setup_parts.append("RSI oversold")
                elif rsi > 70:
                    setup_parts.append("RSI overbought")
            if macd_h is not None:
                if macd_h > 0:
                    setup_parts.append("MACD bullish")
                else:
                    setup_parts.append("MACD bearish")
            if adx is not None and adx > 25:
                setup_parts.append("Strong trend")
            result["technical_setup"] = " | ".join(setup_parts) if setup_parts else "Neutral"

            price = result["price"]
            ann_vol = 0.25
            try:
                df = build_feature_matrix(sym, days=100)
                if df is not None and len(df) > 20:
                    rets = df["close"].pct_change().dropna().values[-20:]
                    ann_vol = float(np.std(rets) * np.sqrt(252))
            except Exception:
                pass
            result["entry_price"] = round(price * 0.98, 2)
            result["stop_loss"] = round(price * (1 - ann_vol * 0.5), 2)
            result["take_profit"] = round(price * (1 + ann_vol * 1.0), 2)

            results.append(result)

    if min_score > 0:
        results = [r for r in results if r["composite_score"] >= min_score]

    if signal_filter:
        allowed = [s.strip().upper() for s in signal_filter.split(",")]
        results = [r for r in results if r["signal"] in allowed]

    if sector_filter:
        allowed_sectors = [s.strip() for s in sector_filter.split(",")]
        results = [r for r in results if r.get("sector") in allowed_sectors]

    results.sort(key=lambda x: x["composite_score"], reverse=True)

    signals = [r["signal"] for r in results]
    scores = [r["composite_score"] for r in results]
    bullish = sum(1 for s in signals if s in ["STRONG_BUY", "BUY"])
    bearish = sum(1 for s in signals if s in ["STRONG_SELL", "SELL"])

    if bullish > bearish * 1.5:
        regime = "BULLISH"
    elif bearish > bullish * 1.5:
        regime = "BEARISH"
    else:
        regime = "MIXED"

    sector_dict: dict = {}
    for r in results:
        sec = r.get("sector", "Other")
        if sec not in sector_dict:
            sector_dict[sec] = {"count": 0, "avg_score": 0, "signals": []}
        sector_dict[sec]["count"] += 1
        sector_dict[sec]["avg_score"] += r["composite_score"]
        sector_dict[sec]["signals"].append(r["signal"])

    sector_summary = []
    for sector_name, sec in sector_dict.items():
        if sec["count"] > 0:
            sec["avg_score"] = round(sec["avg_score"] / sec["count"], 1)
        bullish_in = sum(1 for s in sec["signals"] if s in ["STRONG_BUY", "BUY"])
        top_signal = max(set(sec["signals"]), key=sec["signals"].count) if sec["signals"] else "HOLD"
        sector_summary.append({
            "sector": sector_name,
            "count": sec["count"],
            "avg_score": sec["avg_score"],
            "top_signal": top_signal,
            "bullish_pct": round(bullish_in / sec["count"] * 100, 1) if sec["count"] > 0 else 0,
        })

    return {
        "opportunities": results,
        "market_summary": {
            "regime": regime,
            "avg_score": round(float(np.mean(scores)), 1) if scores else 50,
            "signal_distribution": {
                "strong_buy": sum(1 for s in signals if s == "STRONG_BUY"),
                "buy": sum(1 for s in signals if s == "BUY"),
                "hold": sum(1 for s in signals if s == "HOLD"),
                "sell": sum(1 for s in signals if s == "SELL"),
                "strong_sell": sum(1 for s in signals if s == "STRONG_SELL"),
            },
            "bullish_pct": round(bullish / len(signals) * 100, 1) if signals else 0,
            "total_scanned": len(results),
        },
        "sector_summary": sector_summary,
        "timestamp": datetime.now().isoformat(),
    }
