import logging
from datetime import datetime, timezone
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from prediction.features import fetch_ohlcv

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
RISK_FREE_RATE = 0.045

SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "GOOGL": "Communication Services",
    "AMZN": "Consumer Cyclical", "META": "Communication Services", "TSLA": "Consumer Cyclical",
    "AMD": "Technology", "NFLX": "Communication Services", "CRM": "Technology",
    "JPM": "Financial Services", "V": "Financial Services", "MA": "Financial Services",
    "BAC": "Financial Services", "GS": "Financial Services", "UNH": "Healthcare",
    "JNJ": "Healthcare", "PFE": "Healthcare", "ABBV": "Healthcare", "LLY": "Healthcare",
    "XOM": "Energy", "CVX": "Energy", "DIS": "Communication Services",
    "KO": "Consumer Defensive", "PEP": "Consumer Defensive", "WMT": "Consumer Defensive",
    "HD": "Consumer Cyclical", "MCD": "Consumer Defensive", "COST": "Consumer Defensive",
    "VOO": "ETF", "QQQ": "Technology", "SPY": "ETF", "IWM": "ETF",
    "GLD": "Commodities", "TLT": "Bonds", "BND": "Bonds",
}


def _get_sector(ticker: str) -> str:
    return SECTOR_MAP.get(ticker, "Other")


def _compute_metrics(returns_df: pd.DataFrame, weights: np.ndarray) -> Dict[str, Any]:
    portfolio_returns = (returns_df * weights).sum(axis=1)

    annual_return = float(portfolio_returns.mean() * TRADING_DAYS)
    annual_vol = float(portfolio_returns.std() * np.sqrt(TRADING_DAYS))
    sharpe = round((annual_return - RISK_FREE_RATE) / annual_vol if annual_vol > 0 else 0, 3)

    neg_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = float(neg_returns.std() * np.sqrt(TRADING_DAYS)) if len(neg_returns) > 0 else annual_vol
    sortino = round((annual_return - RISK_FREE_RATE) / downside_std if downside_std > 0 else 0, 3)

    cumulative = (1 + portfolio_returns).cumprod()
    max_drawdown = float(((cumulative / cumulative.cummax()) - 1).min())

    spy_df = fetch_ohlcv("SPY", days=730)
    beta = 1.0
    if spy_df is not None and len(spy_df) > 50:
        spy_prices = spy_df.set_index("date")["close"]
        common_idx = returns_df.index.intersection(spy_prices.index)
        if len(common_idx) > 30:
            spy_ret = spy_prices.reindex(common_idx).pct_change().dropna()
            port_ret_aligned = portfolio_returns.reindex(spy_ret.index).dropna()
            common = port_ret_aligned.index.intersection(spy_ret.index)
            if len(common) > 20:
                cov = np.cov(port_ret_aligned.loc[common], spy_ret.loc[common])
                if cov[1, 1] > 0:
                    beta = round(float(cov[0, 1] / cov[1, 1]), 3)

    n = len(returns_df.columns)
    if n > 1:
        corr_matrix = returns_df.corr()
        avg_corr = (corr_matrix.sum().sum() - n) / (n * (n - 1))
        hhi = float(np.sum(weights ** 2))
        diversification_score = round(max(0, 1 - avg_corr), 3)
    else:
        hhi = 1.0
        diversification_score = 0.0

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "beta": beta,
        "max_drawdown": round(max_drawdown, 4),
        "annual_return": round(annual_return, 4),
        "annual_volatility": round(annual_vol, 4),
        "diversification_score": diversification_score,
        "hhi": round(hhi, 4),
    }


def simulate_portfolio(
    current_holdings: List[dict],
    changes: List[dict],
    real_prices: Dict[str, float],
) -> Dict[str, Any]:
    current_total = 0
    current_positions = {}
    for h in current_holdings:
        t = h["ticker"]
        price = real_prices.get(t, h.get("avg_price", 100.0))
        mv = price * h["quantity"]
        current_total += mv
        current_positions[t] = {
            "ticker": t,
            "quantity": h["quantity"],
            "avg_price": h.get("avg_price", price),
            "price": price,
            "market_value": mv,
            "sector": _get_sector(t),
        }

    sim_positions = {t: dict(p) for t, p in current_positions.items()}

    for change in changes:
        t = change["ticker"].upper()
        action = change.get("action", "add")
        qty = change.get("quantity", 0)
        price = real_prices.get(t, change.get("price", 0))

        if action == "add":
            if t in sim_positions:
                old = sim_positions[t]
                new_qty = old["quantity"] + qty
                new_avg = (old["avg_price"] * old["quantity"] + price * qty) / new_qty if new_qty > 0 else price
                sim_positions[t] = {
                    "ticker": t,
                    "quantity": new_qty,
                    "avg_price": round(new_avg, 2),
                    "price": price,
                    "market_value": price * new_qty,
                    "sector": _get_sector(t),
                }
            else:
                sim_positions[t] = {
                    "ticker": t,
                    "quantity": qty,
                    "avg_price": round(price, 2),
                    "price": price,
                    "market_value": price * qty,
                    "sector": _get_sector(t),
                }
        elif action == "remove":
            if t in sim_positions:
                old = sim_positions[t]
                new_qty = max(0, old["quantity"] - qty)
                if new_qty == 0:
                    del sim_positions[t]
                else:
                    sim_positions[t] = {
                        **old,
                        "quantity": new_qty,
                        "market_value": old["price"] * new_qty,
                    }
        elif action == "set":
            if qty <= 0:
                if t in sim_positions:
                    del sim_positions[t]
            else:
                sim_positions[t] = {
                    "ticker": t,
                    "quantity": qty,
                    "avg_price": round(price, 2),
                    "price": price,
                    "market_value": price * qty,
                    "sector": _get_sector(t),
                }

    sim_total = sum(p["market_value"] for p in sim_positions.values())

    for p in current_positions.values():
        p["weight"] = round(p["market_value"] / current_total, 4) if current_total > 0 else 0
    for p in sim_positions.values():
        p["weight"] = round(p["market_value"] / sim_total, 4) if sim_total > 0 else 0

    all_tickers = list(set(list(current_positions.keys()) + list(sim_positions.keys())))
    price_data = {}
    for sym in all_tickers:
        df = fetch_ohlcv(sym, days=730)
        if df is not None and len(df) > 50:
            price_data[sym] = df.set_index("date")["close"]

    if len(price_data) < 1:
        return {"error": "Insufficient price data for simulation"}

    prices_df = pd.DataFrame(price_data).dropna()
    if len(prices_df) < 30:
        return {"error": "Not enough overlapping price history"}

    returns_df = prices_df.pct_change().dropna()
    returns_df = returns_df[(returns_df.abs() < 0.5).all(axis=1)]

    current_syms_in_data = [s for s in current_positions if s in returns_df.columns]
    sim_syms_in_data = [s for s in sim_positions if s in returns_df.columns]

    if current_syms_in_data:
        current_weights = np.array([
            current_positions[s]["weight"] if s in current_positions else 0
            for s in returns_df.columns
        ])
        cw_sum = current_weights.sum()
        if cw_sum > 0:
            current_weights = current_weights / cw_sum
        current_metrics = _compute_metrics(returns_df, current_weights)
    else:
        current_metrics = {
            "sharpe": 0, "sortino": 0, "beta": 1.0, "max_drawdown": 0,
            "annual_return": 0, "annual_volatility": 0, "diversification_score": 0, "hhi": 1.0,
        }

    if sim_syms_in_data:
        sim_weights = np.array([
            sim_positions[s]["weight"] if s in sim_positions else 0
            for s in returns_df.columns
        ])
        sw_sum = sim_weights.sum()
        if sw_sum > 0:
            sim_weights = sim_weights / sw_sum
        sim_metrics = _compute_metrics(returns_df, sim_weights)
    else:
        sim_metrics = dict(current_metrics)

    deltas = {}
    for key in current_metrics:
        old_val = current_metrics[key]
        new_val = sim_metrics[key]
        deltas[key] = round(new_val - old_val, 4)

    current_sectors: Dict[str, float] = {}
    for p in current_positions.values():
        sec = p["sector"]
        current_sectors[sec] = current_sectors.get(sec, 0) + p["weight"]

    sim_sectors: Dict[str, float] = {}
    for p in sim_positions.values():
        sec = p["sector"]
        sim_sectors[sec] = sim_sectors.get(sec, 0) + p["weight"]

    all_sectors = sorted(set(list(current_sectors.keys()) + list(sim_sectors.keys())))
    sector_comparison = []
    for sec in all_sectors:
        cur_w = round(current_sectors.get(sec, 0), 4)
        sim_w = round(sim_sectors.get(sec, 0), 4)
        sector_comparison.append({
            "sector": sec,
            "current_weight": cur_w,
            "simulated_weight": sim_w,
            "delta": round(sim_w - cur_w, 4),
        })

    current_alloc = []
    for t, p in sorted(current_positions.items()):
        current_alloc.append({
            "ticker": t,
            "quantity": p["quantity"],
            "price": round(p["price"], 2),
            "market_value": round(p["market_value"], 2),
            "weight": p["weight"],
            "sector": p["sector"],
        })

    sim_alloc = []
    for t, p in sorted(sim_positions.items()):
        sim_alloc.append({
            "ticker": t,
            "quantity": p["quantity"],
            "price": round(p["price"], 2),
            "market_value": round(p["market_value"], 2),
            "weight": p["weight"],
            "sector": p["sector"],
        })

    return {
        "current": {
            "total_value": round(current_total, 2),
            "num_positions": len(current_positions),
            "allocation": current_alloc,
            "metrics": current_metrics,
            "sectors": {sec: round(w, 4) for sec, w in current_sectors.items()},
        },
        "simulated": {
            "total_value": round(sim_total, 2),
            "num_positions": len(sim_positions),
            "allocation": sim_alloc,
            "metrics": sim_metrics,
            "sectors": {sec: round(w, 4) for sec, w in sim_sectors.items()},
        },
        "deltas": deltas,
        "sector_comparison": sector_comparison,
        "changes_applied": changes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
