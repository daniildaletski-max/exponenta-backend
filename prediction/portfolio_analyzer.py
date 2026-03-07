import logging
from datetime import datetime, timezone
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from prediction.features import fetch_ohlcv
from prediction.markowitz import compute_efficient_frontier

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
RISK_FREE_RATE = 0.045


def analyze_portfolio_real(holdings: List[dict], real_prices: Dict[str, float] = None) -> Dict[str, Any]:
    if real_prices is None:
        real_prices = {}

    total_value = 0
    allocation = []
    for h in holdings:
        t = h["ticker"]
        price = real_prices.get(t, h.get("avg_price", 100.0))
        mv = price * h["quantity"]
        total_value += mv
        allocation.append({
            "ticker": t,
            "weight": 0,
            "current_price": round(price, 2),
            "market_value": round(mv, 2),
            "quantity": h["quantity"],
            "avg_price": h.get("avg_price", round(price * 0.9, 2)),
        })

    for a in allocation:
        a["weight"] = round(a["market_value"] / total_value, 4) if total_value > 0 else 0

    symbols = [h["ticker"] for h in holdings]
    price_data = {}
    for sym in symbols:
        df = fetch_ohlcv(sym, days=730)
        if df is not None and len(df) > 50:
            price_data[sym] = df.set_index("date")["close"]

    if len(price_data) < 1:
        return _fallback_analysis(allocation, total_value)

    prices_df = pd.DataFrame(price_data).dropna()
    if len(prices_df) < 30:
        return _fallback_analysis(allocation, total_value)

    returns_df = prices_df.pct_change().dropna()
    returns_df = returns_df[(returns_df.abs() < 0.5).all(axis=1)]

    weights = np.array([
        next((a["weight"] for a in allocation if a["ticker"] == sym), 0)
        for sym in returns_df.columns
    ])
    w_sum = weights.sum()
    if w_sum > 0:
        weights = weights / w_sum
    else:
        weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)

    portfolio_returns = (returns_df * weights).sum(axis=1)

    annual_return = float(portfolio_returns.mean() * TRADING_DAYS)
    annual_vol = float(portfolio_returns.std() * np.sqrt(TRADING_DAYS))
    sharpe = round((annual_return - RISK_FREE_RATE) / annual_vol if annual_vol > 0 else 0, 2)

    neg_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = float(neg_returns.std() * np.sqrt(TRADING_DAYS)) if len(neg_returns) > 0 else annual_vol
    sortino = round((annual_return - RISK_FREE_RATE) / downside_std if downside_std > 0 else 0, 2)

    cumulative = (1 + portfolio_returns).cumprod()
    max_drawdown = float(((cumulative / cumulative.cummax()) - 1).min())

    spy_df = fetch_ohlcv("SPY", days=730)
    beta = 1.0
    if spy_df is not None and len(spy_df) > 50:
        spy_prices = spy_df.set_index("date")["close"]
        common_idx = prices_df.index.intersection(spy_prices.index)
        if len(common_idx) > 30:
            spy_ret = spy_prices.reindex(common_idx).pct_change().dropna()
            port_ret_aligned = portfolio_returns.reindex(spy_ret.index).dropna()
            common = port_ret_aligned.index.intersection(spy_ret.index)
            if len(common) > 20:
                cov = np.cov(port_ret_aligned.loc[common], spy_ret.loc[common])
                if cov[1, 1] > 0:
                    beta = round(float(cov[0, 1] / cov[1, 1]), 2)

    var_95 = round(float(np.percentile(portfolio_returns, 5) * np.sqrt(TRADING_DAYS)) * -1, 4)
    cvar_95_daily = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)]
    cvar_95 = round(float(cvar_95_daily.mean() * np.sqrt(TRADING_DAYS)) * -1, 4) if len(cvar_95_daily) > 0 else var_95

    corr_matrix = returns_df.corr()
    n = len(corr_matrix)
    if n > 1:
        avg_corr = (corr_matrix.sum().sum() - n) / (n * (n - 1))
        diversification_score = round(max(0, 1 - avg_corr), 2)
    else:
        diversification_score = 0.0

    if annual_vol < 0.12:
        risk_level = "low"
    elif annual_vol < 0.20:
        risk_level = "moderate-low"
    elif annual_vol < 0.30:
        risk_level = "moderate"
    elif annual_vol < 0.40:
        risk_level = "moderate-high"
    else:
        risk_level = "high"

    rebalance = _compute_rebalance(allocation, returns_df, weights)
    suggestions = _compute_suggestions(allocation, returns_df, sharpe, beta, diversification_score, risk_level)

    return {
        "total_value": round(total_value, 2),
        "current_allocation": allocation,
        "risk_metrics": {
            "var_95": round(var_95, 4),
            "cvar_95": round(cvar_95, 4),
        },
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": round(max_drawdown, 4),
        "beta": beta,
        "annual_return": round(annual_return, 4),
        "annual_volatility": round(annual_vol, 4),
        "diversification_score": diversification_score,
        "risk_level": risk_level,
        "recommended_rebalance": rebalance,
        "suggestions": suggestions,
        "data_source": "real_historical_data",
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }


def _compute_rebalance(allocation: list, returns_df: pd.DataFrame, weights: np.ndarray) -> list:
    symbols = list(returns_df.columns)
    n = len(symbols)
    if n < 2:
        return []

    mean_returns = returns_df.mean() * TRADING_DAYS
    cov_matrix = returns_df.cov() * TRADING_DAYS

    individual_sharpe = {}
    for i, sym in enumerate(symbols):
        vol = float(np.sqrt(cov_matrix.iloc[i, i]))
        ret = float(mean_returns.iloc[i])
        individual_sharpe[sym] = (ret - RISK_FREE_RATE) / vol if vol > 0 else 0

    inv_vol = 1.0 / np.sqrt(np.diag(cov_matrix.values))
    rp_weights = inv_vol / inv_vol.sum()

    rebalance = []
    for i, sym in enumerate(symbols):
        current_w = float(weights[i]) if i < len(weights) else 0
        target_w = float(rp_weights[i])
        delta = target_w - current_w
        sym_sharpe = individual_sharpe.get(sym, 0)

        if sym_sharpe > 0.5 and delta < 0:
            target_w = current_w + abs(delta) * 0.3
            delta = target_w - current_w

        if abs(delta) < 0.02:
            action = "hold"
            conviction = round(0.5 + abs(sym_sharpe) * 0.1, 2)
            rationale = f"{sym} allocation is well-balanced (Sharpe: {sym_sharpe:.2f})"
        elif delta > 0:
            action = "increase"
            conviction = round(min(0.95, 0.5 + delta * 2 + max(0, sym_sharpe) * 0.1), 2)
            rationale = (
                f"{sym} is underweight by {abs(delta)*100:.1f}pp. "
                f"Risk-adjusted return (Sharpe: {sym_sharpe:.2f}) supports increasing allocation"
            )
        else:
            action = "decrease"
            conviction = round(min(0.95, 0.5 + abs(delta) * 2 + max(0, -sym_sharpe) * 0.1), 2)
            rationale = (
                f"{sym} is overweight by {abs(delta)*100:.1f}pp. "
                f"Trimming reduces concentration risk (Sharpe: {sym_sharpe:.2f})"
            )

        alloc_item = next((a for a in allocation if a["ticker"] == sym), None)
        rebalance.append({
            "ticker": sym,
            "action": action,
            "current_weight": round(current_w, 4),
            "target_weight": round(target_w, 4),
            "conviction": min(0.95, max(0.3, conviction)),
            "rationale": rationale,
        })

    rebalance.sort(key=lambda x: abs(x["target_weight"] - x["current_weight"]), reverse=True)
    return rebalance


def _compute_suggestions(allocation, returns_df, sharpe, beta, div_score, risk_level):
    suggestions = []
    tickers = [a["ticker"] for a in allocation]

    tech_tickers = {"AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMD", "AMZN", "CRM", "AVGO", "TSLA"}
    tech_count = sum(1 for t in tickers if t in tech_tickers)
    if tech_count > len(tickers) * 0.6:
        tech_weight = sum(a["weight"] for a in allocation if a["ticker"] in tech_tickers)
        suggestions.append(
            f"Portfolio is {tech_weight*100:.0f}% tech-concentrated. "
            f"Consider adding healthcare (XLV), utilities (XLU), or consumer staples for sector diversification"
        )

    if div_score < 0.4 and len(tickers) > 1:
        suggestions.append(
            f"Diversification score is low ({div_score:.2f}). "
            f"Holdings are highly correlated — consider adding uncorrelated assets like bonds (TLT) or commodities (GLD)"
        )

    if sharpe < 0.5:
        suggestions.append(
            f"Risk-adjusted return is weak (Sharpe: {sharpe:.2f}). "
            f"Consider replacing underperforming holdings with higher-Sharpe alternatives"
        )
    elif sharpe > 1.5:
        suggestions.append(
            f"Strong risk-adjusted performance (Sharpe: {sharpe:.2f}). Portfolio is well-optimized"
        )

    if beta > 1.3:
        suggestions.append(
            f"Portfolio beta is high ({beta:.2f}) — more volatile than market. "
            f"Consider adding low-beta defensive stocks or bond ETFs"
        )
    elif beta < 0.7:
        suggestions.append(
            f"Portfolio beta is low ({beta:.2f}) — consider adding growth exposure if seeking higher returns"
        )

    if len(tickers) < 5:
        suggestions.append(
            f"Only {len(tickers)} holdings — insufficient diversification. "
            f"Academic research suggests 15-30 holdings for optimal risk reduction"
        )

    max_weight = max(a["weight"] for a in allocation) if allocation else 0
    if max_weight > 0.4:
        heavy_ticker = next(a["ticker"] for a in allocation if a["weight"] == max_weight)
        suggestions.append(
            f"{heavy_ticker} represents {max_weight*100:.0f}% of portfolio — "
            f"consider trimming to reduce single-stock concentration risk"
        )

    if "moderate-high" in risk_level or risk_level == "high":
        suggestions.append(
            "Portfolio volatility is elevated — consider adding VIX hedging (VIXY) or protective puts"
        )

    return suggestions[:5]


def _fallback_analysis(allocation, total_value):
    return {
        "total_value": round(total_value, 2),
        "current_allocation": allocation,
        "risk_metrics": {"var_95": 0, "cvar_95": 0},
        "sharpe_ratio": 0,
        "sortino_ratio": 0,
        "max_drawdown": 0,
        "beta": 1.0,
        "annual_return": 0,
        "annual_volatility": 0,
        "diversification_score": 0,
        "risk_level": "unknown",
        "recommended_rebalance": [],
        "suggestions": ["Add more historical data to enable portfolio analysis"],
        "data_source": "insufficient_data",
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }
