import logging
from datetime import datetime, timezone
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from prediction.features import fetch_ohlcv

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
RISK_FREE_RATE = 0.045

FACTOR_ETFS = {
    "Market": "SPY",
    "Size": "IWM",
    "Momentum": "MTUM",
    "Value": "IVE",
    "Quality": "QUAL",
}

SECTOR_ETFS = {
    "Technology": "XLK",
    "Finance": "XLF",
    "Healthcare": "XLV",
    "Consumer": "XLY",
    "Energy": "XLE",
    "Industrial": "XLI",
    "Communication": "XLC",
    "Automotive": "XLY",
    "Entertainment": "XLC",
    "ETF": "SPY",
    "Crypto": "SPY",
}


def _fetch_factor_returns(days: int = 365) -> pd.DataFrame:
    factor_prices = {}
    for name, etf in FACTOR_ETFS.items():
        df = fetch_ohlcv(etf, days=days + 60)
        if df is not None and len(df) > 30:
            factor_prices[name] = df.set_index("date")["close"]

    if not factor_prices:
        return pd.DataFrame()

    prices_df = pd.DataFrame(factor_prices).dropna()
    returns_df = prices_df.pct_change().dropna()
    return returns_df


def _compute_factor_exposures(
    holding_returns: pd.Series,
    factor_returns: pd.DataFrame,
) -> Dict[str, float]:
    aligned = pd.concat([holding_returns, factor_returns], axis=1).dropna()
    if len(aligned) < 30:
        return {}

    y = aligned.iloc[:, 0].values
    X = aligned.iloc[:, 1:].values
    ones = np.ones((X.shape[0], 1))
    X_aug = np.hstack([ones, X])

    try:
        betas = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {}

    exposures = {"Alpha": float(betas[0]) * TRADING_DAYS}
    for i, name in enumerate(factor_returns.columns):
        exposures[name] = float(betas[i + 1])

    return exposures


def _compute_rolling_info_ratio(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    window: int = 126,
) -> List[Dict[str, Any]]:
    excess = portfolio_returns - benchmark_returns
    rolling_ir = []
    for i in range(window, len(excess)):
        chunk = excess[i - window : i]
        mean_e = np.mean(chunk)
        std_e = np.std(chunk, ddof=1) if np.std(chunk, ddof=1) > 0 else 1e-9
        ir = float(mean_e / std_e * np.sqrt(TRADING_DAYS))
        rolling_ir.append({"index": i, "info_ratio": round(ir, 3)})
    return rolling_ir


def compute_alpha_attribution(
    holdings: List[dict],
    real_prices: Dict[str, float],
) -> Dict[str, Any]:
    if not holdings:
        return {"error": "No holdings provided"}

    total_value = 0.0
    for h in holdings:
        price = real_prices.get(h["ticker"], h.get("avg_price", 100.0))
        total_value += price * h["quantity"]

    if total_value <= 0:
        return {"error": "Portfolio value is zero"}

    weights = {}
    for h in holdings:
        price = real_prices.get(h["ticker"], h.get("avg_price", 100.0))
        weights[h["ticker"]] = (price * h["quantity"]) / total_value

    days = 365
    holding_prices = {}
    for h in holdings:
        df = fetch_ohlcv(h["ticker"], days=days + 60)
        if df is not None and len(df) > 30:
            holding_prices[h["ticker"]] = df.set_index("date")["close"]

    if not holding_prices:
        return {"error": "Could not fetch price history for any holdings"}

    prices_df = pd.DataFrame(holding_prices).dropna()
    if len(prices_df) < 30:
        return {"error": "Insufficient overlapping price data"}

    returns_df = prices_df.pct_change().dropna()

    w_arr = np.array([weights.get(col, 0) for col in returns_df.columns])
    if w_arr.sum() > 0:
        w_arr = w_arr / w_arr.sum()

    portfolio_returns = (returns_df.values * w_arr).sum(axis=1)

    spy_df = fetch_ohlcv("SPY", days=days + 60)
    if spy_df is not None and len(spy_df) > 30:
        spy_series = spy_df.set_index("date")["close"]
        spy_returns_full = spy_series.pct_change().dropna()
        spy_returns = spy_returns_full.reindex(returns_df.index).dropna()
    else:
        spy_returns = pd.Series(np.zeros(len(portfolio_returns)), index=returns_df.index)

    min_len = min(len(portfolio_returns), len(spy_returns))
    portfolio_returns = portfolio_returns[-min_len:]
    spy_returns_arr = spy_returns.values[-min_len:]

    factor_returns = _fetch_factor_returns(days=days)

    factor_decomposition = {}
    if len(factor_returns) > 30:
        port_series = pd.Series(portfolio_returns, index=returns_df.index[-min_len:])
        factor_aligned = factor_returns.reindex(port_series.index).dropna()
        port_aligned = port_series.reindex(factor_aligned.index).dropna()
        factor_aligned = factor_aligned.reindex(port_aligned.index)

        if len(port_aligned) > 30:
            y = port_aligned.values
            X = factor_aligned.values
            ones = np.ones((X.shape[0], 1))
            X_aug = np.hstack([ones, X])

            try:
                betas = np.linalg.lstsq(X_aug, y, rcond=None)[0]
                alpha_daily = betas[0]
                factor_betas = betas[1:]

                total_return_ann = float(np.mean(y) * TRADING_DAYS)
                alpha_ann = float(alpha_daily * TRADING_DAYS)

                factor_contributions = {}
                for i, name in enumerate(factor_aligned.columns):
                    contrib = float(factor_betas[i] * np.mean(factor_aligned.iloc[:, i].values) * TRADING_DAYS)
                    factor_contributions[name] = round(contrib * 100, 2)

                residual = y - X_aug @ betas
                r_squared = 1 - np.var(residual) / np.var(y) if np.var(y) > 0 else 0

                factor_decomposition = {
                    "alpha_annualized_pct": round(alpha_ann * 100, 2),
                    "factor_contributions": factor_contributions,
                    "total_return_pct": round(total_return_ann * 100, 2),
                    "r_squared": round(float(r_squared), 3),
                    "factor_exposures": {
                        name: round(float(factor_betas[i]), 3)
                        for i, name in enumerate(factor_aligned.columns)
                    },
                }
            except Exception as e:
                logger.warning(f"Factor regression failed: {e}")

    if not factor_decomposition:
        excess = portfolio_returns - spy_returns_arr
        alpha_ann = float(np.mean(excess) * TRADING_DAYS)
        beta = float(
            np.cov(portfolio_returns, spy_returns_arr)[0, 1]
            / np.var(spy_returns_arr)
        ) if np.var(spy_returns_arr) > 0 else 1.0

        factor_decomposition = {
            "alpha_annualized_pct": round(alpha_ann * 100, 2),
            "factor_contributions": {
                "Market": round(float(np.mean(spy_returns_arr) * beta * TRADING_DAYS * 100), 2),
            },
            "total_return_pct": round(float(np.mean(portfolio_returns) * TRADING_DAYS * 100), 2),
            "r_squared": 0.0,
            "factor_exposures": {"Market": round(beta, 3)},
        }

    holding_contributions = []
    for col_idx, col in enumerate(returns_df.columns):
        col_returns = returns_df[col].values[-min_len:]
        holding_ret = float(np.mean(col_returns) * TRADING_DAYS)
        spy_mean = float(np.mean(spy_returns_arr))
        beta_h = float(
            np.cov(col_returns, spy_returns_arr)[0, 1] / np.var(spy_returns_arr)
        ) if np.var(spy_returns_arr) > 0 else 1.0
        alpha_h = holding_ret - (RISK_FREE_RATE + beta_h * (spy_mean * TRADING_DAYS - RISK_FREE_RATE))
        weight = float(w_arr[col_idx])

        holding_contributions.append({
            "ticker": col,
            "weight_pct": round(weight * 100, 2),
            "total_return_pct": round(holding_ret * 100, 2),
            "alpha_contribution_pct": round(alpha_h * weight * 100, 2),
            "beta": round(beta_h, 3),
            "alpha_pct": round(alpha_h * 100, 2),
        })

    holding_contributions.sort(key=lambda x: x["alpha_contribution_pct"], reverse=True)

    rolling_ir = _compute_rolling_info_ratio(portfolio_returns, spy_returns_arr)

    current_ir = rolling_ir[-1]["info_ratio"] if rolling_ir else 0
    if current_ir > 0.5:
        skill_score = min(100, int(50 + current_ir * 30))
    elif current_ir > 0:
        skill_score = int(30 + current_ir * 40)
    else:
        skill_score = max(0, int(30 + current_ir * 30))

    port_cumret = float((np.prod(1 + portfolio_returns) - 1) * 100)
    spy_cumret = float((np.prod(1 + spy_returns_arr) - 1) * 100)
    qqq_df = fetch_ohlcv("QQQ", days=days + 60)
    qqq_cumret = 0.0
    if qqq_df is not None and len(qqq_df) > 30:
        qqq_series = qqq_df.set_index("date")["close"]
        qqq_rets = qqq_series.pct_change().dropna().reindex(returns_df.index[-min_len:]).dropna()
        if len(qqq_rets) > 10:
            qqq_cumret = float((np.prod(1 + qqq_rets.values) - 1) * 100)

    port_vol = float(np.std(portfolio_returns, ddof=1) * np.sqrt(TRADING_DAYS))
    spy_vol = float(np.std(spy_returns_arr, ddof=1) * np.sqrt(TRADING_DAYS)) if len(spy_returns_arr) > 1 else 0

    port_sharpe = (
        (np.mean(portfolio_returns) * TRADING_DAYS - RISK_FREE_RATE)
        / (port_vol if port_vol > 0 else 1)
    )
    spy_sharpe = (
        (np.mean(spy_returns_arr) * TRADING_DAYS - RISK_FREE_RATE)
        / (spy_vol if spy_vol > 0 else 1)
    )

    alpha_pct = factor_decomposition.get("alpha_annualized_pct", 0)
    market_beta = factor_decomposition.get("factor_exposures", {}).get("Market", 1.0)
    if abs(alpha_pct) > 5 and skill_score > 60:
        edge_summary = f"Strong stock selection alpha of {alpha_pct:+.1f}% with consistent Information Ratio of {current_ir:.2f}"
    elif abs(alpha_pct) > 2:
        edge_summary = f"Moderate alpha of {alpha_pct:+.1f}% — returns partly driven by factor exposure (beta={market_beta:.2f})"
    elif market_beta > 1.2:
        edge_summary = f"Returns primarily from market beta ({market_beta:.2f}). Alpha is minimal at {alpha_pct:+.1f}%"
    else:
        edge_summary = f"Portfolio tracks the market closely. Alpha is {alpha_pct:+.1f}% with beta of {market_beta:.2f}"

    alpha_beta_history = []
    window = 63
    for i in range(window, min_len):
        chunk_port = portfolio_returns[i - window : i]
        chunk_spy = spy_returns_arr[i - window : i]
        var_spy = np.var(chunk_spy)
        beta_t = float(np.cov(chunk_port, chunk_spy)[0, 1] / var_spy) if var_spy > 0 else 1.0
        alpha_t = float((np.mean(chunk_port) - beta_t * np.mean(chunk_spy)) * TRADING_DAYS)
        alpha_beta_history.append({
            "index": i,
            "alpha_pct": round(alpha_t * 100, 2),
            "beta": round(beta_t, 3),
        })

    return {
        "factor_decomposition": factor_decomposition,
        "holding_contributions": holding_contributions,
        "skill_score": skill_score,
        "information_ratio": round(current_ir, 3),
        "rolling_ir": rolling_ir[-60:] if len(rolling_ir) > 60 else rolling_ir,
        "alpha_beta_history": alpha_beta_history[-60:] if len(alpha_beta_history) > 60 else alpha_beta_history,
        "benchmarks": {
            "portfolio_return_pct": round(port_cumret, 2),
            "spy_return_pct": round(spy_cumret, 2),
            "qqq_return_pct": round(qqq_cumret, 2),
            "risk_free_pct": round(RISK_FREE_RATE * 100, 2),
            "portfolio_sharpe": round(float(port_sharpe), 3),
            "spy_sharpe": round(float(spy_sharpe), 3),
            "portfolio_volatility_pct": round(port_vol * 100, 2),
            "spy_volatility_pct": round(spy_vol * 100, 2),
        },
        "edge_summary": edge_summary,
        "total_value": round(total_value, 2),
        "num_holdings": len(holdings),
        "data_points": min_len,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
