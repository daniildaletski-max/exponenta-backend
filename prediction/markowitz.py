import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from prediction.features import fetch_ohlcv

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
FALLBACK_RISK_FREE_RATE = 0.045
NUM_FRONTIER_POINTS = 80


def _fetch_risk_free_rate() -> float:
    try:
        df = fetch_ohlcv("^IRX", days=30)
        if df is not None and len(df) > 0:
            latest = float(df.iloc[-1]["close"])
            if 0 < latest < 20:
                return latest / 100.0
    except Exception as e:
        logger.warning(f"Failed to fetch treasury yield: {e}")
    return FALLBACK_RISK_FREE_RATE


def _get_returns(symbols: List[str], days: int = 730) -> Optional[pd.DataFrame]:
    price_data = {}
    for sym in symbols:
        df = fetch_ohlcv(sym, days=days)
        if df is not None and len(df) > 50:
            price_data[sym] = df.set_index("date")["close"]

    if len(price_data) < 2:
        return None

    prices = pd.DataFrame(price_data).dropna()
    if len(prices) < 60:
        return None

    returns = prices.pct_change().dropna()
    returns = returns[(returns.abs() < 0.5).all(axis=1)]
    return returns


def _portfolio_stats(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    rf: float,
) -> Tuple[float, float, float]:
    port_ret = float(np.dot(weights, mean_returns))
    port_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
    sharpe = (port_ret - rf) / port_vol if port_vol > 0 else 0.0
    return port_ret, port_vol, sharpe


def _neg_sharpe(weights, mean_returns, cov_matrix, rf):
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -(ret - rf) / vol if vol > 0 else 0.0


def _portfolio_vol(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def _neg_diversification_ratio(weights, cov_matrix, asset_vols):
    weighted_avg_vol = np.dot(weights, asset_vols)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -(weighted_avg_vol / port_vol) if port_vol > 0 else 0.0


def _optimize_portfolio(
    objective,
    n: int,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    rf: float,
    constraints: Optional[Dict] = None,
    extra_args: tuple = (),
) -> np.ndarray:
    min_w = constraints.get("min_weight", 0.0) if constraints else 0.0
    max_w = constraints.get("max_weight", 1.0) if constraints else 1.0
    bounds = tuple((min_w, max_w) for _ in range(n))
    sum_constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    init = np.ones(n) / n

    if objective == "max_sharpe":
        res = minimize(
            _neg_sharpe, init,
            args=(mean_returns, cov_matrix, rf),
            method="SLSQP", bounds=bounds,
            constraints=[sum_constraint],
            options={"maxiter": 1000, "ftol": 1e-12},
        )
    elif objective == "min_vol":
        res = minimize(
            _portfolio_vol, init,
            args=(cov_matrix,),
            method="SLSQP", bounds=bounds,
            constraints=[sum_constraint],
            options={"maxiter": 1000, "ftol": 1e-12},
        )
    elif objective == "max_diversification":
        asset_vols = extra_args[0] if extra_args else np.sqrt(np.diag(cov_matrix))
        res = minimize(
            _neg_diversification_ratio, init,
            args=(cov_matrix, asset_vols),
            method="SLSQP", bounds=bounds,
            constraints=[sum_constraint],
            options={"maxiter": 1000, "ftol": 1e-12},
        )
    elif objective == "target_return":
        target = extra_args[0]
        ret_constraint = {"type": "eq", "fun": lambda w: np.dot(w, mean_returns) - target}
        res = minimize(
            _portfolio_vol, init,
            args=(cov_matrix,),
            method="SLSQP", bounds=bounds,
            constraints=[sum_constraint, ret_constraint],
            options={"maxiter": 1000, "ftol": 1e-12},
        )
    else:
        res = minimize(
            _neg_sharpe, init,
            args=(mean_returns, cov_matrix, rf),
            method="SLSQP", bounds=bounds,
            constraints=[sum_constraint],
            options={"maxiter": 1000, "ftol": 1e-12},
        )

    if res.success:
        w = res.x
        w = np.maximum(w, 0)
        w = w / w.sum()
        return w
    return init


def _compute_risk_parity(cov_matrix: np.ndarray, n: int) -> np.ndarray:
    def _risk_budget_obj(weights, cov):
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        marginal = np.dot(cov, weights) / port_vol
        risk_contrib = weights * marginal
        target_risk = port_vol / n
        return np.sum((risk_contrib - target_risk) ** 2)

    bounds = tuple((0.01, 1.0) for _ in range(n))
    init = np.ones(n) / n
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    res = minimize(
        _risk_budget_obj, init,
        args=(cov_matrix,),
        method="SLSQP", bounds=bounds,
        constraints=[cons],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if res.success:
        w = res.x
        w = np.maximum(w, 0)
        return w / w.sum()
    inv_vol = 1.0 / np.sqrt(np.diag(cov_matrix))
    return inv_vol / inv_vol.sum()


def _black_litterman(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    market_caps: Optional[np.ndarray],
    rf: float,
    views: Optional[Dict] = None,
    tau: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(mean_returns)

    if market_caps is not None and market_caps.sum() > 0:
        market_weights = market_caps / market_caps.sum()
    else:
        market_weights = np.ones(n) / n

    delta = (float(np.dot(market_weights, mean_returns)) - rf) / float(
        np.dot(market_weights.T, np.dot(cov_matrix, market_weights))
    ) if np.dot(market_weights.T, np.dot(cov_matrix, market_weights)) > 0 else 2.5

    pi = delta * np.dot(cov_matrix, market_weights)

    if views and len(views) > 0:
        k = len(views)
        P = np.zeros((k, n))
        Q = np.zeros(k)
        omega_diag = np.zeros(k)

        for idx, (asset_idx, view_return, confidence) in enumerate(views.values() if isinstance(views, dict) else views):
            P[idx, asset_idx] = 1.0
            Q[idx] = view_return
            omega_diag[idx] = (1.0 / confidence - 1.0) * tau * cov_matrix[asset_idx, asset_idx]

        Omega = np.diag(omega_diag)
        tau_cov = tau * cov_matrix
        tau_cov_inv = np.linalg.inv(tau_cov)
        P_T = P.T
        Omega_inv = np.linalg.inv(Omega)

        bl_mu = np.linalg.inv(tau_cov_inv + P_T @ Omega_inv @ P) @ (
            tau_cov_inv @ pi + P_T @ Omega_inv @ Q
        )
        bl_cov = np.linalg.inv(tau_cov_inv + P_T @ Omega_inv @ P)
    else:
        bl_mu = pi
        bl_cov = (1 + tau) * cov_matrix

    return bl_mu, bl_cov


def _compute_omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    excess = returns - threshold / TRADING_DAYS
    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()
    if losses == 0:
        return 10.0
    return float(gains / losses)


def _compute_information_ratio(
    port_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    active = port_returns - benchmark_returns
    te = active.std() * np.sqrt(TRADING_DAYS)
    if te == 0:
        return 0.0
    return float(active.mean() * TRADING_DAYS / te)


def compute_efficient_frontier(
    symbols: List[str],
    current_weights: Optional[Dict[str, float]] = None,
    num_points: int = NUM_FRONTIER_POINTS,
    constraints: Optional[Dict] = None,
) -> Dict[str, Any]:
    returns = _get_returns(symbols)
    if returns is None:
        return {"error": "Insufficient data for optimization"}

    valid_symbols = list(returns.columns)
    n = len(valid_symbols)

    rf = _fetch_risk_free_rate()
    mean_ret = returns.mean().values * TRADING_DAYS
    cov_mat = returns.cov().values * TRADING_DAYS
    asset_vols = np.sqrt(np.diag(cov_mat))

    ms_weights = _optimize_portfolio("max_sharpe", n, mean_ret, cov_mat, rf, constraints)
    mv_weights = _optimize_portfolio("min_vol", n, mean_ret, cov_mat, rf, constraints)
    eq_weights = np.ones(n) / n
    rp_weights = _compute_risk_parity(cov_mat, n)
    md_weights = _optimize_portfolio("max_diversification", n, mean_ret, cov_mat, rf, constraints, extra_args=(asset_vols,))

    bl_mu, bl_cov = _black_litterman(mean_ret, cov_mat, None, rf)
    bl_weights = _optimize_portfolio("max_sharpe", n, bl_mu, bl_cov, rf, constraints)

    frontier_points = []
    min_ret = float(np.min(mean_ret))
    max_ret = float(np.max(mean_ret))
    mv_ret = float(np.dot(mv_weights, mean_ret))
    frontier_min = max(min_ret * 0.5, mv_ret * 0.8)
    frontier_max = max_ret * 1.1
    targets = np.linspace(frontier_min, frontier_max, num_points)

    for target in targets:
        try:
            fw = _optimize_portfolio("target_return", n, mean_ret, cov_mat, rf, constraints, extra_args=(target,))
            f_ret, f_vol, f_sharpe = _portfolio_stats(fw, mean_ret, cov_mat, rf)
            frontier_points.append({
                "return": round(f_ret * 100, 2),
                "volatility": round(f_vol * 100, 2),
                "sharpe": round(f_sharpe, 3),
            })
        except Exception:
            pass

    def _format_portfolio(weights, name, mu=mean_ret, cov=cov_mat):
        ret, vol, sharpe = _portfolio_stats(weights, mu, cov, rf)
        port_daily = returns[valid_symbols].values @ weights
        port_series = pd.Series(port_daily, index=returns.index)
        neg_ret = port_series[port_series < 0]
        downside_vol = float(neg_ret.std() * np.sqrt(TRADING_DAYS)) if len(neg_ret) > 0 else 0
        sortino = (ret - rf) / downside_vol if downside_vol > 0 else 0
        cumulative = (1 + port_series).cumprod()
        max_dd = float(((cumulative / cumulative.cummax()) - 1).min())
        calmar = ret / abs(max_dd) if max_dd != 0 else 0
        omega = _compute_omega_ratio(port_series, rf)

        spy_returns = returns.get("SPY")
        info_ratio = 0.0
        if spy_returns is not None:
            info_ratio = _compute_information_ratio(port_series, spy_returns)
        elif len(valid_symbols) > 0:
            eq_daily = returns[valid_symbols].mean(axis=1)
            info_ratio = _compute_information_ratio(port_series, eq_daily)

        diversification_ratio = float(np.dot(weights, asset_vols) / vol) if vol > 0 else 1.0

        return {
            "name": name,
            "weights": {s: round(float(weights[i]), 4) for i, s in enumerate(valid_symbols)},
            "expected_return": round(ret * 100, 2),
            "volatility": round(vol * 100, 2),
            "sharpe": round(sharpe, 3),
            "sortino": round(sortino, 3),
            "max_drawdown": round(max_dd * 100, 2),
            "calmar": round(calmar, 3),
            "omega_ratio": round(omega, 3),
            "information_ratio": round(info_ratio, 3),
            "diversification_ratio": round(diversification_ratio, 3),
        }

    optimal_portfolios = {
        "max_sharpe": _format_portfolio(ms_weights, "Maximum Sharpe Ratio"),
        "min_volatility": _format_portfolio(mv_weights, "Minimum Volatility"),
        "equal_weight": _format_portfolio(eq_weights, "Equal Weight"),
        "risk_parity": _format_portfolio(rp_weights, "Risk Parity"),
        "max_diversification": _format_portfolio(md_weights, "Maximum Diversification"),
        "black_litterman": _format_portfolio(bl_weights, "Black-Litterman", bl_mu, (1 + 0.05) * cov_mat),
    }

    current_portfolio = None
    if current_weights:
        cw = np.array([current_weights.get(s, 0) for s in valid_symbols])
        cw_sum = cw.sum()
        if cw_sum > 0:
            cw = cw / cw_sum
            cur_ret, cur_vol, cur_sharpe = _portfolio_stats(cw, mean_ret, cov_mat, rf)
            port_daily = returns[valid_symbols].values @ cw
            port_series = pd.Series(port_daily, index=returns.index)
            neg_ret = port_series[port_series < 0]
            downside_vol = float(neg_ret.std() * np.sqrt(TRADING_DAYS)) if len(neg_ret) > 0 else 0
            cur_sortino = (cur_ret - rf) / downside_vol if downside_vol > 0 else 0
            cumulative = (1 + port_series).cumprod()
            cur_max_dd = float(((cumulative / cumulative.cummax()) - 1).min())
            cur_omega = _compute_omega_ratio(port_series, rf)

            current_portfolio = {
                "return": round(cur_ret * 100, 2),
                "volatility": round(cur_vol * 100, 2),
                "sharpe": round(cur_sharpe, 3),
                "sortino": round(cur_sortino, 3),
                "max_drawdown": round(cur_max_dd * 100, 2),
                "omega_ratio": round(cur_omega, 3),
                "weights": {s: round(float(cw[i]), 4) for i, s in enumerate(valid_symbols)},
            }

    corr_matrix = returns.corr()
    correlations = {}
    for i, s1 in enumerate(valid_symbols):
        for j, s2 in enumerate(valid_symbols):
            if i < j:
                correlations[f"{s1}/{s2}"] = round(float(corr_matrix.iloc[i, j]), 3)

    individual_stats = {}
    for i, sym in enumerate(valid_symbols):
        sym_ret = returns[sym]
        ann_ret = float(mean_ret[i])
        ann_vol = float(asset_vols[i])
        neg_ret = sym_ret[sym_ret < 0]
        downside_vol = float(neg_ret.std() * np.sqrt(TRADING_DAYS)) if len(neg_ret) > 0 else 0
        sortino = (ann_ret - rf) / downside_vol if downside_vol > 0 else 0
        cumulative = (1 + sym_ret).cumprod()
        max_dd = float(((cumulative / cumulative.cummax()) - 1).min())
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
        omega = _compute_omega_ratio(sym_ret, rf)

        individual_stats[sym] = {
            "annual_return": round(ann_ret * 100, 2),
            "annual_volatility": round(ann_vol * 100, 2),
            "sharpe": round((ann_ret - rf) / ann_vol if ann_vol > 0 else 0, 3),
            "sortino": round(sortino, 3),
            "max_drawdown": round(max_dd * 100, 2),
            "calmar": round(calmar, 3),
            "omega_ratio": round(omega, 3),
        }

    transitions = None
    if current_weights and current_portfolio:
        best_w = ms_weights
        transitions = []
        for i, sym in enumerate(valid_symbols):
            cur_w = current_weights.get(sym, 0)
            cur_sum = sum(current_weights.get(s, 0) for s in valid_symbols)
            if cur_sum > 0:
                cur_w = cur_w / cur_sum
            opt_w = float(best_w[i])
            delta = opt_w - cur_w
            if abs(delta) > 0.01:
                transitions.append({
                    "symbol": sym,
                    "current_weight": round(cur_w * 100, 2),
                    "optimal_weight": round(opt_w * 100, 2),
                    "delta": round(delta * 100, 2),
                    "action": "INCREASE" if delta > 0 else "DECREASE",
                })
        transitions.sort(key=lambda x: abs(x["delta"]), reverse=True)

    return {
        "symbols": valid_symbols,
        "optimal_portfolios": optimal_portfolios,
        "current_portfolio": current_portfolio,
        "transitions": transitions,
        "efficient_frontier": frontier_points,
        "correlations": correlations,
        "individual_stats": individual_stats,
        "risk_free_rate": round(rf * 100, 2),
        "optimization_method": "scipy_analytical",
        "constraints_applied": constraints or {"min_weight": 0.0, "max_weight": 1.0},
        "timestamp": datetime.now().isoformat(),
    }
