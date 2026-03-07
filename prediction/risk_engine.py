import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from prediction.features import fetch_ohlcv

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
RISK_FREE_RATE = 0.045

HISTORICAL_SCENARIOS = {
    "2008_financial_crisis": {
        "name": "2008 Financial Crisis",
        "description": "Global financial meltdown triggered by subprime mortgage collapse",
        "market_shock": -0.55,
        "sector_shocks": {
            "Technology": -0.50, "Financial Services": -0.75, "Consumer Cyclical": -0.55,
            "Healthcare": -0.30, "Energy": -0.60, "Industrials": -0.55,
            "Consumer Defensive": -0.25, "Communication Services": -0.45,
            "Real Estate": -0.65, "Utilities": -0.20, "Basic Materials": -0.50,
        },
        "volatility_multiplier": 3.5,
        "duration_months": 18,
    },
    "covid_crash_2020": {
        "name": "COVID-19 Crash (2020)",
        "description": "Pandemic-driven market crash with fastest bear market in history",
        "market_shock": -0.34,
        "sector_shocks": {
            "Technology": -0.20, "Financial Services": -0.40, "Consumer Cyclical": -0.50,
            "Healthcare": -0.15, "Energy": -0.60, "Industrials": -0.40,
            "Consumer Defensive": -0.15, "Communication Services": -0.25,
            "Real Estate": -0.30, "Utilities": -0.25, "Basic Materials": -0.35,
        },
        "volatility_multiplier": 4.0,
        "duration_months": 2,
    },
    "2022_tech_selloff": {
        "name": "2022 Tech Selloff",
        "description": "Rate hike cycle crushing growth/tech valuations",
        "market_shock": -0.25,
        "sector_shocks": {
            "Technology": -0.40, "Financial Services": -0.15, "Consumer Cyclical": -0.35,
            "Healthcare": -0.10, "Energy": 0.30, "Industrials": -0.15,
            "Consumer Defensive": -0.05, "Communication Services": -0.50,
            "Real Estate": -0.30, "Utilities": 0.05, "Basic Materials": -0.10,
        },
        "volatility_multiplier": 2.0,
        "duration_months": 12,
    },
    "rate_shock": {
        "name": "Rate Shock (+300bps)",
        "description": "Sudden 300 basis point rate increase scenario",
        "market_shock": -0.20,
        "sector_shocks": {
            "Technology": -0.30, "Financial Services": 0.05, "Consumer Cyclical": -0.25,
            "Healthcare": -0.10, "Energy": -0.05, "Industrials": -0.15,
            "Consumer Defensive": -0.08, "Communication Services": -0.25,
            "Real Estate": -0.40, "Utilities": -0.20, "Basic Materials": -0.10,
        },
        "volatility_multiplier": 2.5,
        "duration_months": 6,
    },
    "stagflation": {
        "name": "Stagflation Scenario",
        "description": "High inflation with economic stagnation, similar to 1970s",
        "market_shock": -0.30,
        "sector_shocks": {
            "Technology": -0.35, "Financial Services": -0.20, "Consumer Cyclical": -0.40,
            "Healthcare": -0.10, "Energy": 0.20, "Industrials": -0.25,
            "Consumer Defensive": -0.05, "Communication Services": -0.30,
            "Real Estate": -0.15, "Utilities": 0.05, "Basic Materials": 0.10,
        },
        "volatility_multiplier": 2.0,
        "duration_months": 24,
    },
}

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
    "VOO": "Other", "QQQ": "Technology", "SPY": "Other", "IWM": "Other",
}


def _get_sector(ticker: str) -> str:
    return SECTOR_MAP.get(ticker, "Other")


def _detect_regime(returns: np.ndarray, window: int = 63) -> Dict[str, Any]:
    if len(returns) < window * 2:
        return {"regime": "normal", "bull_prob": 0.33, "bear_prob": 0.33, "sideways_prob": 0.34, "drift_adj": 1.0, "vol_adj": 1.0}

    recent = returns[-window:]
    older = returns[:-window]

    recent_mean = np.mean(recent) * TRADING_DAYS
    recent_vol = np.std(recent) * np.sqrt(TRADING_DAYS)
    older_mean = np.mean(older) * TRADING_DAYS
    older_vol = np.std(older) * np.sqrt(TRADING_DAYS)

    rolling_mean = pd.Series(returns).rolling(window).mean().dropna().values
    rolling_vol = pd.Series(returns).rolling(window).std().dropna().values

    mean_trend = (rolling_mean[-1] - np.median(rolling_mean)) / (np.std(rolling_mean) + 1e-10)
    vol_trend = (rolling_vol[-1] - np.median(rolling_vol)) / (np.std(rolling_vol) + 1e-10)

    bull_score = max(0, mean_trend) + max(0, -vol_trend) * 0.5
    bear_score = max(0, -mean_trend) + max(0, vol_trend) * 0.5
    sideways_score = max(0, 1.0 - abs(mean_trend)) + max(0, 1.0 - abs(vol_trend)) * 0.5

    total = bull_score + bear_score + sideways_score + 1e-10
    bull_prob = bull_score / total
    bear_prob = bear_score / total
    sideways_prob = sideways_score / total

    if bull_prob > bear_prob and bull_prob > sideways_prob:
        regime = "bull"
        drift_adj = 1.2
        vol_adj = 0.85
    elif bear_prob > bull_prob and bear_prob > sideways_prob:
        regime = "bear"
        drift_adj = 0.6
        vol_adj = 1.4
    else:
        regime = "sideways"
        drift_adj = 0.9
        vol_adj = 1.1

    return {
        "regime": regime,
        "bull_prob": round(float(bull_prob), 3),
        "bear_prob": round(float(bear_prob), 3),
        "sideways_prob": round(float(sideways_prob), 3),
        "drift_adj": drift_adj,
        "vol_adj": vol_adj,
        "recent_annual_return": round(float(recent_mean * 100), 2),
        "recent_annual_vol": round(float(recent_vol * 100), 2),
    }


def _fit_student_t(returns: np.ndarray) -> Dict[str, float]:
    try:
        df_param, loc, scale = scipy_stats.t.fit(returns)
        df_param = max(2.1, min(df_param, 30.0))
    except Exception:
        df_param = 5.0
        loc = float(np.mean(returns))
        scale = float(np.std(returns))
    return {"df": float(df_param), "loc": float(loc), "scale": float(scale)}


def _cornish_fisher_var(returns: np.ndarray, confidence: float) -> float:
    z = scipy_stats.norm.ppf(1 - confidence)
    s = float(pd.Series(returns).skew())
    k = float(pd.Series(returns).kurtosis())

    z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * (k) / 24 - (2*z**3 - 5*z) * (s**2) / 36

    mu = np.mean(returns)
    sigma = np.std(returns)
    return float(mu + z_cf * sigma)


def _drawdown_analysis(portfolio_paths: np.ndarray) -> Dict[str, Any]:
    num_sims = portfolio_paths.shape[0]
    max_drawdowns = []
    max_drawdown_durations = []
    recovery_times = []

    for sim in range(num_sims):
        path = portfolio_paths[sim]
        running_max = np.maximum.accumulate(path)
        drawdowns = (path - running_max) / running_max
        max_dd = float(np.min(drawdowns))
        max_drawdowns.append(max_dd)

        in_dd = drawdowns < 0
        if np.any(in_dd):
            dd_changes = np.diff(in_dd.astype(int))
            starts = np.where(dd_changes == 1)[0]
            ends = np.where(dd_changes == -1)[0]

            if len(starts) > 0:
                if len(ends) == 0 or (len(ends) > 0 and ends[-1] < starts[-1]):
                    ends = np.append(ends, len(path) - 1)
                durations = []
                for s_idx in range(min(len(starts), len(ends))):
                    durations.append(int(ends[s_idx] - starts[s_idx]))
                if durations:
                    max_drawdown_durations.append(max(durations))
                    recovery_times.append(np.mean(durations))

    max_drawdowns = np.array(max_drawdowns)

    return {
        "expected_max_drawdown_pct": round(float(np.mean(max_drawdowns) * 100), 2),
        "median_max_drawdown_pct": round(float(np.median(max_drawdowns) * 100), 2),
        "worst_drawdown_pct": round(float(np.min(max_drawdowns) * 100), 2),
        "drawdown_95th_pct": round(float(np.percentile(max_drawdowns, 5) * 100), 2),
        "avg_recovery_days": round(float(np.mean(recovery_times)), 1) if recovery_times else None,
        "max_recovery_days": int(max(max_drawdown_durations)) if max_drawdown_durations else None,
        "prob_10pct_drawdown": round(float(np.sum(max_drawdowns < -0.10) / num_sims * 100), 1),
        "prob_20pct_drawdown": round(float(np.sum(max_drawdowns < -0.20) / num_sims * 100), 1),
    }


def _portfolio_sensitivities(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    total_value: float,
) -> Dict[str, Any]:
    port_returns = returns_df.values @ weights

    try:
        spy_df = fetch_ohlcv("SPY", days=730)
        if spy_df is not None and len(spy_df) > 60:
            spy_prices = spy_df.set_index("date")["close"]
            spy_returns = spy_prices.pct_change().dropna()
            common_idx = returns_df.index.intersection(spy_returns.index)
            if len(common_idx) > 30:
                spy_r = spy_returns.loc[common_idx].values
                port_r = (returns_df.loc[common_idx].values @ weights)

                cov_with_market = np.cov(port_r, spy_r)[0, 1]
                var_market = np.var(spy_r)
                beta = cov_with_market / var_market if var_market > 0 else 1.0
                correlation_to_market = float(np.corrcoef(port_r, spy_r)[0, 1])
            else:
                beta = 1.0
                correlation_to_market = 0.0
        else:
            beta = 1.0
            correlation_to_market = 0.0
    except Exception:
        beta = 1.0
        correlation_to_market = 0.0

    port_vol = float(np.std(port_returns) * np.sqrt(TRADING_DAYS))
    vega_proxy = total_value * port_vol * 0.01

    corr_matrix = returns_df.corr().values
    avg_corr = float((corr_matrix.sum() - np.trace(corr_matrix)) / (corr_matrix.size - len(corr_matrix)))

    corr_sensitivity = total_value * port_vol * avg_corr * 0.1

    return {
        "market_beta": round(float(beta), 3),
        "correlation_to_market": round(correlation_to_market, 3),
        "dollar_delta": round(float(beta * total_value / 100), 2),
        "vega_proxy": round(vega_proxy, 2),
        "avg_correlation": round(avg_corr, 3),
        "correlation_sensitivity": round(corr_sensitivity, 2),
    }


def run_monte_carlo_var(
    holdings: List[dict],
    real_prices: Dict[str, float],
    num_simulations: int = 10000,
    horizon_days: int = 21,
    confidence_levels: List[float] = None,
) -> Dict[str, Any]:
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    symbols = [h["ticker"] for h in holdings]
    price_data = {}
    for sym in symbols:
        df = fetch_ohlcv(sym, days=730)
        if df is not None and len(df) > 60:
            price_data[sym] = df.set_index("date")["close"]

    if len(price_data) < 1:
        return {"error": "Insufficient price data for simulation"}

    prices_df = pd.DataFrame(price_data).dropna()
    if len(prices_df) < 60:
        return {"error": "Not enough overlapping price history"}

    returns_df = prices_df.pct_change().dropna()

    total_value = 0
    weights = {}
    for h in holdings:
        t = h["ticker"]
        price = real_prices.get(t, h.get("avg_price", 100))
        mv = price * h["quantity"]
        total_value += mv
        if t in returns_df.columns:
            weights[t] = mv

    if not weights or total_value == 0:
        return {"error": "No valid positions for simulation"}

    w = np.array([weights.get(col, 0) for col in returns_df.columns])
    w = w / w.sum() if w.sum() > 0 else w

    mean_returns = returns_df.mean().values
    cov_matrix = returns_df.cov().values

    port_daily_returns = np.dot(returns_df.values, w)
    regime_info = _detect_regime(port_daily_returns)

    t_params = _fit_student_t(port_daily_returns)

    per_asset_t_params = {}
    for i, col in enumerate(returns_df.columns):
        per_asset_t_params[col] = _fit_student_t(returns_df[col].values)

    try:
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        cov_matrix += np.eye(len(cov_matrix)) * 1e-8
        L = np.linalg.cholesky(cov_matrix)

    adjusted_mean = mean_returns * regime_info["drift_adj"]
    vol_adj = regime_info["vol_adj"]

    np.random.seed(42)
    n_assets = len(w)
    avg_df = np.mean([per_asset_t_params[col]["df"] for col in returns_df.columns])
    avg_df = max(2.1, min(avg_df, 30.0))

    chi2_samples = np.random.chisquare(avg_df, size=(num_simulations, horizon_days))
    scaling = np.sqrt(avg_df / chi2_samples)

    Z_normal = np.random.standard_normal((num_simulations, horizon_days, n_assets))

    portfolio_paths = np.zeros((num_simulations, horizon_days + 1))
    portfolio_paths[:, 0] = total_value

    for sim in range(num_simulations):
        cumulative = 1.0
        for day in range(horizon_days):
            correlated = Z_normal[sim, day] @ L.T
            fat_tailed = correlated * scaling[sim, day] * vol_adj
            asset_returns = adjusted_mean + fat_tailed
            port_return = np.dot(w, asset_returns)
            cumulative *= (1 + port_return)
            portfolio_paths[sim, day + 1] = total_value * cumulative

    final_values = portfolio_paths[:, -1]
    pnl = final_values - total_value

    var_results = {}
    cvar_results = {}
    cf_var_results = {}
    for cl in confidence_levels:
        var_pct = np.percentile(pnl, (1 - cl) * 100)
        var_results[f"{int(cl*100)}%"] = {
            "value": round(float(var_pct), 2),
            "pct": round(float(var_pct / total_value * 100), 2),
        }
        losses_beyond = pnl[pnl <= var_pct]
        cvar_val = float(np.mean(losses_beyond)) if len(losses_beyond) > 0 else float(var_pct)
        cvar_results[f"{int(cl*100)}%"] = {
            "value": round(cvar_val, 2),
            "pct": round(cvar_val / total_value * 100, 2),
        }

        cf_daily_var = _cornish_fisher_var(port_daily_returns, cl)
        cf_horizon_var = cf_daily_var * np.sqrt(horizon_days) * total_value
        cf_var_results[f"{int(cl*100)}%"] = {
            "value": round(float(cf_horizon_var), 2),
            "pct": round(float(cf_horizon_var / total_value * 100), 2),
        }

    percentiles = [5, 10, 25, 50, 75, 90, 95]
    fan_chart = []
    for day in range(horizon_days + 1):
        day_values = portfolio_paths[:, day]
        point = {"day": day}
        for p in percentiles:
            point[f"p{p}"] = round(float(np.percentile(day_values, p)), 2)
        point["mean"] = round(float(np.mean(day_values)), 2)
        fan_chart.append(point)

    skewness = float(pd.Series(port_daily_returns).skew())
    kurtosis = float(pd.Series(port_daily_returns).kurtosis())

    risk_contributions = []
    port_vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
    marginal_risk = np.dot(cov_matrix, w) / port_vol if port_vol > 0 else np.zeros_like(w)
    component_risk = w * marginal_risk

    for i, col in enumerate(returns_df.columns):
        contrib_pct = float(component_risk[i] / port_vol * 100) if port_vol > 0 else 0
        risk_contributions.append({
            "ticker": col,
            "weight_pct": round(float(w[i] * 100), 1),
            "risk_contribution_pct": round(contrib_pct, 1),
            "marginal_var": round(float(marginal_risk[i] * total_value * 1.645 / np.sqrt(TRADING_DAYS) * np.sqrt(horizon_days)), 2),
        })

    drawdown_info = _drawdown_analysis(portfolio_paths)

    sensitivities = _portfolio_sensitivities(returns_df, w, total_value)

    stress_overlays = []
    for key, scenario in HISTORICAL_SCENARIOS.items():
        stressed_pnl = total_value * scenario["market_shock"]
        stress_overlays.append({
            "scenario": scenario["name"],
            "scenario_key": key,
            "estimated_loss": round(float(stressed_pnl), 2),
            "estimated_loss_pct": round(float(scenario["market_shock"] * 100), 1),
            "vol_multiplier": scenario["volatility_multiplier"],
        })

    expected_return = round(float(np.mean(pnl)), 2)
    prob_loss = round(float(np.sum(pnl < 0) / num_simulations * 100), 1)
    prob_gain_10 = round(float(np.sum(pnl > total_value * 0.10) / num_simulations * 100), 1)
    max_loss = round(float(np.min(pnl)), 2)
    max_gain = round(float(np.max(pnl)), 2)

    return {
        "portfolio_value": round(total_value, 2),
        "horizon_days": horizon_days,
        "num_simulations": num_simulations,
        "var": var_results,
        "cvar": cvar_results,
        "cornish_fisher_var": cf_var_results,
        "fan_chart": fan_chart,
        "risk_contributions": sorted(risk_contributions, key=lambda x: abs(x["risk_contribution_pct"]), reverse=True),
        "regime": regime_info,
        "distribution": {
            "type": "student_t",
            "degrees_of_freedom": round(t_params["df"], 2),
            "fat_tail_indicator": "heavy" if t_params["df"] < 5 else "moderate" if t_params["df"] < 10 else "light",
        },
        "drawdown_analysis": drawdown_info,
        "sensitivities": sensitivities,
        "stress_overlays": stress_overlays,
        "statistics": {
            "expected_pnl": expected_return,
            "expected_return_pct": round(expected_return / total_value * 100, 2),
            "probability_of_loss": prob_loss,
            "probability_of_10pct_gain": prob_gain_10,
            "max_simulated_loss": max_loss,
            "max_simulated_gain": max_gain,
            "skewness": round(skewness, 3),
            "kurtosis": round(kurtosis, 3),
            "annualized_volatility": round(float(port_vol * np.sqrt(TRADING_DAYS) * 100), 2),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_stress_test(
    holdings: List[dict],
    real_prices: Dict[str, float],
    scenarios: List[str] = None,
) -> Dict[str, Any]:
    if scenarios is None:
        scenarios = list(HISTORICAL_SCENARIOS.keys())

    total_value = 0
    positions = []
    for h in holdings:
        t = h["ticker"]
        price = real_prices.get(t, h.get("avg_price", 100))
        mv = price * h["quantity"]
        total_value += mv
        positions.append({
            "ticker": t,
            "sector": _get_sector(t),
            "market_value": mv,
            "weight": 0,
            "price": price,
            "quantity": h["quantity"],
        })

    for p in positions:
        p["weight"] = p["market_value"] / total_value if total_value > 0 else 0

    results = []
    for scenario_key in scenarios:
        scenario = HISTORICAL_SCENARIOS.get(scenario_key)
        if not scenario:
            continue

        scenario_pnl = 0
        position_impacts = []
        for pos in positions:
            sector_shock = scenario["sector_shocks"].get(pos["sector"], scenario["market_shock"])
            idio_factor = 1.0 + np.random.uniform(-0.1, 0.1)
            shock = sector_shock * idio_factor
            pnl = pos["market_value"] * shock
            scenario_pnl += pnl

            position_impacts.append({
                "ticker": pos["ticker"],
                "sector": pos["sector"],
                "current_value": round(pos["market_value"], 2),
                "shock_pct": round(shock * 100, 1),
                "pnl": round(pnl, 2),
                "stressed_value": round(pos["market_value"] + pnl, 2),
            })

        results.append({
            "scenario_key": scenario_key,
            "name": scenario["name"],
            "description": scenario["description"],
            "duration_months": scenario["duration_months"],
            "portfolio_pnl": round(scenario_pnl, 2),
            "portfolio_pnl_pct": round(scenario_pnl / total_value * 100, 2) if total_value > 0 else 0,
            "stressed_portfolio_value": round(total_value + scenario_pnl, 2),
            "volatility_multiplier": scenario["volatility_multiplier"],
            "position_impacts": sorted(position_impacts, key=lambda x: x["pnl"]),
            "worst_hit": min(position_impacts, key=lambda x: x["pnl"])["ticker"] if position_impacts else None,
            "best_relative": max(position_impacts, key=lambda x: x["shock_pct"])["ticker"] if position_impacts else None,
        })

    results.sort(key=lambda x: x["portfolio_pnl"])

    worst = results[0] if results else None
    best = results[-1] if results else None

    return {
        "portfolio_value": round(total_value, 2),
        "num_positions": len(positions),
        "scenarios": results,
        "summary": {
            "worst_scenario": worst["name"] if worst else None,
            "worst_loss_pct": worst["portfolio_pnl_pct"] if worst else 0,
            "best_scenario": best["name"] if best else None,
            "best_loss_pct": best["portfolio_pnl_pct"] if best else 0,
            "avg_loss_pct": round(float(np.mean([r["portfolio_pnl_pct"] for r in results])), 2) if results else 0,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
