import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from prediction.features import fetch_ohlcv
from prediction.news_feed import get_earnings_calendar_sync as get_earnings_calendar

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
RISK_FREE_RATE = 0.045

SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Technology", "META": "Technology", "AMD": "Technology",
    "AVGO": "Technology", "CRM": "Technology", "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary", "NFLX": "Communication Services",
    "DIS": "Communication Services", "JPM": "Financials", "V": "Financials",
    "MA": "Financials", "UNH": "Healthcare", "JNJ": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "LLY": "Healthcare",
    "XOM": "Energy", "WMT": "Consumer Staples", "PG": "Consumer Staples",
    "KO": "Consumer Staples", "PEP": "Consumer Staples", "HD": "Consumer Discretionary",
    "BA": "Industrials", "VOO": "ETF", "SPY": "ETF", "QQQ": "ETF",
    "BTC-USD": "Crypto", "ETH-USD": "Crypto",
}

HEDGE_CANDIDATES = {
    "TLT": {"name": "iShares 20+ Year Treasury Bond", "type": "bonds", "typical_beta": -0.3},
    "GLD": {"name": "SPDR Gold Shares", "type": "commodity", "typical_beta": 0.05},
    "VXX": {"name": "iPath Series B S&P 500 VIX", "type": "volatility", "typical_beta": -3.0},
    "XLU": {"name": "Utilities Select Sector SPDR", "type": "defensive_equity", "typical_beta": 0.4},
    "XLP": {"name": "Consumer Staples Select Sector SPDR", "type": "defensive_equity", "typical_beta": 0.5},
    "SH": {"name": "ProShares Short S&P500", "type": "inverse", "typical_beta": -1.0},
}


def _get_sector(ticker: str) -> str:
    if ticker in SECTOR_MAP:
        return SECTOR_MAP[ticker]
    try:
        info = yf.Ticker(ticker).info
        return info.get("sector", "Other")
    except Exception:
        return "Other"


def _concentration_risk(holdings: List[dict], real_prices: Dict[str, float]) -> Dict[str, Any]:
    total_value = 0
    stock_values = []
    sector_values: Dict[str, float] = {}

    for h in holdings:
        price = real_prices.get(h["ticker"], h.get("avg_price", 100))
        mv = price * h["quantity"]
        total_value += mv
        stock_values.append({"ticker": h["ticker"], "market_value": mv})

        sector = _get_sector(h["ticker"])
        sector_values[sector] = sector_values.get(sector, 0) + mv

    if total_value <= 0:
        return {"score": 0, "warnings": [], "stock_weights": {}, "sector_weights": {}}

    stock_weights = {s["ticker"]: round(s["market_value"] / total_value, 4) for s in stock_values}
    sector_weights = {sec: round(val / total_value, 4) for sec, val in sector_values.items()}

    warnings = []
    score = 0

    for ticker, weight in stock_weights.items():
        if weight > 0.25:
            warnings.append({
                "type": "single_stock",
                "severity": "high" if weight > 0.40 else "medium",
                "message": f"{ticker} represents {weight*100:.1f}% of portfolio — exceeds 25% single-stock threshold",
                "ticker": ticker,
                "weight": round(weight, 4),
            })
            score += 30 if weight > 0.40 else 15

    for sector, weight in sector_weights.items():
        if weight > 0.40 and sector not in ("ETF", "Crypto"):
            warnings.append({
                "type": "sector_concentration",
                "severity": "high" if weight > 0.60 else "medium",
                "message": f"{sector} sector at {weight*100:.1f}% — exceeds 40% sector concentration threshold",
                "sector": sector,
                "weight": round(weight, 4),
            })
            score += 25 if weight > 0.60 else 12

    hhi = sum(w ** 2 for w in stock_weights.values())
    if hhi > 0.25:
        score += 10

    score = min(100, score)

    return {
        "score": score,
        "warnings": warnings,
        "stock_weights": stock_weights,
        "sector_weights": sector_weights,
        "hhi": round(hhi, 4),
    }


def _correlation_clustering(holdings: List[dict]) -> Dict[str, Any]:
    tickers = [h["ticker"] for h in holdings]
    if len(tickers) < 2:
        return {"score": 0, "avg_correlation": 0, "is_single_bet": False, "clusters": [], "warnings": []}

    price_data = {}
    for t in tickers:
        df = fetch_ohlcv(t, days=365)
        if df is not None and len(df) > 30:
            price_data[t] = df.set_index("date")["close"]

    if len(price_data) < 2:
        return {"score": 0, "avg_correlation": 0, "is_single_bet": False, "clusters": [], "warnings": []}

    prices_df = pd.DataFrame(price_data).dropna()
    if len(prices_df) < 20:
        return {"score": 0, "avg_correlation": 0, "is_single_bet": False, "clusters": [], "warnings": []}

    returns_df = prices_df.pct_change().dropna()
    corr = returns_df.corr()
    cols = list(corr.columns)

    pairwise = []
    high_corr_pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = float(corr.iloc[i, j])
            pairwise.append(c)
            if c > 0.75:
                high_corr_pairs.append({"pair": [cols[i], cols[j]], "correlation": round(c, 3)})

    avg_corr = float(np.mean(pairwise)) if pairwise else 0
    is_single_bet = avg_corr > 0.70

    clusters = []
    visited = set()
    for i in range(len(cols)):
        if cols[i] in visited:
            continue
        cluster = [cols[i]]
        visited.add(cols[i])
        for j in range(i + 1, len(cols)):
            if cols[j] not in visited and float(corr.iloc[i, j]) > 0.70:
                cluster.append(cols[j])
                visited.add(cols[j])
        if len(cluster) > 1:
            clusters.append(cluster)

    warnings = []
    if is_single_bet:
        warnings.append({
            "type": "single_bet",
            "severity": "high",
            "message": f"Portfolio is effectively one bet — average correlation is {avg_corr:.2f}. Diversification benefits are minimal.",
        })

    for pair in high_corr_pairs[:5]:
        warnings.append({
            "type": "high_correlation_pair",
            "severity": "medium",
            "message": f"{pair['pair'][0]} and {pair['pair'][1]} have {pair['correlation']:.0%} correlation — they move almost identically",
        })

    score = min(100, int(avg_corr * 100)) if avg_corr > 0 else 0

    return {
        "score": score,
        "avg_correlation": round(avg_corr, 3),
        "is_single_bet": is_single_bet,
        "clusters": clusters,
        "high_corr_pairs": high_corr_pairs[:5],
        "warnings": warnings,
    }


def _tail_risk(holdings: List[dict], real_prices: Dict[str, float]) -> Dict[str, Any]:
    tickers = [h["ticker"] for h in holdings]

    total_value = 0
    weights = {}
    for h in holdings:
        price = real_prices.get(h["ticker"], h.get("avg_price", 100))
        mv = price * h["quantity"]
        total_value += mv
        weights[h["ticker"]] = mv

    if total_value <= 0:
        return {"score": 0, "drawdown_probability": 0, "var_95": 0, "cvar_95": 0, "warnings": []}

    for t in weights:
        weights[t] /= total_value

    price_data = {}
    for t in tickers:
        df = fetch_ohlcv(t, days=730)
        if df is not None and len(df) > 50:
            price_data[t] = df.set_index("date")["close"]

    if not price_data:
        return {"score": 0, "drawdown_probability": 0, "var_95": 0, "cvar_95": 0, "warnings": []}

    prices_df = pd.DataFrame(price_data).dropna()
    if len(prices_df) < 30:
        return {"score": 0, "drawdown_probability": 0, "var_95": 0, "cvar_95": 0, "warnings": []}

    returns_df = prices_df.pct_change().dropna()
    returns_df = returns_df[(returns_df.abs() < 0.5).all(axis=1)]

    w = np.array([weights.get(col, 0) for col in returns_df.columns])
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum

    portfolio_returns = (returns_df.values * w).sum(axis=1)

    annual_vol = float(np.std(portfolio_returns) * np.sqrt(TRADING_DAYS))
    daily_vol = float(np.std(portfolio_returns))

    var_95_daily = float(np.percentile(portfolio_returns, 5))
    cvar_95_daily = float(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean()) if len(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)]) > 0 else var_95_daily

    monthly_vol = daily_vol * np.sqrt(21)
    z_score_10pct = -0.10 / monthly_vol if monthly_vol > 0 else -10
    from scipy.stats import norm
    drawdown_prob = float(norm.cdf(z_score_10pct)) * 100

    warnings = []
    score = 0

    if drawdown_prob > 20:
        warnings.append({
            "type": "high_tail_risk",
            "severity": "high",
            "message": f"Estimated {drawdown_prob:.1f}% probability of >10% portfolio drawdown in next month",
        })
        score = min(100, int(drawdown_prob * 2))
    elif drawdown_prob > 10:
        warnings.append({
            "type": "moderate_tail_risk",
            "severity": "medium",
            "message": f"Estimated {drawdown_prob:.1f}% probability of >10% drawdown — elevated but manageable",
        })
        score = min(100, int(drawdown_prob * 1.5))
    else:
        score = max(0, int(drawdown_prob))

    if annual_vol > 0.30:
        warnings.append({
            "type": "high_volatility",
            "severity": "medium",
            "message": f"Portfolio annualized volatility is {annual_vol*100:.1f}% — consider reducing position sizes",
        })

    return {
        "score": score,
        "drawdown_probability_1m": round(drawdown_prob, 2),
        "var_95_daily": round(abs(var_95_daily) * 100, 2),
        "cvar_95_daily": round(abs(cvar_95_daily) * 100, 2),
        "annual_volatility": round(annual_vol * 100, 2),
        "warnings": warnings,
    }


def _hedging_suggestions(holdings: List[dict], real_prices: Dict[str, float]) -> List[Dict[str, Any]]:
    tickers = [h["ticker"] for h in holdings]

    total_value = 0
    weights = {}
    for h in holdings:
        price = real_prices.get(h["ticker"], h.get("avg_price", 100))
        mv = price * h["quantity"]
        total_value += mv
        weights[h["ticker"]] = mv

    if total_value <= 0:
        return []

    for t in weights:
        weights[t] /= total_value

    spy_df = fetch_ohlcv("SPY", days=365)
    if spy_df is None or len(spy_df) < 50:
        return []

    spy_returns = spy_df.set_index("date")["close"].pct_change().dropna()

    price_data = {}
    for t in tickers:
        df = fetch_ohlcv(t, days=365)
        if df is not None and len(df) > 50:
            price_data[t] = df.set_index("date")["close"]

    if not price_data:
        return []

    prices_df = pd.DataFrame(price_data).dropna()
    returns_df = prices_df.pct_change().dropna()

    w = np.array([weights.get(col, 0) for col in returns_df.columns])
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum

    portfolio_returns = pd.Series((returns_df.values * w).sum(axis=1), index=returns_df.index)

    common_idx = portfolio_returns.index.intersection(spy_returns.index)
    if len(common_idx) < 30:
        return []

    port_aligned = portfolio_returns.loc[common_idx]
    spy_aligned = spy_returns.loc[common_idx]
    cov_matrix = np.cov(port_aligned, spy_aligned)
    portfolio_beta = float(cov_matrix[0, 1] / cov_matrix[1, 1]) if cov_matrix[1, 1] > 0 else 1.0

    suggestions = []

    if portfolio_beta > 1.2:
        suggestions.append({
            "hedge": "TLT",
            "name": "iShares 20+ Year Treasury Bond ETF",
            "rationale": f"Portfolio beta is {portfolio_beta:.2f}. Long-duration bonds typically have negative equity correlation, reducing overall portfolio beta.",
            "suggested_allocation": "5-15%",
            "expected_beta_reduction": round(portfolio_beta * 0.15, 2),
            "priority": "high",
        })

    if portfolio_beta > 1.0:
        suggestions.append({
            "hedge": "GLD",
            "name": "SPDR Gold Shares",
            "rationale": f"Gold provides portfolio insurance with near-zero equity beta. Helps during inflation spikes and market stress.",
            "suggested_allocation": "3-10%",
            "expected_beta_reduction": round(portfolio_beta * 0.05, 2),
            "priority": "medium",
        })

    sectors = {}
    for h in holdings:
        sec = _get_sector(h["ticker"])
        price = real_prices.get(h["ticker"], h.get("avg_price", 100))
        mv = price * h["quantity"]
        sectors[sec] = sectors.get(sec, 0) + mv

    tech_weight = sectors.get("Technology", 0) / total_value if total_value > 0 else 0
    if tech_weight > 0.50:
        suggestions.append({
            "hedge": "XLU",
            "name": "Utilities Select Sector SPDR",
            "rationale": f"Tech exposure is {tech_weight*100:.0f}%. Utilities have low correlation with tech and provide stable dividends as a defensive offset.",
            "suggested_allocation": "5-10%",
            "expected_beta_reduction": round(portfolio_beta * 0.08, 2),
            "priority": "medium",
        })
        suggestions.append({
            "hedge": "XLP",
            "name": "Consumer Staples Select Sector SPDR",
            "rationale": "Consumer staples are recession-resistant and provide counter-cyclical balance to growth-heavy portfolios.",
            "suggested_allocation": "5-10%",
            "expected_beta_reduction": round(portfolio_beta * 0.07, 2),
            "priority": "medium",
        })

    if portfolio_beta > 1.5:
        suggestions.append({
            "hedge": "SH",
            "name": "ProShares Short S&P500",
            "rationale": f"Portfolio beta is very high ({portfolio_beta:.2f}). A small inverse ETF position can act as tactical downside protection.",
            "suggested_allocation": "2-5%",
            "expected_beta_reduction": round(portfolio_beta * 0.20, 2),
            "priority": "high",
        })

    suggestions.sort(key=lambda x: 0 if x["priority"] == "high" else 1)
    return suggestions[:5]


def _earnings_risk_exposure(holdings: List[dict], real_prices: Dict[str, float]) -> Dict[str, Any]:
    tickers = [h["ticker"] for h in holdings]

    total_value = 0
    ticker_values = {}
    for h in holdings:
        price = real_prices.get(h["ticker"], h.get("avg_price", 100))
        mv = price * h["quantity"]
        total_value += mv
        ticker_values[h["ticker"]] = mv

    if total_value <= 0:
        return {"score": 0, "exposed_weight": 0, "upcoming_earnings": [], "warnings": []}

    earnings = get_earnings_calendar(tickers)

    now = datetime.now()
    two_weeks = now + timedelta(days=14)
    upcoming = []
    exposed_value = 0

    for e in earnings:
        try:
            earn_date = datetime.strptime(e.get("date", ""), "%Y-%m-%d")
            if earn_date <= two_weeks:
                ticker = e.get("ticker", "")
                mv = ticker_values.get(ticker, 0)
                weight = mv / total_value if total_value > 0 else 0
                exposed_value += mv
                upcoming.append({
                    "ticker": ticker,
                    "date": e.get("date", ""),
                    "days_until": e.get("days_until", 0),
                    "weight": round(weight, 4),
                    "market_value": round(mv, 2),
                })
        except (ValueError, KeyError):
            continue

    exposed_weight = exposed_value / total_value if total_value > 0 else 0

    warnings = []
    score = 0

    if exposed_weight > 0.40:
        warnings.append({
            "type": "high_earnings_exposure",
            "severity": "high",
            "message": f"{exposed_weight*100:.1f}% of portfolio has earnings in next 2 weeks — significant event risk",
        })
        score = min(100, int(exposed_weight * 150))
    elif exposed_weight > 0.20:
        warnings.append({
            "type": "moderate_earnings_exposure",
            "severity": "medium",
            "message": f"{exposed_weight*100:.1f}% of portfolio has upcoming earnings — consider hedging or trimming",
        })
        score = min(100, int(exposed_weight * 100))
    elif exposed_weight > 0:
        score = max(0, int(exposed_weight * 50))

    return {
        "score": score,
        "exposed_weight": round(exposed_weight, 4),
        "exposed_value": round(exposed_value, 2),
        "upcoming_earnings": upcoming,
        "warnings": warnings,
    }


def compute_risk_intelligence(holdings: List[dict], real_prices: Dict[str, float]) -> Dict[str, Any]:
    concentration = _concentration_risk(holdings, real_prices)
    correlation = _correlation_clustering(holdings)
    tail = _tail_risk(holdings, real_prices)
    hedging = _hedging_suggestions(holdings, real_prices)
    earnings = _earnings_risk_exposure(holdings, real_prices)

    component_scores = {
        "concentration": concentration["score"],
        "correlation": correlation["score"],
        "tail_risk": tail["score"],
        "earnings_exposure": earnings["score"],
    }

    weights = {"concentration": 0.30, "correlation": 0.25, "tail_risk": 0.30, "earnings_exposure": 0.15}
    overall_score = sum(component_scores[k] * weights[k] for k in component_scores)
    overall_score = round(min(100, max(0, overall_score)), 1)

    if overall_score <= 25:
        health = "Excellent"
    elif overall_score <= 45:
        health = "Good"
    elif overall_score <= 65:
        health = "Fair"
    else:
        health = "Poor"

    all_warnings = (
        concentration.get("warnings", [])
        + correlation.get("warnings", [])
        + tail.get("warnings", [])
        + earnings.get("warnings", [])
    )
    all_warnings.sort(key=lambda w: {"high": 0, "medium": 1, "low": 2}.get(w.get("severity", "low"), 2))

    return {
        "overall_risk_score": overall_score,
        "portfolio_health": health,
        "component_scores": component_scores,
        "concentration": {
            "score": concentration["score"],
            "stock_weights": concentration.get("stock_weights", {}),
            "sector_weights": concentration.get("sector_weights", {}),
            "hhi": concentration.get("hhi", 0),
            "warnings": concentration.get("warnings", []),
        },
        "correlation": {
            "score": correlation["score"],
            "avg_correlation": correlation.get("avg_correlation", 0),
            "is_single_bet": correlation.get("is_single_bet", False),
            "clusters": correlation.get("clusters", []),
            "high_corr_pairs": correlation.get("high_corr_pairs", []),
            "warnings": correlation.get("warnings", []),
        },
        "tail_risk": {
            "score": tail["score"],
            "drawdown_probability_1m": tail.get("drawdown_probability_1m", 0),
            "var_95_daily": tail.get("var_95_daily", 0),
            "cvar_95_daily": tail.get("cvar_95_daily", 0),
            "annual_volatility": tail.get("annual_volatility", 0),
            "warnings": tail.get("warnings", []),
        },
        "hedging_suggestions": hedging,
        "earnings_exposure": {
            "score": earnings["score"],
            "exposed_weight": earnings.get("exposed_weight", 0),
            "exposed_value": earnings.get("exposed_value", 0),
            "upcoming_earnings": earnings.get("upcoming_earnings", []),
            "warnings": earnings.get("warnings", []),
        },
        "all_warnings": all_warnings[:10],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
