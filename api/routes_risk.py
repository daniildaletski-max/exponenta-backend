from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import get_current_user
from core.risk_engine import RiskEngine

router = APIRouter()
_engine = RiskEngine()


class StressTestRequest(BaseModel):
    holdings: list[dict] | None = None  # if None, use demo
    scenarios: list[str] | None = None


class StressTestItem(BaseModel):
    scenario_name: str
    portfolio_impact_pct: float
    worst_asset: str
    worst_asset_impact_pct: float


class StressTestResponse(BaseModel):
    results: list[StressTestItem]
    total_value: float
    generated_at: datetime


class RiskMetricsRequest(BaseModel):
    tickers: list[str] = Field(default=["AAPL", "VOO", "TSLA", "NVDA"])
    period_days: int = Field(default=252, ge=30, le=1260)


class RiskMetricsResponse(BaseModel):
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    expected_shortfall: float
    omega_ratio: float
    tail_ratio: float
    max_drawdown: float
    generated_at: datetime


class MonteCarloRequest(BaseModel):
    ticker: str = "AAPL"
    n_simulations: int = Field(default=10000, ge=100, le=100000)
    n_days: int = Field(default=252, ge=5, le=504)


class MonteCarloResponse(BaseModel):
    ticker: str
    current_price: float
    percentiles: dict[str, float]
    var_95: float
    cvar_95: float
    n_simulations: int
    n_days: int
    generated_at: datetime


class CorrelationRequest(BaseModel):
    tickers: list[str] = Field(default=["AAPL", "VOO", "TSLA", "NVDA", "BTC-USD"])
    period_days: int = Field(default=252, ge=30, le=1260)
    stress: bool = False


class CorrelationResponse(BaseModel):
    tickers: list[str]
    matrix: list[list[float]]
    stressed: bool
    generated_at: datetime


@router.post("/stress-test", response_model=StressTestResponse)
async def stress_test(req: StressTestRequest, user: dict = Depends(get_current_user)):
    """Run stress-test scenarios against portfolio."""
    if req.holdings:
        holdings = req.holdings
    else:
        # Fallback to demo holdings
        holdings = [
            {"ticker": "AAPL", "value": 9765.0},
            {"ticker": "VOO", "value": 13458.0},
            {"ticker": "BTC-USD", "value": 35600.0, "asset_class": "crypto"},
            {"ticker": "TLT", "value": 3712.0, "asset_class": "bond"},
        ]

    total_value = sum(h.get("value", 0) for h in holdings)
    results = _engine.stress_test(holdings, scenarios=req.scenarios)

    return StressTestResponse(
        results=[StressTestItem(
            scenario_name=r.scenario_name,
            portfolio_impact_pct=r.portfolio_impact_pct,
            worst_asset=r.worst_asset,
            worst_asset_impact_pct=r.worst_asset_impact_pct,
        ) for r in results],
        total_value=total_value,
        generated_at=datetime.now(timezone.utc),
    )


@router.post("/metrics", response_model=RiskMetricsResponse)
async def risk_metrics(req: RiskMetricsRequest, user: dict = Depends(get_current_user)):
    """Compute tail risk metrics for a set of tickers."""
    import asyncio
    from data.market_data import get_ticker_history

    period_map = {252: "1y", 126: "6mo", 63: "3mo", 21: "1mo"}
    period = period_map.get(req.period_days, "1y" if req.period_days > 180 else "3mo")

    returns_list = []
    for ticker in req.tickers:
        try:
            hist = await asyncio.to_thread(get_ticker_history, ticker, period)
            if hist and len(hist) > 1:
                closes = [bar["close"] for bar in hist]
                p = np.array(closes)
                r = np.diff(p) / p[:-1]
                returns_list.append(r)
        except Exception:
            continue

    if not returns_list:
        raise HTTPException(400, "No valid price data found")

    # Equal-weighted portfolio return
    min_len = min(len(r) for r in returns_list)
    portfolio_returns = np.mean([r[:min_len] for r in returns_list], axis=0)

    m = _engine.compute_tail_metrics(portfolio_returns)
    return RiskMetricsResponse(
        var_95=m.var_95,
        var_99=m.var_99,
        cvar_95=m.cvar_95,
        cvar_99=m.cvar_99,
        expected_shortfall=m.expected_shortfall,
        omega_ratio=m.omega_ratio,
        tail_ratio=m.tail_ratio,
        max_drawdown=m.max_drawdown,
        generated_at=datetime.now(timezone.utc),
    )


@router.post("/monte-carlo", response_model=MonteCarloResponse)
async def monte_carlo(req: MonteCarloRequest, user: dict = Depends(get_current_user)):
    """Run jump-diffusion Monte Carlo simulation."""
    from data.market_data import fetch_historical_prices

    try:
        prices = await fetch_historical_prices([req.ticker], days=504)
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch price data: {e}")

    ticker_prices = prices.get(req.ticker)
    if not ticker_prices or len(ticker_prices) < 30:
        raise HTTPException(400, f"Insufficient price data for {req.ticker}")

    price_array = np.array(ticker_prices)
    result = _engine.jump_diffusion_monte_carlo(
        price_array,
        n_simulations=req.n_simulations,
        n_days=req.n_days,
    )

    return MonteCarloResponse(
        ticker=req.ticker,
        current_price=float(price_array[-1]),
        percentiles=result["percentiles"],
        var_95=result["var_95"],
        cvar_95=result["cvar_95"],
        n_simulations=req.n_simulations,
        n_days=req.n_days,
        generated_at=datetime.now(timezone.utc),
    )


@router.post("/correlation", response_model=CorrelationResponse)
async def correlation_matrix(req: CorrelationRequest, user: dict = Depends(get_current_user)):
    """Compute correlation matrix with optional stress simulation."""
    from data.market_data import fetch_historical_prices

    try:
        prices = await fetch_historical_prices(req.tickers, days=req.period_days)
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch price data: {e}")

    returns_list = []
    valid_tickers = []
    for ticker in req.tickers:
        if ticker in prices and len(prices[ticker]) > 1:
            p = np.array(prices[ticker])
            r = np.diff(p) / p[:-1]
            returns_list.append(r)
            valid_tickers.append(ticker)

    if len(valid_tickers) < 2:
        raise HTTPException(400, "Need at least 2 tickers with valid data")

    min_len = min(len(r) for r in returns_list)
    returns_matrix = np.column_stack([r[:min_len] for r in returns_list])

    if req.stress:
        corr = _engine.correlation_stress(returns_matrix)
    else:
        corr = np.corrcoef(returns_matrix, rowvar=False)

    return CorrelationResponse(
        tickers=valid_tickers,
        matrix=corr.tolist(),
        stressed=req.stress,
        generated_at=datetime.now(timezone.utc),
    )
