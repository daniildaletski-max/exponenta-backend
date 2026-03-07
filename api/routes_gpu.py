"""
GPU-accelerated computation endpoints.

Provides Monte Carlo simulation, LSTM/TFT training, and device info.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.dependencies import get_current_user

router = APIRouter()


class MonteCarloRequest(BaseModel):
    ticker: str = Field(examples=["AAPL"])
    n_simulations: int = Field(default=100_000, ge=1000, le=1_000_000)
    n_days: int = Field(default=252, ge=5, le=504)


class MonteCarloResponse(BaseModel):
    ticker: str
    percentiles: dict[str, float]
    mean: float
    std: float
    initial_price: float
    var_95: float
    cvar_95: float
    n_simulations: int
    n_days: int
    device: str
    generated_at: datetime


class DeviceInfoResponse(BaseModel):
    device: str
    torch_version: str
    gpu_name: str | None = None
    gpu_memory_gb: float | None = None
    cuda_version: str | None = None
    note: str | None = None


class DeepPredictionResponse(BaseModel):
    ticker: str
    model: str
    predicted_prices: list[float]
    lower_bound: list[float]
    upper_bound: list[float]
    horizon_days: int
    device: str
    generated_at: datetime


@router.get("/device", response_model=DeviceInfoResponse)
async def get_device_info(user: dict = Depends(get_current_user)):
    from gpu.device import device_info
    return DeviceInfoResponse(**device_info())


@router.post("/monte-carlo", response_model=MonteCarloResponse)
async def run_monte_carlo(
    req: MonteCarloRequest,
    user: dict = Depends(get_current_user),
):
    from data.polygon_client import PolygonClient
    from gpu.monte_carlo import gpu_monte_carlo
    import numpy as np

    client = PolygonClient()
    try:
        bars = await client.get_daily_bars(req.ticker, limit=252)
    finally:
        await client.close()

    if not bars:
        raise HTTPException(404, f"No price data for {req.ticker}")

    prices = np.array([b["close"] for b in bars], dtype=np.float64)
    if len(prices) < 30:
        raise HTTPException(400, "Need at least 30 days of price data")

    result = await asyncio.to_thread(
        gpu_monte_carlo,
        prices=prices,
        n_simulations=req.n_simulations,
        n_days=req.n_days,
    )

    return MonteCarloResponse(
        ticker=req.ticker,
        percentiles=result["percentiles"],
        mean=result["mean"],
        std=result["std"],
        initial_price=result["initial_price"],
        var_95=result["var_95"],
        cvar_95=result["cvar_95"],
        n_simulations=result["n_simulations"],
        n_days=result["n_days"],
        device=result["device"],
        generated_at=datetime.now(timezone.utc),
    )


@router.get("/predict/deep/{ticker}", response_model=DeepPredictionResponse)
async def predict_deep(
    ticker: str,
    model: Literal["lstm", "tft"] = Query("lstm"),
    horizon: int = Query(30, ge=7, le=90),
    user: dict = Depends(get_current_user),
):
    from data.polygon_client import PolygonClient
    import numpy as np

    client = PolygonClient()
    try:
        bars = await client.get_daily_bars(ticker, limit=500)
    finally:
        await client.close()

    if not bars:
        raise HTTPException(404, f"No price data for {ticker}")

    prices = np.array([b["close"] for b in bars], dtype=np.float64)

    if model == "lstm":
        from models.lstm_model import LSTMForecaster
        forecaster = LSTMForecaster()
        result = await asyncio.to_thread(forecaster.predict, prices, horizon)
    elif model == "tft":
        from models.tft_model import TFTForecaster
        forecaster = TFTForecaster(prediction_length=horizon)
        result = await asyncio.to_thread(forecaster.predict, prices, horizon)
    else:
        raise HTTPException(400, f"Unknown model: {model}")

    from gpu.device import get_device

    return DeepPredictionResponse(
        ticker=ticker,
        model=model.upper(),
        predicted_prices=[round(float(p), 2) for p in result["predicted"]],
        lower_bound=[round(float(p), 2) for p in result["lower"]],
        upper_bound=[round(float(p), 2) for p in result["upper"]],
        horizon_days=horizon,
        device=str(get_device()),
        generated_at=datetime.now(timezone.utc),
    )
