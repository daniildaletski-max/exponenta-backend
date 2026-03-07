"""
Prediction & ML endpoints migrated from Replit server.

Includes: ML prediction, scanning, composite intelligence, trade thesis,
backtesting, signal tracking, options flow, event intelligence.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.dependencies import get_current_user

router = APIRouter()


@router.get("/ml/{ticker}")
async def ml_predict(ticker: str, user: dict = Depends(get_current_user)):
    from prediction.engine import get_engine
    from prediction.smart_engine import get_tracker

    def _predict():
        engine = get_engine()
        result = engine.predict(ticker.upper())
        if "error" in result:
            return result
        try:
            tracker = get_tracker()
            tracker.track_prediction(ticker.upper(), result)
        except Exception:
            pass
        try:
            from prediction.signal_tracker import auto_record_ml_prediction
            auto_record_ml_prediction(ticker.upper(), result)
        except Exception:
            pass
        return result

    result = await asyncio.to_thread(_predict)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.get("/accuracy")
async def predict_accuracy(user: dict = Depends(get_current_user)):
    from prediction.smart_engine import get_tracker
    tracker = get_tracker()
    return await asyncio.to_thread(tracker.get_accuracy_stats)


@router.get("/scan/opportunities")
async def scan_opportunities(
    universe: str = Query(default="us_mega_cap"),
    max_assets: int = Query(default=15, ge=1, le=50),
    use_ml: bool = Query(default=False),
    min_score: float = Query(default=0, ge=0, le=100),
    signal_filter: str = Query(default=None),
    sector_filter: str = Query(default=None),
    user: dict = Depends(get_current_user),
):
    from prediction.scanner import scan_universe
    keys = [k.strip() for k in universe.split(",")]
    result = await asyncio.to_thread(
        scan_universe,
        universe_keys=keys, max_assets=max_assets, use_ml=use_ml,
        min_score=min_score, signal_filter=signal_filter, sector_filter=sector_filter,
    )
    return result


@router.get("/advisor/{ticker}")
async def advisor_analyze(
    ticker: str,
    risk_tolerance: str = Query(default="moderate"),
    user: dict = Depends(get_current_user),
):
    from prediction.agentic import run_agentic_analysis
    result = await run_agentic_analysis(ticker.upper(), user_risk_tolerance=risk_tolerance)
    return result


@router.get("/intel/{ticker}/composite")
async def composite_intelligence(ticker: str, user: dict = Depends(get_current_user)):
    from prediction.composite_intel import get_composite_score
    result = await get_composite_score(ticker.upper())
    try:
        from prediction.signal_tracker import auto_record_composite_score
        auto_record_composite_score(ticker.upper(), result)
    except Exception:
        pass
    return result


@router.get("/thesis/{ticker}")
async def generate_thesis(ticker: str, user: dict = Depends(get_current_user)):
    from prediction.trade_thesis import generate_trade_thesis
    result = await generate_trade_thesis(ticker.upper())
    try:
        from prediction.signal_tracker import auto_record_thesis
        auto_record_thesis(ticker.upper(), result)
    except Exception:
        pass
    return result


@router.get("/flow/{ticker}/smart-money")
async def smart_money_flow(ticker: str, user: dict = Depends(get_current_user)):
    from prediction.flow_tracker import get_smart_money_flow

    def _fetch():
        result = get_smart_money_flow(ticker.upper())
        try:
            from prediction.signal_tracker import auto_record_smart_money
            auto_record_smart_money(ticker.upper(), result)
        except Exception:
            pass
        return result

    return await asyncio.to_thread(_fetch)


@router.get("/options/{ticker}/flow")
async def options_flow(ticker: str, user: dict = Depends(get_current_user)):
    from prediction.options_flow import get_options_flow
    result = await asyncio.to_thread(get_options_flow, ticker.upper())
    return result


@router.get("/events/{ticker}/intelligence")
async def event_intelligence(ticker: str, user: dict = Depends(get_current_user)):
    from prediction.event_intel import get_event_intelligence
    result = await get_event_intelligence(ticker.upper())
    return result


@router.get("/events/live-feed")
async def live_event_feed(
    tickers: str = Query(default=None),
    user: dict = Depends(get_current_user),
):
    from prediction.event_intel import get_live_event_feed
    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
    else:
        ticker_list = ["AAPL", "TSLA", "NVDA", "GOOGL", "MSFT", "AMZN", "META"]
    result = await get_live_event_feed(ticker_list)
    return result


@router.get("/fundamentals/{ticker}")
async def get_fundamentals(ticker: str, user: dict = Depends(get_current_user)):
    from prediction.fundamentals import get_fundamental_analysis
    result = await asyncio.to_thread(get_fundamental_analysis, ticker.upper())
    return result


@router.get("/signals/performance")
async def signal_performance(user: dict = Depends(get_current_user)):
    from prediction.signal_tracker import get_signal_performance
    return await asyncio.to_thread(get_signal_performance)


class RecordSignalRequest(BaseModel):
    signal_type: str
    ticker: str
    direction: str
    score: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=100)
    price_at_signal: float = Field(gt=0)
    details: dict | None = None


@router.post("/signals/record")
async def record_signal(req: RecordSignalRequest, user: dict = Depends(get_current_user)):
    from prediction.signal_tracker import record_signal as _record_signal
    result = _record_signal(
        signal_type=req.signal_type,
        ticker=req.ticker.upper(),
        direction=req.direction,
        score=req.score,
        confidence=req.confidence,
        price_at_signal=req.price_at_signal,
        details=req.details,
    )
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


class BacktestRequest(BaseModel):
    ticker: str = Field(default="AAPL")
    strategy: str = Field(default="trend_following")
    params: dict | None = None
    period: str = Field(default="1y")
    initial_capital: float = Field(default=10000, gt=0)
    benchmark: str = Field(default="SPY")


@router.post("/backtest/run")
async def run_backtest(req: BacktestRequest, user: dict = Depends(get_current_user)):
    from prediction.backtester import run_backtest as _run_backtest
    result = await asyncio.to_thread(
        _run_backtest,
        ticker=req.ticker.upper(),
        strategy=req.strategy,
        params=req.params,
        period=req.period,
        initial_capital=req.initial_capital,
        benchmark=req.benchmark.upper(),
    )
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.get("/macro/dashboard")
async def macro_dashboard(user: dict = Depends(get_current_user)):
    from prediction.macro_intel import get_macro_dashboard
    return await get_macro_dashboard()


@router.get("/market/intelligence")
async def market_intelligence(user: dict = Depends(get_current_user)):
    from prediction.market_intelligence import MarketIntelligence
    mi = MarketIntelligence()
    tickers = ["AAPL", "TSLA", "NVDA", "GOOGL", "MSFT", "AMZN", "META"]
    report = await asyncio.to_thread(mi.generate_intelligence_report, tickers)
    return report


@router.get("/market/heatmap")
async def market_heatmap(
    timeframe: str = Query(default="1D"),
    user: dict = Depends(get_current_user),
):
    from prediction.market_map import get_market_heatmap
    return await asyncio.to_thread(get_market_heatmap, timeframe)


@router.get("/watchlist/smart")
async def smart_watchlist(
    tickers: str = Query(default=None),
    user: dict = Depends(get_current_user),
):
    from prediction.smart_watchlist import get_smart_watchlist
    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
    else:
        ticker_list = ["AAPL", "TSLA", "NVDA", "GOOGL", "MSFT", "AMZN"]
    return await get_smart_watchlist(ticker_list)


@router.get("/briefing/daily")
async def daily_briefing(user: dict = Depends(get_current_user)):
    from prediction.daily_briefing import generate_daily_briefing
    return await generate_daily_briefing()


@router.get("/alerts")
async def get_alerts(
    risk_tolerance: str = Query(default="moderate"),
    user: dict = Depends(get_current_user),
):
    from prediction.alert_engine import get_all_alerts
    tickers = ["AAPL", "TSLA", "NVDA", "GOOGL", "MSFT", "AMZN"]
    result = await asyncio.to_thread(get_all_alerts, tickers, risk_tolerance)
    return result
