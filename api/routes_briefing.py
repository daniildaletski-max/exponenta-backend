from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.dependencies import get_current_user

router = APIRouter()


class MarketOverview(BaseModel):
    sp500_change_pct: float
    nasdaq_change_pct: float
    vix: float
    ten_year_yield: float
    dxy: float
    btc_change_pct: float
    summary: str


class PortfolioBrief(BaseModel):
    total_value: float
    daily_pnl: float
    daily_pnl_pct: float
    top_mover: str
    top_mover_change_pct: float
    risk_alert: str | None = None


class ActionItem(BaseModel):
    priority: str  # "high" | "medium" | "low"
    category: str  # "rebalance" | "risk" | "opportunity" | "alert"
    title: str
    description: str
    ticker: str | None = None


class DailyBriefing(BaseModel):
    date: str
    greeting: str
    market_overview: MarketOverview
    portfolio_brief: PortfolioBrief
    action_items: list[ActionItem]
    key_events: list[str]
    generated_at: datetime


@router.get("", response_model=DailyBriefing)
async def get_daily_briefing(user: dict = Depends(get_current_user)):
    """Generate a personalized daily market briefing."""
    now = datetime.now(timezone.utc)
    hour = now.hour

    if hour < 12:
        greeting = "Good morning"
    elif hour < 18:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"

    # Try to fetch real data, fall back to demo
    try:
        from data.market_data import fetch_historical_prices
        prices = await fetch_historical_prices(["SPY", "QQQ"], days=2)
        spy = prices.get("SPY", [100, 100])
        spy_chg = ((spy[-1] - spy[-2]) / spy[-2]) * 100 if len(spy) >= 2 else 0.12
    except Exception:
        spy_chg = 0.12

    market = MarketOverview(
        sp500_change_pct=round(spy_chg, 2),
        nasdaq_change_pct=round(spy_chg * 1.2, 2),
        vix=18.5,
        ten_year_yield=4.25,
        dxy=104.2,
        btc_change_pct=1.8,
        summary="Markets are mixed with tech leading. VIX remains subdued, suggesting low implied volatility.",
    )

    portfolio = PortfolioBrief(
        total_value=62535.0,
        daily_pnl=340.50,
        daily_pnl_pct=0.55,
        top_mover="NVDA",
        top_mover_change_pct=2.3,
        risk_alert=None,
    )

    actions = [
        ActionItem(
            priority="high",
            category="rebalance",
            title="BTC-USD overweight",
            description="Bitcoin allocation at 22% exceeds your 15% target. Consider trimming to lock in gains.",
            ticker="BTC-USD",
        ),
        ActionItem(
            priority="medium",
            category="opportunity",
            title="AAPL near support level",
            description="Apple is approaching its 200-day moving average at $187. Potential buy opportunity if it holds.",
            ticker="AAPL",
        ),
        ActionItem(
            priority="low",
            category="alert",
            title="Fed meeting next week",
            description="FOMC rate decision on Wednesday. Consider reducing leverage before the announcement.",
        ),
    ]

    key_events = [
        "FOMC rate decision Wednesday 2:00 PM ET",
        "AAPL earnings report Thursday after close",
        "Non-farm payrolls Friday 8:30 AM ET",
        "CPI data release next Tuesday",
    ]

    return DailyBriefing(
        date=now.strftime("%Y-%m-%d"),
        greeting=f"{greeting}, {user.get('email', 'investor').split('@')[0]}!",
        market_overview=market,
        portfolio_brief=portfolio,
        action_items=actions,
        key_events=key_events,
        generated_at=now,
    )
