from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from api.dependencies import get_current_user
from api.schemas import (
    AssetClass,
    Holding,
    PortfolioAnalysis,
    PortfolioResponse,
)

router = APIRouter()

_DEMO_HOLDINGS = [
    Holding(
        ticker="AAPL", name="Apple Inc.", asset_class=AssetClass.STOCK,
        quantity=50, avg_price=178.50, current_price=195.30,
        pnl=840.0, pnl_pct=9.41, weight=0.25,
    ),
    Holding(
        ticker="VOO", name="Vanguard S&P 500 ETF", asset_class=AssetClass.ETF,
        quantity=30, avg_price=420.0, current_price=448.60,
        pnl=858.0, pnl_pct=6.81, weight=0.35,
    ),
    Holding(
        ticker="BTC-USD", name="Bitcoin", asset_class=AssetClass.CRYPTO,
        quantity=0.5, avg_price=62000.0, current_price=71200.0,
        pnl=4600.0, pnl_pct=14.84, weight=0.20,
    ),
    Holding(
        ticker="TLT", name="iShares 20+ Year Treasury", asset_class=AssetClass.BOND,
        quantity=40, avg_price=95.0, current_price=92.80,
        pnl=-88.0, pnl_pct=-2.32, weight=0.20,
    ),
]


@router.get("", response_model=PortfolioResponse)
async def get_portfolio(user: dict = Depends(get_current_user)):
    total = sum(h.current_price * h.quantity for h in _DEMO_HOLDINGS)
    daily_pnl = sum(h.pnl for h in _DEMO_HOLDINGS)
    return PortfolioResponse(
        total_value=total,
        daily_pnl=daily_pnl,
        daily_pnl_pct=(daily_pnl / total) * 100 if total else 0,
        holdings=_DEMO_HOLDINGS,
        updated_at=datetime.now(timezone.utc),
    )


@router.get("/analysis", response_model=PortfolioAnalysis)
async def get_analysis(user: dict = Depends(get_current_user)):
    from agents.portfolio_agent import run_portfolio_analysis

    holdings = [
        {"ticker": h.ticker, "quantity": h.quantity, "avg_price": h.avg_price}
        for h in _DEMO_HOLDINGS
    ]

    try:
        raw = await run_portfolio_analysis(holdings)
        return PortfolioAnalysis(
            sharpe_ratio=raw.get("sharpe_ratio", 0),
            sortino_ratio=raw.get("sortino_ratio", 0),
            max_drawdown=raw.get("max_drawdown", 0),
            beta=raw.get("beta", 1.0),
            diversification_score=raw.get("diversification_score", 0.5),
            risk_level=raw.get("risk_level", "moderate"),
            suggestions=raw.get("suggestions", []),
            efficient_frontier=[],
        )
    except Exception:
        return PortfolioAnalysis(
            sharpe_ratio=1.42,
            sortino_ratio=1.87,
            max_drawdown=-0.12,
            beta=0.91,
            diversification_score=0.73,
            risk_level="moderate",
            suggestions=[
                "Consider reducing BTC-USD weight to below 15% to decrease portfolio volatility",
                "TLT position is underperforming — evaluate short-duration bond alternatives",
                "Adding international equity exposure (e.g. VXUS) would improve diversification",
            ],
            efficient_frontier=[],
            is_fallback=True,
        )
