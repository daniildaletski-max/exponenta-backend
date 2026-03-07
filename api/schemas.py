from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ── Auth ──────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str | None = None
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str
    created_at: datetime


# ── Portfolio ─────────────────────────────────────────────────────────

class AssetClass(str, Enum):
    STOCK = "stock"
    ETF = "etf"
    CRYPTO = "crypto"
    BOND = "bond"


class Holding(BaseModel):
    ticker: str
    name: str
    asset_class: AssetClass
    quantity: float
    avg_price: float
    current_price: float
    pnl: float
    pnl_pct: float
    weight: float = Field(ge=0, le=1)


class PortfolioResponse(BaseModel):
    total_value: float
    daily_pnl: float
    daily_pnl_pct: float
    holdings: list[Holding]
    updated_at: datetime


class EfficientFrontierPoint(BaseModel):
    volatility: float
    expected_return: float
    sharpe_ratio: float


class PortfolioAnalysis(BaseModel):
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    beta: float
    diversification_score: float = Field(ge=0, le=1)
    risk_level: str
    suggestions: list[str]
    efficient_frontier: list[EfficientFrontierPoint] = []
    is_fallback: bool = False


# ── Market ────────────────────────────────────────────────────────────

class Quote(BaseModel):
    ticker: str
    name: str
    price: float
    change: float
    change_pct: float
    volume: int
    timestamp: datetime


class QuoteItem(BaseModel):
    ticker: str
    price: float
    change: float
    change_pct: float
    volume: int
    prev_close: float


class QuotesResponse(BaseModel):
    quotes: list[QuoteItem]
    count: int
    timestamp: datetime


class SentimentSourceBreakdown(BaseModel):
    source: str
    score: float = Field(ge=-1, le=1)
    article_count: int = 0


class SentimentResult(BaseModel):
    ticker: str
    overall_score: float = Field(ge=-1, le=1)
    news_score: float = Field(ge=-1, le=1)
    social_score: float = Field(ge=-1, le=1)
    signal: str  # "bullish" | "bearish" | "neutral"
    top_headlines: list[str]
    analyzed_at: datetime
    source_breakdown: list[SentimentSourceBreakdown] = []
    is_fallback: bool = False


# ── AI / Predictions ─────────────────────────────────────────────────

class BacktestingMetrics(BaseModel):
    mae: float = 0
    mape: float = 0
    directional_accuracy: float = 0


class TrendPrediction(BaseModel):
    ticker: str
    horizon_days: int
    predicted_prices: list[float]
    confidence_lower: list[float]
    confidence_upper: list[float]
    model_weights: dict[str, float]
    generated_at: datetime
    backtesting: BacktestingMetrics = BacktestingMetrics()
    is_fallback: bool = False


class Recommendation(BaseModel):
    ticker: str
    action: str  # "buy" | "sell" | "hold" | "rebalance"
    conviction: float = Field(ge=0, le=1)
    rationale: str
    target_weight: float | None = None
    expected_return: float | None = None


class RecommendationSet(BaseModel):
    recommendations: list[Recommendation]
    portfolio_score_before: float
    portfolio_score_after: float
    generated_at: datetime
    is_fallback: bool = False
