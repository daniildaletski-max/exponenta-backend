"""
Exponenta Backend — FastAPI application.

Run: uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from middleware import (
    CorrelationMiddleware,
    MetricsMiddleware,
    RateLimitMiddleware,
    get_metrics_text,
)

logger = logging.getLogger(__name__)

APP_VERSION = "0.4.0"

# ── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        from db.database import init_db
        await init_db()
    except Exception as e:
        logger.warning(f"Database init skipped: {e}")

    try:
        from gpu.device import device_info
        info = device_info()
        logger.info(f"Compute device: {info['device']} — {info.get('gpu_name', 'CPU')}")
    except Exception:
        logger.info("PyTorch device detection skipped")

    yield
    # Shutdown (clean up resources if needed)


# ── App ───────────────────────────────────────────────────────────────

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8080",
    "https://app.exponenta.io",
]

app = FastAPI(
    title="Exponenta API",
    version=APP_VERSION,
    description="AI-powered investment analysis backend",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Order: outermost runs first → Correlation → Metrics → Rate Limiter → App
app.add_middleware(RateLimitMiddleware, default_rpm=60, ai_rpm=10)
app.add_middleware(MetricsMiddleware)
app.add_middleware(CorrelationMiddleware)

from api.routes_auth import router as auth_router
from api.routes_portfolio import router as portfolio_router
from api.routes_market import router as market_router
from api.routes_ai import router as ai_router
from api.routes_gpu import router as gpu_router
from api.routes_prediction import router as prediction_router
from api.routes_chat import router as chat_router
from api.routes_system import router as system_router
from api.routes_risk import router as risk_router
from api.routes_holdings import router as holdings_router
from api.routes_portfolio_history import router as portfolio_history_router
from api.routes_chat_history import router as chat_history_router
from api.routes_briefing import router as briefing_router
from api.routes_alerts import router as alerts_router
from api.routes_ml import router as ml_router

app.include_router(auth_router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(portfolio_router, prefix="/api/v1/portfolio", tags=["portfolio"])
app.include_router(market_router, prefix="/api/v1/market", tags=["market"])
app.include_router(ai_router, prefix="/api/v1/ai", tags=["ai"])
app.include_router(gpu_router, prefix="/api/v1/gpu", tags=["gpu"])
app.include_router(prediction_router, prefix="/api/v1/predict", tags=["prediction"])
app.include_router(chat_router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(system_router, prefix="/api/v1/system", tags=["system"])
app.include_router(risk_router, prefix="/api/v1/risk", tags=["risk"])
app.include_router(holdings_router, prefix="/api/v1/holdings", tags=["holdings"])
app.include_router(portfolio_history_router, prefix="/api/v1/portfolio/history", tags=["portfolio"])
app.include_router(chat_history_router, prefix="/api/v1/chat/history", tags=["chat"])
app.include_router(briefing_router, prefix="/api/v1/briefing", tags=["briefing"])
app.include_router(alerts_router, prefix="/api/v1/alerts", tags=["alerts"])
app.include_router(ml_router, prefix="/api/v1/ml", tags=["ml"])




@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    readable = []
    for err in errors:
        loc = " -> ".join(str(l) for l in err.get("loc", []))
        readable.append(f"{loc}: {err.get('msg', 'validation error')}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation failed",
            "errors": readable,
            "hint": "Ensure Content-Type: application/json and valid JSON body",
        },
    )


# ── Schemas ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = APP_VERSION
    timestamp: datetime


# -- Sentiment --

class SentimentRequest(BaseModel):
    tickers: list[str] | None = Field(default=None, examples=[["AAPL", "TSLA"]])
    query: str | None = Field(default=None, examples=["Apple"])

class ArticleSentiment(BaseModel):
    title: str
    url: str
    sentiment: str

class TickerSentimentResult(BaseModel):
    ticker: str
    overall_score: int = Field(ge=0, le=100)
    sentiment: str
    confidence: int = Field(ge=0, le=100)
    key_factors: list[str]
    top_articles: list[ArticleSentiment]
    market_data: dict
    analyzed_at: str

class SentimentResponse(BaseModel):
    results: dict[str, TickerSentimentResult]
    count: int
    analyzed_at: datetime


# -- Portfolio --

class HoldingInput(BaseModel):
    ticker: str = Field(examples=["AAPL"])
    quantity: float = Field(gt=0, examples=[50])
    avg_price: float | None = Field(default=None, ge=0, examples=[178.50])

class PortfolioRequest(BaseModel):
    holdings: list[HoldingInput] = Field(
        min_length=1,
        examples=[[
            {"ticker": "AAPL", "quantity": 50, "avg_price": 178.50},
            {"ticker": "VOO", "quantity": 30, "avg_price": 420.00},
        ]]
    )

class CurrentAllocation(BaseModel):
    ticker: str
    weight: float
    current_price: float
    market_value: float

class RebalanceAction(BaseModel):
    ticker: str
    action: str
    current_weight: float
    target_weight: float
    conviction: float
    rationale: str

class PortfolioResponse(BaseModel):
    total_value: float
    current_allocation: list[CurrentAllocation]
    risk_metrics: dict[str, float]
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    beta: float
    annual_return: float
    annual_volatility: float
    diversification_score: float
    risk_level: str
    recommended_rebalance: list[RebalanceAction]
    suggestions: list[str]
    analyzed_at: datetime


# -- Prediction --

class PredictionRequest(BaseModel):
    tickers: list[str] = Field(examples=[["AAPL", "TSLA"]])

class ConfidenceBand(BaseModel):
    lower: float
    upper: float
    predicted_price: float

class TickerPrediction(BaseModel):
    ticker: str
    forecast: dict[str, float]
    confidence: int = Field(ge=0, le=100)
    confidence_bands: dict[str, ConfidenceBand]
    model_weights: dict[str, float]
    last_price: float
    reasoning: str
    predicted_at: str

class PredictionResponse(BaseModel):
    results: dict[str, TickerPrediction]
    count: int
    generated_at: datetime


# -- Recommendations --

class RecommendationItem(BaseModel):
    ticker: str
    action: str
    conviction: float
    rationale: str
    target_weight: float | None = None

class RecommendationsResponse(BaseModel):
    recommendations: list[RecommendationItem]
    portfolio_score_before: float
    portfolio_score_after: float
    generated_at: datetime


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to Swagger docs."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    return HealthResponse(timestamp=datetime.now(timezone.utc))


@app.get("/health/ready", tags=["system"])
async def health_ready():
    """Readiness probe — checks DB and Redis connectivity."""
    checks: dict[str, str] = {}

    # Database
    try:
        from db.database import async_engine
        from sqlalchemy import text
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {e}"

    # Redis
    try:
        import redis.asyncio as aioredis
        from config import get_settings
        r = aioredis.from_url(get_settings().redis_url)
        await r.ping()
        await r.aclose()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"

    all_ok = all(v == "ok" for v in checks.values())
    status_code = 200 if all_ok else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "ready" if all_ok else "degraded", "checks": checks, "timestamp": datetime.now(timezone.utc).isoformat()},
    )


@app.get("/health/live", tags=["system"])
async def health_live():
    """Liveness probe — always returns 200 if the process is alive."""
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/metrics", tags=["system"])
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    text = await get_metrics_text()
    return PlainTextResponse(text, media_type="text/plain; version=0.0.4")


@app.post("/api/v1/sentiment/analyze", response_model=SentimentResponse, tags=["ai"])
async def analyze_sentiment(req: SentimentRequest):
    """Analyse market sentiment via LangGraph agent (Claude 4 Opus + heuristic fallback)."""
    if not req.tickers and not req.query:
        raise HTTPException(400, "Provide at least 'tickers' or 'query'")

    from agents.sentiment_agent import run_sentiment_analysis
    raw = await run_sentiment_analysis(tickers=req.tickers, query=req.query)

    if "error" in raw:
        raise HTTPException(400, raw["error"])

    results = {}
    for ticker, data in raw.items():
        results[ticker] = TickerSentimentResult(
            ticker=data.get("ticker", ticker),
            overall_score=data.get("overall_score", 50),
            sentiment=data.get("sentiment", "neutral"),
            confidence=data.get("confidence", 0),
            key_factors=data.get("key_factors", []),
            top_articles=[ArticleSentiment(**a) for a in data.get("top_articles", [])],
            market_data=data.get("market_data", {}),
            analyzed_at=data.get("analyzed_at", datetime.now(timezone.utc).isoformat()),
        )

    return SentimentResponse(results=results, count=len(results), analyzed_at=datetime.now(timezone.utc))


@app.post("/api/v1/portfolio/analyze", response_model=PortfolioResponse, tags=["ai"])
async def analyze_portfolio(req: PortfolioRequest):
    """Full portfolio analysis via LangGraph agent (MPT + PyPortfolioOpt + RL PPO)."""
    from agents.portfolio_agent import run_portfolio_analysis

    try:
        raw = await run_portfolio_analysis([h.model_dump() for h in req.holdings])
    except Exception as e:
        raise HTTPException(500, detail=f"Portfolio analysis failed: {e}")

    if "error" in raw:
        raise HTTPException(400, detail=raw["error"])

    try:
        return PortfolioResponse(
            total_value=raw.get("total_value", 0),
            current_allocation=[CurrentAllocation(**a) for a in raw.get("current_allocation", [])],
            risk_metrics=raw.get("risk_metrics", {}),
            sharpe_ratio=raw.get("sharpe_ratio", 0),
            sortino_ratio=raw.get("sortino_ratio", 0),
            max_drawdown=raw.get("max_drawdown", 0),
            beta=raw.get("beta", 1.0),
            annual_return=raw.get("annual_return", 0),
            annual_volatility=raw.get("annual_volatility", 0),
            diversification_score=raw.get("diversification_score", 0.5),
            risk_level=raw.get("risk_level", "unknown"),
            recommended_rebalance=[RebalanceAction(**a) for a in raw.get("recommended_rebalance", [])],
            suggestions=raw.get("suggestions", []),
            analyzed_at=datetime.now(timezone.utc),
        )
    except Exception as e:
        raise HTTPException(500, detail=f"Response serialization failed: {e}")


@app.post("/api/v1/predict/trend", response_model=PredictionResponse, tags=["ai"])
async def predict_trend(req: PredictionRequest):
    """Ensemble trend prediction (Chronos-2 + TimesFM + Lag-Llama)."""
    if not req.tickers:
        raise HTTPException(400, detail="Provide at least one ticker")

    from agents.prediction_agent import run_prediction

    try:
        raw = await run_prediction(req.tickers)
    except Exception as e:
        raise HTTPException(500, detail=f"Prediction pipeline failed: {e}")

    if "error" in raw:
        raise HTTPException(400, detail=raw["error"])

    try:
        results = {}
        for ticker, data in raw.items():
            results[ticker] = TickerPrediction(
                ticker=data.get("ticker", ticker),
                forecast=data.get("forecast", {"1d": 0, "3d": 0, "7d": 0, "30d": 0}),
                confidence=data.get("confidence", 50),
                confidence_bands={
                    k: ConfidenceBand(**v)
                    for k, v in data.get("confidence_bands", {
                        h: {"lower": 0, "upper": 0, "predicted_price": 0} for h in ["1d", "3d", "7d", "30d"]
                    }).items()
                },
                model_weights=data.get("model_weights", {}),
                last_price=data.get("last_price", 0),
                reasoning=data.get("reasoning", ""),
                predicted_at=data.get("predicted_at", datetime.now(timezone.utc).isoformat()),
            )
        return PredictionResponse(results=results, count=len(results), generated_at=datetime.now(timezone.utc))
    except Exception as e:
        raise HTTPException(500, detail=f"Response serialization failed: {e}")


@app.post("/api/v1/recommendations", response_model=RecommendationsResponse, tags=["ai"])
async def get_recommendations(req: PredictionRequest):
    """AI-generated investment recommendations via the full orchestrator pipeline."""
    if not req.tickers:
        raise HTTPException(400, detail="Provide at least one ticker")

    from agents.orchestrator import run_full_analysis

    try:
        result = await run_full_analysis(tickers=req.tickers)
    except Exception as e:
        raise HTTPException(500, detail=f"Recommendation pipeline failed: {e}")

    if "error" in result:
        raise HTTPException(400, detail=result["error"])

    recs = result.get("recommendations", {})
    return RecommendationsResponse(
        recommendations=[RecommendationItem(**r) for r in recs.get("recommendations", [])],
        portfolio_score_before=recs.get("portfolio_score_before", 0.5),
        portfolio_score_after=recs.get("portfolio_score_after", 0.7),
        generated_at=datetime.now(timezone.utc),
    )


# ── WebSocket ─────────────────────────────────────────────────────────

@app.websocket("/api/v1/ws/quotes")
async def ws_quotes(websocket: WebSocket):
    """Stream live quote snapshots at ~5s intervals."""
    await websocket.accept()

    from data.polygon_client import PolygonClient

    tickers = ["AAPL", "VOO", "TSLA", "NVDA", "BTC-USD"]

    client = PolygonClient()
    try:
        init = await websocket.receive_text()
        try:
            msg = json.loads(init)
            if "tickers" in msg:
                tickers = [t.strip().upper() for t in msg["tickers"] if t.strip()]
        except json.JSONDecodeError:
            pass

        while True:
            quotes = []
            try:
                snapshots = await asyncio.gather(
                    *(client.get_snapshot(ticker) for ticker in tickers),
                    return_exceptions=True,
                )
                for snap in snapshots:
                    if isinstance(snap, Exception):
                        continue
                    quotes.append({
                        "ticker": snap.ticker,
                        "price": snap.price,
                        "change": snap.change,
                        "change_pct": snap.change_pct,
                        "volume": snap.volume,
                    })
            except Exception as e:
                logger.warning(f"WebSocket quote fetch error: {e}")

            await websocket.send_json({
                "type": "quotes",
                "data": quotes,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            await asyncio.sleep(5)

    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        await client.close()
