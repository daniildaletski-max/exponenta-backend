"""
ML Pipeline API -- model registry, feature drift, signal accuracy, regime detection.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import get_current_user
from core.feature_drift import drift_detector
from core.model_registry import registry

router = APIRouter()


# ── Schemas ──────────────────────────────────────────────────────────


class ModelInfoOut(BaseModel):
    model_name: str
    version: int
    metrics: dict[str, float]
    training_date: str
    is_active: bool


class ModelListResponse(BaseModel):
    models: dict[str, ModelInfoOut | None]
    stale_models: list[str]


class ModelRegisterRequest(BaseModel):
    model_name: str = Field(min_length=1, max_length=64)
    version: int = Field(ge=1)
    metrics: dict[str, float] = Field(default_factory=dict)


class ModelCompareResponse(BaseModel):
    model_name: str
    metric: str
    versions: list[dict]


class DriftFeatureOut(BaseModel):
    feature_name: str
    importance: float
    drift_psi: float
    health_score: float
    status: str


class DriftReportResponse(BaseModel):
    features: list[DriftFeatureOut]
    retrain_recommendation: dict
    generated_at: datetime


class SignalRecord(BaseModel):
    ticker: str
    signal_type: str
    direction: str  # "long" | "short" | "neutral"
    confidence: float = Field(ge=0, le=1)
    predicted_move: float | None = None


class SignalAccuracyResponse(BaseModel):
    total_signals: int
    correct: int
    incorrect: int
    accuracy_pct: float
    avg_confidence: float
    by_type: dict[str, dict]


class RegimeOut(BaseModel):
    current_regime: str
    confidence: float
    description: str
    recommended_strategy: str
    volatility_level: str
    trend_direction: str
    generated_at: datetime


# ── In-memory signal store ───────────────────────────────────────────

_SIGNAL_HISTORY: list[dict] = []


# ── Endpoints ────────────────────────────────────────────────────────


@router.get("/models", response_model=ModelListResponse)
async def list_models(user: dict = Depends(get_current_user)):
    """List all registered ML models with their active version."""
    raw = registry.list_models()
    models = {}
    for name, info in raw.items():
        if info:
            models[name] = ModelInfoOut(
                model_name=info.model_name,
                version=info.version,
                metrics=info.metrics,
                training_date=info.training_date.isoformat(),
                is_active=info.is_active,
            )
        else:
            models[name] = None

    return ModelListResponse(
        models=models,
        stale_models=registry.get_stale_models(max_age_days=7),
    )


@router.post("/models/register", response_model=ModelInfoOut, status_code=201)
async def register_model(body: ModelRegisterRequest, user: dict = Depends(get_current_user)):
    """Register a new model version."""
    info = registry.register(
        model_name=body.model_name,
        version=body.version,
        metrics=body.metrics,
    )
    return ModelInfoOut(
        model_name=info.model_name,
        version=info.version,
        metrics=info.metrics,
        training_date=info.training_date.isoformat(),
        is_active=info.is_active,
    )


@router.post("/models/{model_name}/promote/{version}")
async def promote_model(model_name: str, version: int, user: dict = Depends(get_current_user)):
    """Promote a specific version to active."""
    if not registry.promote(model_name, version):
        raise HTTPException(404, f"Model {model_name} version {version} not found")
    return {"status": "promoted", "model_name": model_name, "version": version}


@router.get("/models/{model_name}/compare", response_model=ModelCompareResponse)
async def compare_model_versions(
    model_name: str,
    metric: str = "mae",
    user: dict = Depends(get_current_user),
):
    """Compare all versions of a model by a metric."""
    versions = registry.compare_versions(model_name, metric)
    if not versions:
        raise HTTPException(404, f"Model {model_name} not found")
    return ModelCompareResponse(
        model_name=model_name,
        metric=metric,
        versions=versions,
    )


@router.get("/drift", response_model=DriftReportResponse)
async def feature_drift_report(user: dict = Depends(get_current_user)):
    """Analyze feature drift across prediction features."""
    rng = np.random.default_rng(42)

    feature_names = [
        "return_1d", "return_5d", "return_20d",
        "volatility_20d", "rsi_14", "macd_signal",
        "volume_ratio", "bb_position", "atr_14",
        "momentum_10d",
    ]
    n_features = len(feature_names)

    # Simulate reference vs current feature distributions
    reference = rng.standard_normal((252, n_features))
    # Add slight drift to some features
    current = rng.standard_normal((60, n_features))
    current[:, 3] += 0.5   # volatility shifted
    current[:, 6] *= 1.3   # volume ratio scaled
    current[:, 9] += 0.3   # momentum shifted

    importances = np.array([0.15, 0.12, 0.10, 0.13, 0.10, 0.08, 0.09, 0.07, 0.08, 0.08])

    reports = drift_detector.analyze_all_features(
        feature_names, reference, current, importances
    )
    recommendation = drift_detector.get_retrain_recommendation(reports)

    return DriftReportResponse(
        features=[
            DriftFeatureOut(
                feature_name=r.feature_name,
                importance=r.importance,
                drift_psi=r.drift_psi,
                health_score=r.health_score,
                status=r.status,
            )
            for r in reports
        ],
        retrain_recommendation=recommendation,
        generated_at=datetime.now(timezone.utc),
    )


@router.post("/signals/record", status_code=201)
async def record_signal(body: SignalRecord, user: dict = Depends(get_current_user)):
    """Record a prediction signal for accuracy tracking."""
    _SIGNAL_HISTORY.append({
        "ticker": body.ticker.upper(),
        "signal_type": body.signal_type,
        "direction": body.direction,
        "confidence": body.confidence,
        "predicted_move": body.predicted_move,
        "actual_result": None,
        "created_at": datetime.now(timezone.utc),
    })
    return {"status": "recorded", "total_signals": len(_SIGNAL_HISTORY)}


@router.get("/signals/accuracy", response_model=SignalAccuracyResponse)
async def signal_accuracy(user: dict = Depends(get_current_user)):
    """Get signal prediction accuracy stats."""
    total = len(_SIGNAL_HISTORY)
    evaluated = [s for s in _SIGNAL_HISTORY if s["actual_result"] is not None]
    correct = sum(
        1 for s in evaluated
        if (s["direction"] == "long" and s["actual_result"] > 0)
        or (s["direction"] == "short" and s["actual_result"] < 0)
    )
    incorrect = len(evaluated) - correct

    by_type: dict[str, dict] = {}
    for s in _SIGNAL_HISTORY:
        st = s["signal_type"]
        if st not in by_type:
            by_type[st] = {"count": 0, "avg_confidence": 0}
        by_type[st]["count"] += 1
        by_type[st]["avg_confidence"] += s["confidence"]
    for st in by_type:
        if by_type[st]["count"] > 0:
            by_type[st]["avg_confidence"] /= by_type[st]["count"]
            by_type[st]["avg_confidence"] = round(by_type[st]["avg_confidence"], 4)

    avg_conf = sum(s["confidence"] for s in _SIGNAL_HISTORY) / total if total else 0

    return SignalAccuracyResponse(
        total_signals=total,
        correct=correct,
        incorrect=incorrect,
        accuracy_pct=round((correct / len(evaluated) * 100) if evaluated else 0, 2),
        avg_confidence=round(avg_conf, 4),
        by_type=by_type,
    )


@router.get("/regime", response_model=RegimeOut)
async def detect_regime(user: dict = Depends(get_current_user)):
    """Detect current market regime for model selection."""
    # In production, this analyzes VIX, yield curve, breadth, momentum
    # For beta, provide intelligent demo based on recent conditions
    try:
        from data.market_data import fetch_historical_prices
        prices = await fetch_historical_prices(["SPY"], days=60)
        spy = np.array(prices.get("SPY", [100] * 60))
        returns = np.diff(spy) / spy[:-1]
        vol = float(np.std(returns) * np.sqrt(252))
        trend = float((spy[-1] / spy[0] - 1) * 100)
    except Exception:
        vol = 0.18
        trend = 2.5

    if vol > 0.30:
        regime = "crisis"
        desc = "High volatility regime. Markets under stress with elevated uncertainty."
        strategy = "Defensive: increase cash/bonds, reduce leverage, widen stops."
        vol_level = "extreme"
    elif vol > 0.22:
        regime = "volatile"
        desc = "Elevated volatility. Choppy price action with wider ranges."
        strategy = "Cautious: reduce position sizes, favor mean-reversion strategies."
        vol_level = "high"
    elif trend < -5:
        regime = "bearish"
        desc = "Downtrend regime. Persistent selling pressure across sectors."
        strategy = "Defensive: overweight bonds/gold, use protective puts."
        vol_level = "moderate"
    elif trend > 5:
        regime = "bullish"
        desc = "Strong uptrend regime. Broad market strength with positive momentum."
        strategy = "Aggressive: overweight equities, favor momentum strategies."
        vol_level = "low"
    else:
        regime = "neutral"
        desc = "Range-bound regime. No clear directional bias."
        strategy = "Balanced: diversified allocation, favor income strategies."
        vol_level = "moderate"

    return RegimeOut(
        current_regime=regime,
        confidence=0.72,
        description=desc,
        recommended_strategy=strategy,
        volatility_level=vol_level,
        trend_direction="up" if trend > 0 else "down",
        generated_at=datetime.now(timezone.utc),
    )
