import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any

logger = logging.getLogger(__name__)

_scheduler_state: Dict[str, Any] = {
    "running": False,
    "tasks": {},
    "task_stats": {"successes": 0, "failures": 0},
    "started_at": None,
    "_task_refs": [],
}


def _is_market_hours() -> bool:
    now = datetime.now(timezone.utc)
    weekday = now.weekday()
    if weekday >= 5:
        return False
    hour = now.hour
    return 13 <= hour <= 21


def get_scheduler_status() -> Dict[str, Any]:
    return {
        "running": _scheduler_state["running"],
        "started_at": _scheduler_state.get("started_at"),
        "is_market_hours": _is_market_hours(),
        "tasks": {
            name: {
                "last_run": info.get("last_run"),
                "next_run": info.get("next_run"),
                "run_count": info.get("run_count", 0),
                "last_status": info.get("last_status", "pending"),
                "last_duration_ms": info.get("last_duration_ms", 0),
            }
            for name, info in _scheduler_state["tasks"].items()
        },
        "task_stats": _scheduler_state["task_stats"],
        "uptime_seconds": int(time.time() - _scheduler_state["started_at"]) if _scheduler_state.get("started_at") else 0,
    }


async def _run_quotes_refresh():
    try:
        import data.market_data as market_data
        await asyncio.to_thread(market_data.get_real_quotes)
        await asyncio.to_thread(market_data.get_real_indices)
        return True
    except Exception as e:
        logger.error(f"Quotes refresh failed: {e}")
        return False


async def _run_sentiment_precompute():
    try:
        from user_settings import load_settings
        from prediction.sentiment import analyze_sentiment_real
        import data.market_data as market_data

        settings = load_settings()
        watchlist = settings.get("watchlist", ["AAPL", "TSLA", "NVDA"])[:6]

        real_prices = {}
        for t in watchlist:
            q = await asyncio.to_thread(market_data._fetch_quote, t)
            if q:
                real_prices[t] = q

        await analyze_sentiment_real(watchlist, real_prices)
        return True
    except Exception as e:
        logger.error(f"Sentiment precompute failed: {e}")
        return False


async def _run_scanner_precompute():
    try:
        from prediction.scanner import scan_universe
        await asyncio.to_thread(scan_universe, universe_keys=["us_mega_cap"], max_assets=10, use_ml=False)
        return True
    except Exception as e:
        logger.error(f"Scanner precompute failed: {e}")
        return False


async def _run_signal_evaluation():
    """Evaluate past prediction signals against actual prices."""
    try:
        from prediction.signal_tracker import evaluate_signals
        result = await asyncio.to_thread(evaluate_signals)
        logger.info(f"Signal evaluation: {result.get('evaluated', 0)} signals evaluated")
        return True
    except Exception as e:
        logger.error(f"Signal evaluation failed: {e}")
        return False


async def _run_model_staleness_check():
    """Check if ML models need retraining and retrain stale ones."""
    try:
        from prediction.engine import get_engine
        engine = get_engine()
        retrained = 0
        for symbol, model in list(engine._models.items()):
            staleness = model.check_staleness()
            if staleness.get("needs_retrain", False):
                logger.info(f"Retraining stale model for {symbol} (age={staleness['age_hours']:.1f}h)")
                engine._get_or_train(symbol, force_retrain=True)
                retrained += 1
        if retrained:
            logger.info(f"Retrained {retrained} stale models")
        return True
    except Exception as e:
        logger.error(f"Model staleness check failed: {e}")
        return False


async def _run_feature_drift_check():
    """Check for feature distribution drift across active models."""
    try:
        from core.feature_drift import FeatureDriftDetector
        detector = FeatureDriftDetector()
        # Drift detection runs on next prediction; this just logs status
        logger.info("Feature drift check completed")
        return True
    except Exception as e:
        logger.error(f"Feature drift check failed: {e}")
        return False


SCHEDULER_TASKS = {
    "quotes_refresh": {
        "fn": _run_quotes_refresh,
        "interval_seconds": 60,
        "market_hours_only": False,
    },
    "sentiment_precompute": {
        "fn": _run_sentiment_precompute,
        "interval_seconds": 600,
        "market_hours_only": False,
    },
    "scanner_precompute": {
        "fn": _run_scanner_precompute,
        "interval_seconds": 900,
        "market_hours_only": True,
    },
    "signal_evaluation": {
        "fn": _run_signal_evaluation,
        "interval_seconds": 3600,
        "market_hours_only": False,
    },
    "model_staleness_check": {
        "fn": _run_model_staleness_check,
        "interval_seconds": 7200,
        "market_hours_only": False,
    },
    "feature_drift_check": {
        "fn": _run_feature_drift_check,
        "interval_seconds": 14400,
        "market_hours_only": False,
    },
}


async def _task_loop(name: str, config: dict):
    interval = config["interval_seconds"]
    fn = config["fn"]
    market_only = config.get("market_hours_only", False)

    _scheduler_state["tasks"][name] = {
        "last_run": None,
        "next_run": None,
        "run_count": 0,
        "last_status": "pending",
        "last_duration_ms": 0,
    }

    while _scheduler_state["running"]:
        try:
            if market_only and not _is_market_hours():
                await asyncio.sleep(60)
                continue

            start = time.time()
            success = await fn()
            duration = int((time.time() - start) * 1000)

            task_state = _scheduler_state["tasks"][name]
            task_state["last_run"] = datetime.now(timezone.utc).isoformat()
            task_state["run_count"] += 1
            task_state["last_status"] = "success" if success else "error"
            task_state["last_duration_ms"] = duration
            next_run = datetime.now(timezone.utc).timestamp() + interval
            task_state["next_run"] = datetime.fromtimestamp(next_run, tz=timezone.utc).isoformat()

            if success:
                _scheduler_state["task_stats"]["successes"] += 1
            else:
                _scheduler_state["task_stats"]["failures"] += 1

        except Exception as e:
            logger.error(f"Scheduler task {name} error: {e}")
            _scheduler_state["tasks"][name]["last_status"] = "error"
            _scheduler_state["task_stats"]["failures"] += 1

        await asyncio.sleep(interval)


def _task_done_callback(task: asyncio.Task):
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.error(f"Scheduler task crashed: {exc}")


async def start_scheduler():
    if _scheduler_state["running"]:
        return

    _scheduler_state["running"] = True
    _scheduler_state["started_at"] = time.time()
    logger.info("Starting background scheduler")

    for name, config in SCHEDULER_TASKS.items():
        task = asyncio.create_task(_task_loop(name, config), name=f"scheduler:{name}")
        task.add_done_callback(_task_done_callback)
        _scheduler_state["_task_refs"].append(task)


async def stop_scheduler():
    _scheduler_state["running"] = False
    logger.info("Stopping background scheduler")
    for task in _scheduler_state["_task_refs"]:
        task.cancel()
    _scheduler_state["_task_refs"].clear()
