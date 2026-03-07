"""
System monitoring and status endpoints.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from config import get_settings

router = APIRouter()


@router.get("/status")
async def system_status():
    try:
        from scheduler import get_scheduler_status
        status = get_scheduler_status()
    except Exception:
        status = {}
    status["version"] = get_settings().version
    return status


@router.get("/cache-stats")
async def system_cache_stats():
    from prediction.cache_manager import cache_stats
    stats = cache_stats()
    stats["timestamp"] = datetime.now(timezone.utc).isoformat()
    return stats


@router.get("/health")
async def system_health():
    from prediction.api_utils import get_all_circuit_breakers
    from prediction.cache_manager import cache_stats as _cache_stats

    circuit_breakers = get_all_circuit_breakers()
    api_sources = {}

    for src in ["polygon", "fmp", "finnhub", "alpha_vantage", "yfinance"]:
        if src in circuit_breakers:
            cb = circuit_breakers[src]
            state = cb["state"]
            failures = cb["consecutive_failures"]
            last_fail = cb["last_failure_time"]
            if state == "closed" and failures == 0:
                status = "healthy"
            elif state == "half-open" or (state == "closed" and failures > 0):
                status = "degraded"
            else:
                status = "down"
            api_sources[src] = {
                "status": status,
                "circuit_breaker_state": state,
                "consecutive_failures": failures,
                "last_failure_time": datetime.fromtimestamp(last_fail, tz=timezone.utc).isoformat() if last_fail > 0 else None,
            }
        else:
            api_sources[src] = {"status": "unknown", "circuit_breaker_state": "no_data"}

    cache = _cache_stats()
    overall_statuses = [s["status"] for s in api_sources.values()]

    if all(s == "healthy" for s in overall_statuses):
        overall = "healthy"
    elif any(s == "down" for s in overall_statuses):
        overall = "degraded"
    else:
        overall = "unknown"

    return {
        "overall_status": overall,
        "api_sources": api_sources,
        "cache": cache,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
