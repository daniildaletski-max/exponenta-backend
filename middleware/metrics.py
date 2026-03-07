"""
Prometheus-compatible metrics collection middleware for FastAPI/ASGI.

Collects four metric families with zero external dependencies:

* **http_requests_total** -- counter, labels: method, path, status
* **http_request_duration_seconds** -- histogram, labels: method, path
* **http_active_websocket_connections** -- gauge (current count)
* **http_errors_total** -- counter, labels: method, path, status

Call :func:`get_metrics_text` to obtain the current snapshot in the
Prometheus text exposition format (``text/plain; version=0.0.4``).
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple

LATENCY_BUCKETS: Tuple[float, ...] = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)

# ---------------------------------------------------------------------------
# Internal storage -- module-level singletons guarded by an asyncio.Lock
# ---------------------------------------------------------------------------
_lock = asyncio.Lock()

_request_counts: Dict[Tuple[str, str, int], int] = defaultdict(int)

_error_counts: Dict[Tuple[str, str, int], int] = defaultdict(int)

_histogram_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
_histogram_sums: Dict[Tuple[str, str], float] = defaultdict(float)
_histogram_totals: Dict[Tuple[str, str], int] = defaultdict(int)

_ws_connections: int = 0


def _normalize_path(path: str) -> str:
    """Collapse path segments that look like IDs into a placeholder.

    This keeps the cardinality of the ``path`` label manageable.  Segments
    that are pure digits, valid UUIDs (length 32/36 hex), or longer than 30
    characters are replaced with ``{id}``.
    """
    parts = path.split("/")
    normalized: List[str] = []
    for part in parts:
        if not part:
            normalized.append(part)
            continue
        if part.isdigit():
            normalized.append("{id}")
            continue
        stripped = part.replace("-", "")
        if len(stripped) in (32, 36) and all(c in "0123456789abcdefABCDEF" for c in stripped):
            normalized.append("{id}")
            continue
        if len(part) > 30:
            normalized.append("{id}")
            continue
        normalized.append(part)
    return "/".join(normalized)


def _bucket_label(le: float) -> str:
    if le == float("inf"):
        return "+Inf"
    if le == int(le):
        return f"{int(le)}"
    return f"{le:g}"


class MetricsMiddleware:
    """ASGI middleware that records per-request metrics.

    Parameters
    ----------
    app:
        The wrapped ASGI application.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] == "websocket":
            await self._handle_websocket(scope, receive, send)
            return

        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method: str = scope.get("method", "GET")
        path: str = _normalize_path(scope.get("path", "/"))
        start = time.perf_counter()

        captured_status: List[int] = [0]

        async def send_wrapper(message: dict) -> None:
            if message["type"] == "http.response.start":
                captured_status[0] = message.get("status", 0)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            captured_status[0] = captured_status[0] or 500
            raise
        finally:
            elapsed = time.perf_counter() - start
            status = captured_status[0]
            await self._record(method, path, status, elapsed)

    # ------------------------------------------------------------------
    # WebSocket tracking
    # ------------------------------------------------------------------
    async def _handle_websocket(self, scope: dict, receive: Callable, send: Callable) -> None:
        global _ws_connections

        async with _lock:
            _ws_connections += 1
        try:
            await self.app(scope, receive, send)
        finally:
            async with _lock:
                _ws_connections = max(0, _ws_connections - 1)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    @staticmethod
    async def _record(method: str, path: str, status: int, elapsed: float) -> None:
        async with _lock:
            _request_counts[(method, path, status)] += 1

            if status >= 400:
                _error_counts[(method, path, status)] += 1

            key_mp = (method, path)
            _histogram_sums[key_mp] += elapsed
            _histogram_totals[key_mp] += 1

            for bucket in LATENCY_BUCKETS:
                if elapsed <= bucket:
                    _histogram_counts[(method, path, _bucket_label(bucket))] += 1
            _histogram_counts[(method, path, "+Inf")] += 1


# ---------------------------------------------------------------------------
# Exposition
# ---------------------------------------------------------------------------

async def get_metrics_text() -> str:
    """Return all collected metrics in Prometheus text exposition format.

    This is an *async* function because it acquires the module-level lock
    to produce a consistent snapshot.
    """
    async with _lock:
        return _render()


def _render() -> str:
    lines: List[str] = []

    # -- http_requests_total (counter) --
    lines.append("# HELP http_requests_total Total number of HTTP requests.")
    lines.append("# TYPE http_requests_total counter")
    for (method, path, status), count in sorted(_request_counts.items()):
        lines.append(
            f'http_requests_total{{method="{method}",path="{path}",status="{status}"}} {count}'
        )

    # -- http_errors_total (counter) --
    lines.append("# HELP http_errors_total Total number of HTTP error responses (4xx/5xx).")
    lines.append("# TYPE http_errors_total counter")
    for (method, path, status), count in sorted(_error_counts.items()):
        lines.append(
            f'http_errors_total{{method="{method}",path="{path}",status="{status}"}} {count}'
        )

    # -- http_request_duration_seconds (histogram) --
    lines.append(
        "# HELP http_request_duration_seconds Histogram of HTTP request durations in seconds."
    )
    lines.append("# TYPE http_request_duration_seconds histogram")

    seen_keys: set = set()
    for (method, path, _le) in sorted(_histogram_counts.keys()):
        key_mp = (method, path)
        if key_mp in seen_keys:
            continue
        seen_keys.add(key_mp)

        for bucket in LATENCY_BUCKETS:
            le_label = _bucket_label(bucket)
            val = _histogram_counts.get((method, path, le_label), 0)
            lines.append(
                f'http_request_duration_seconds_bucket{{method="{method}",path="{path}",le="{le_label}"}} {val}'
            )
        inf_val = _histogram_counts.get((method, path, "+Inf"), 0)
        lines.append(
            f'http_request_duration_seconds_bucket{{method="{method}",path="{path}",le="+Inf"}} {inf_val}'
        )
        lines.append(
            f'http_request_duration_seconds_sum{{method="{method}",path="{path}"}} {_histogram_sums[key_mp]:.6f}'
        )
        lines.append(
            f'http_request_duration_seconds_count{{method="{method}",path="{path}"}} {_histogram_totals[key_mp]}'
        )

    # -- http_active_websocket_connections (gauge) --
    lines.append(
        "# HELP http_active_websocket_connections Current number of active WebSocket connections."
    )
    lines.append("# TYPE http_active_websocket_connections gauge")
    lines.append(f"http_active_websocket_connections {_ws_connections}")

    lines.append("")  # trailing newline
    return "\n".join(lines)
