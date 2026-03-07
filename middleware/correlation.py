"""
Request correlation ID middleware for FastAPI/ASGI applications.

Assigns a unique UUID4 to every inbound HTTP request (or reuses the caller-
supplied ``X-Request-ID`` header) and makes it available via *contextvars* so
that any downstream code -- especially structured loggers -- can include the
correlation ID without explicit parameter passing.

Also measures wall-clock request duration and injects an ``X-Response-Time``
header (value in milliseconds).
"""

from __future__ import annotations

import contextvars
import time
import uuid
from typing import Any, Callable, List, Optional, Tuple

correlation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)

request_start_var: contextvars.ContextVar[Optional[float]] = contextvars.ContextVar(
    "request_start", default=None
)


def get_correlation_id() -> Optional[str]:
    """Return the correlation ID for the current request context."""
    return correlation_id_var.get()


class CorrelationMiddleware:
    """ASGI middleware that manages per-request correlation IDs and timing.

    Parameters
    ----------
    app:
        The wrapped ASGI application.
    header_name:
        HTTP header used to propagate the correlation ID (default
        ``X-Request-ID``).
    """

    def __init__(self, app: Any, header_name: str = "X-Request-ID") -> None:
        self.app = app
        self.header_name = header_name
        self._header_name_lower = header_name.lower().encode("latin-1")
        self._header_name_bytes = header_name.encode("latin-1")

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = self._extract_or_generate(scope)
        start_time = time.perf_counter()

        token_cid = correlation_id_var.set(request_id)
        token_start = request_start_var.set(start_time)

        async def send_with_headers(message: dict) -> None:
            if message["type"] == "http.response.start":
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                headers: List[List[bytes]] = list(message.get("headers", []))
                headers.append(
                    [self._header_name_bytes, request_id.encode("latin-1")]
                )
                headers.append(
                    [b"x-response-time", f"{elapsed_ms:.2f}ms".encode("latin-1")]
                )
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        finally:
            correlation_id_var.reset(token_cid)
            request_start_var.reset(token_start)

    def _extract_or_generate(self, scope: dict) -> str:
        """Return the caller-supplied request ID or generate a new UUID4."""
        headers: List[Tuple[bytes, bytes]] = scope.get("headers", [])
        for name, value in headers:
            if name.lower() == self._header_name_lower:
                candidate = value.decode("latin-1").strip()
                if candidate:
                    return candidate
        return str(uuid.uuid4())
