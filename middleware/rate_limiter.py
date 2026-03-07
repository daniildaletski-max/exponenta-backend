"""
Token-bucket rate limiter middleware for FastAPI/ASGI applications.

Provides two tiers of rate limiting:
  - Default: 60 requests/minute per client IP
  - AI/GPU endpoints: 10 requests/minute per client IP

Uses in-memory counters with periodic cleanup of stale entries.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

_AI_PATH_SEGMENTS: Tuple[str, ...] = ("/ai/", "/gpu/", "/predict/", "/sentiment/")

_CLEANUP_INTERVAL_SECONDS: float = 300.0  # 5 minutes


class _ClientBucket:
    """Sliding-window counter for a single client IP on a single tier."""

    __slots__ = ("tokens", "last_refill", "rpm")

    def __init__(self, rpm: int) -> None:
        self.rpm: int = rpm
        self.tokens: float = float(rpm)
        self.last_refill: float = time.monotonic()

    def consume(self) -> bool:
        """Try to consume one token.  Returns True if the request is allowed."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(float(self.rpm), self.tokens + elapsed * (self.rpm / 60.0))
        self.last_refill = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    def retry_after(self) -> float:
        """Seconds until at least one token is available."""
        if self.tokens >= 1.0:
            return 0.0
        deficit = 1.0 - self.tokens
        return deficit / (self.rpm / 60.0)


class RateLimitMiddleware:
    """ASGI middleware that enforces per-IP token-bucket rate limits.

    Parameters
    ----------
    app:
        The wrapped ASGI application.
    default_rpm:
        Requests per minute allowed for regular endpoints (default 60).
    ai_rpm:
        Requests per minute allowed for AI/GPU-heavy endpoints (default 10).
    """

    def __init__(
        self,
        app: Any,
        default_rpm: int = 60,
        ai_rpm: int = 10,
    ) -> None:
        self.app = app
        self.default_rpm = default_rpm
        self.ai_rpm = ai_rpm

        self._buckets_default: Dict[str, _ClientBucket] = {}
        self._buckets_ai: Dict[str, _ClientBucket] = {}
        self._lock = asyncio.Lock()

        self._last_cleanup: float = time.monotonic()

    # ------------------------------------------------------------------
    # ASGI interface
    # ------------------------------------------------------------------
    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        client_ip = self._extract_client_ip(scope)
        path: str = scope.get("path", "")
        is_ai = any(seg in path for seg in _AI_PATH_SEGMENTS)

        async with self._lock:
            await self._maybe_cleanup()
            allowed, retry_after = self._check(client_ip, is_ai)

        if not allowed:
            await self._send_429(send, retry_after)
            return

        await self.app(scope, receive, send)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_client_ip(scope: dict) -> str:
        """Return the client IP from the ASGI scope.

        Checks the ``X-Forwarded-For`` header first (first entry) so the
        middleware works correctly behind reverse proxies, then falls back
        to the direct peer address.
        """
        headers: List[Tuple[bytes, bytes]] = scope.get("headers", [])
        for name, value in headers:
            if name.lower() == b"x-forwarded-for":
                forwarded = value.decode("latin-1").split(",")[0].strip()
                if forwarded:
                    return forwarded

        client = scope.get("client")
        if client:
            return client[0]
        return "unknown"

    def _check(self, client_ip: str, is_ai: bool) -> Tuple[bool, float]:
        """Check the appropriate bucket and return (allowed, retry_after)."""
        if is_ai:
            bucket_map = self._buckets_ai
            rpm = self.ai_rpm
        else:
            bucket_map = self._buckets_default
            rpm = self.default_rpm

        bucket = bucket_map.get(client_ip)
        if bucket is None:
            bucket = _ClientBucket(rpm)
            bucket_map[client_ip] = bucket

        if bucket.consume():
            return True, 0.0
        return False, bucket.retry_after()

    async def _maybe_cleanup(self) -> None:
        """Remove stale bucket entries if enough time has passed."""
        now = time.monotonic()
        if now - self._last_cleanup < _CLEANUP_INTERVAL_SECONDS:
            return
        self._last_cleanup = now

        cutoff = now - 120.0  # entries idle for > 2 minutes
        for bucket_map in (self._buckets_default, self._buckets_ai):
            stale_keys = [
                ip
                for ip, bucket in bucket_map.items()
                if bucket.last_refill < cutoff
            ]
            for key in stale_keys:
                del bucket_map[key]

    @staticmethod
    async def _send_429(send: Callable, retry_after: float) -> None:
        """Send an HTTP 429 Too Many Requests response."""
        retry_after_int = max(1, int(retry_after + 0.5))
        body = b'{"detail":"Rate limit exceeded. Please retry later."}'
        await send(
            {
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"retry-after", str(retry_after_int).encode()],
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
            }
        )
