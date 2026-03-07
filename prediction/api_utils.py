import time
import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 5, cooldown: float = 30.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        self.consecutive_failures = 0
        self.last_failure_time = 0.0
        self.state = "closed"

    def is_open(self) -> bool:
        if self.state == "open":
            if time.time() - self.last_failure_time >= self.cooldown:
                self.state = "half-open"
                return False
            return True
        return False

    def record_success(self):
        self.consecutive_failures = 0
        self.state = "closed"

    def record_failure(self):
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        if self.consecutive_failures >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                f"CircuitBreaker [{self.name}] OPEN after {self.consecutive_failures} consecutive failures. "
                f"Cooldown: {self.cooldown}s"
            )

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state,
            "consecutive_failures": self.consecutive_failures,
            "last_failure_time": self.last_failure_time,
        }


_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str) -> CircuitBreaker:
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name)
    return _circuit_breakers[name]


def get_all_circuit_breakers() -> Dict[str, Dict[str, Any]]:
    return {name: cb.get_status() for name, cb in _circuit_breakers.items()}


def resilient_get(
    url: str,
    params: Optional[Dict] = None,
    retries: int = 2,
    timeout: float = 10.0,
    source: str = "unknown",
) -> Optional[httpx.Response]:
    cb = get_circuit_breaker(source)

    if cb.is_open():
        logger.debug(f"CircuitBreaker [{source}] is open, skipping request to {url}")
        return None

    last_exc = None
    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(url, params=params)
                if resp.status_code >= 500:
                    last_exc = Exception(f"HTTP {resp.status_code}")
                    if attempt < retries:
                        backoff = 0.5 * (2 ** attempt)
                        time.sleep(backoff)
                        continue
                    cb.record_failure()
                    return resp
                cb.record_success()
                return resp
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout,
                httpx.PoolTimeout, httpx.ConnectTimeout, OSError) as e:
            last_exc = e
            if attempt < retries:
                backoff = 0.5 * (2 ** attempt)
                logger.debug(f"[{source}] Retry {attempt+1}/{retries} after {backoff}s: {e}")
                time.sleep(backoff)
                continue
            cb.record_failure()
            logger.warning(f"[{source}] Request failed after {retries+1} attempts: {e}")
            return None
        except Exception as e:
            cb.record_failure()
            logger.warning(f"[{source}] Unexpected error: {e}")
            return None

    cb.record_failure()
    return None


async def async_resilient_get(
    url: str,
    params: Optional[Dict] = None,
    retries: int = 2,
    timeout: float = 15.0,
    source: str = "unknown",
) -> Optional[httpx.Response]:
    import asyncio

    cb = get_circuit_breaker(source)

    if cb.is_open():
        logger.debug(f"CircuitBreaker [{source}] is open, skipping async request to {url}")
        return None

    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url, params=params)
                if resp.status_code >= 500:
                    if attempt < retries:
                        backoff = 0.5 * (2 ** attempt)
                        await asyncio.sleep(backoff)
                        continue
                    cb.record_failure()
                    return resp
                cb.record_success()
                return resp
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout,
                httpx.PoolTimeout, httpx.ConnectTimeout, OSError) as e:
            if attempt < retries:
                backoff = 0.5 * (2 ** attempt)
                logger.debug(f"[{source}] Async retry {attempt+1}/{retries} after {backoff}s: {e}")
                await asyncio.sleep(backoff)
                continue
            cb.record_failure()
            logger.warning(f"[{source}] Async request failed after {retries+1} attempts: {e}")
            return None
        except Exception as e:
            cb.record_failure()
            logger.warning(f"[{source}] Unexpected async error: {e}")
            return None

    cb.record_failure()
    return None
