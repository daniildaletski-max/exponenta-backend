"""
Polygon.io market data connector.

Provides real-time and historical OHLCV data, trades, and quotes.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import structlog

from config import get_settings

log = structlog.get_logger()


class PolygonConnector:
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        settings = get_settings()
        if not settings.polygon_api_key:
            log.warning("polygon.no_api_key — returning synthetic data")
            return None
        try:
            from polygon import RESTClient
            self._client = RESTClient(api_key=settings.polygon_api_key)
            return self._client
        except ImportError:
            log.warning("polygon.not_installed")
            return None

    def fetch_ohlcv(
        self,
        ticker: str,
        lookback_days: int = 365,
    ) -> dict[str, np.ndarray]:
        """Fetch daily OHLCV bars for a ticker."""
        client = self._get_client()
        end = date.today()
        start = end - timedelta(days=lookback_days)

        if client is not None:
            bars = client.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=start.isoformat(),
                to=end.isoformat(),
                limit=5000,
            )
            return {
                "open": np.array([b.open for b in bars]),
                "high": np.array([b.high for b in bars]),
                "low": np.array([b.low for b in bars]),
                "close": np.array([b.close for b in bars]),
                "volume": np.array([b.volume for b in bars]),
            }

        # Synthetic data for development
        rng = np.random.default_rng(hash(ticker) % 2**32)
        n = min(lookback_days, 252)
        base = 100 + rng.uniform(0, 200)
        returns = rng.normal(0.0003, 0.015, n)
        close = base * np.cumprod(1 + returns)
        return {
            "open": close * (1 + rng.uniform(-0.01, 0.01, n)),
            "high": close * (1 + rng.uniform(0, 0.02, n)),
            "low": close * (1 - rng.uniform(0, 0.02, n)),
            "close": close,
            "volume": rng.integers(1_000_000, 50_000_000, n),
        }

    def fetch_quote(self, ticker: str) -> dict:
        """Fetch latest quote snapshot."""
        client = self._get_client()
        if client is not None:
            snapshot = client.get_snapshot_ticker("stocks", ticker)
            return {
                "price": snapshot.day.close,
                "change": snapshot.today_change,
                "change_pct": snapshot.today_change_percent,
                "volume": snapshot.day.volume,
            }

        # Stub
        return {"price": 195.30, "change": 2.10, "change_pct": 1.09, "volume": 54_200_000}
