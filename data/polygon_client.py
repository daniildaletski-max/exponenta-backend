"""
Async Polygon.io client for Exponenta.

Provides news headlines and market snapshots via the polygon-api-client
library, wrapped in async helpers for use inside FastAPI endpoints and
LangGraph agents.

Gracefully degrades to stub data when POLYGON_API_KEY is missing, so
local development works without credentials.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, date, timedelta
from dataclasses import dataclass, field

import httpx
from dotenv import load_dotenv

load_dotenv()

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
POLYGON_BASE = "https://api.polygon.io"


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class TickerNews:
    title: str
    author: str
    published_utc: str
    article_url: str
    description: str = ""
    tickers: list[str] = field(default_factory=list)


@dataclass
class TickerSnapshot:
    ticker: str
    price: float
    change: float
    change_pct: float
    volume: int
    prev_close: float
    updated: str


# ── Client ────────────────────────────────────────────────────────────

class PolygonClient:
    """Lightweight async wrapper around Polygon.io REST v3 endpoints."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or POLYGON_API_KEY
        self._http: httpx.AsyncClient | None = None

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                base_url=POLYGON_BASE,
                timeout=15.0,
                params={"apiKey": self.api_key},
            )
        return self._http

    async def close(self):
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    # ── News ──────────────────────────────────────────────────────────

    async def get_news(
        self,
        ticker: str,
        limit: int = 10,
    ) -> list[TickerNews]:
        """
        Fetch recent news articles for a ticker via
        GET /v2/reference/news?ticker=...
        """
        if not self.api_key:
            return self._stub_news(ticker)

        try:
            http = await self._client()
            resp = await http.get(
                "/v2/reference/news",
                params={"ticker": ticker, "limit": limit, "order": "desc", "sort": "published_utc"},
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                TickerNews(
                    title=a.get("title", ""),
                    author=a.get("author", ""),
                    published_utc=a.get("published_utc", ""),
                    article_url=a.get("article_url", ""),
                    description=a.get("description", ""),
                    tickers=a.get("tickers", []),
                )
                for a in data.get("results", [])
            ]
        except Exception:
            return self._stub_news(ticker)

    # ── Snapshot ──────────────────────────────────────────────────────

    async def get_snapshot(self, ticker: str) -> TickerSnapshot:
        """
        Fetch latest price snapshot via
        GET /v2/snapshot/locale/us/markets/stocks/tickers/{ticker}
        """
        if not self.api_key:
            return self._stub_snapshot(ticker)

        try:
            http = await self._client()
            resp = await http.get(
                f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}",
            )
            resp.raise_for_status()
            t = resp.json().get("ticker", {})
            day = t.get("day", {})
            return TickerSnapshot(
                ticker=t.get("ticker", ticker),
                price=day.get("c", 0),
                change=t.get("todaysChange", 0),
                change_pct=t.get("todaysChangePerc", 0),
                volume=day.get("v", 0),
                prev_close=t.get("prevDay", {}).get("c", 0),
                updated=t.get("updated", ""),
            )
        except Exception:
            return self._stub_snapshot(ticker)

    # ── Aggregates (daily bars) ───────────────────────────────────────

    async def get_daily_bars(
        self,
        ticker: str,
        lookback_days: int = 90,
    ) -> list[dict]:
        """
        Fetch daily OHLCV bars via
        GET /v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}
        """
        end = date.today()
        start = end - timedelta(days=lookback_days)

        if not self.api_key:
            return self._stub_bars(ticker, lookback_days)

        try:
            http = await self._client()
            resp = await http.get(
                f"/v2/aggs/ticker/{ticker}/range/1/day/{start.isoformat()}/{end.isoformat()}",
                params={"adjusted": "true", "sort": "asc", "limit": 5000},
            )
            resp.raise_for_status()
            return [
                {"date": b["t"], "open": b["o"], "high": b["h"],
                 "low": b["l"], "close": b["c"], "volume": b["v"]}
                for b in resp.json().get("results", [])
            ]
        except Exception:
            return self._stub_bars(ticker, lookback_days)

    # ── Stubs for local dev ───────────────────────────────────────────

    @staticmethod
    def _stub_news(ticker: str) -> list[TickerNews]:
        now = datetime.now(timezone.utc).isoformat()
        return [
            TickerNews(
                title=f"{ticker} beats Q4 earnings expectations, stock rallies 4%",
                author="Market Watch",
                published_utc=now,
                article_url="https://example.com/1",
                description=f"{ticker} reported EPS of $2.18 vs $2.04 expected.",
                tickers=[ticker],
            ),
            TickerNews(
                title=f"Goldman Sachs upgrades {ticker} to Buy with $220 target",
                author="Reuters",
                published_utc=now,
                article_url="https://example.com/2",
                description=f"Analyst cites strong services growth and AI monetization.",
                tickers=[ticker],
            ),
            TickerNews(
                title=f"{ticker} announces $100B share buyback program",
                author="Bloomberg",
                published_utc=now,
                article_url="https://example.com/3",
                description=f"Board approved largest buyback in company history.",
                tickers=[ticker],
            ),
            TickerNews(
                title=f"Regulatory concerns mount for {ticker} in EU market",
                author="Financial Times",
                published_utc=now,
                article_url="https://example.com/4",
                description=f"European Commission investigating potential antitrust violations.",
                tickers=[ticker],
            ),
            TickerNews(
                title=f"{ticker} expands AI infrastructure with new data centers",
                author="TechCrunch",
                published_utc=now,
                article_url="https://example.com/5",
                description=f"$10B investment in GPU clusters for enterprise AI products.",
                tickers=[ticker],
            ),
        ]

    @staticmethod
    def _stub_snapshot(ticker: str) -> TickerSnapshot:
        return TickerSnapshot(
            ticker=ticker,
            price=195.30,
            change=2.10,
            change_pct=1.09,
            volume=54_200_000,
            prev_close=193.20,
            updated=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _stub_bars(ticker: str, days: int) -> list[dict]:
        import numpy as np

        rng = np.random.default_rng(hash(ticker) % 2**32)
        base = 180.0
        returns = rng.normal(0.0005, 0.015, days)
        close = base * np.cumprod(1 + returns)
        return [
            {"date": i, "open": round(c * 0.998, 2), "high": round(c * 1.01, 2),
             "low": round(c * 0.99, 2), "close": round(c, 2), "volume": int(rng.integers(20e6, 60e6))}
            for i, c in enumerate(close)
        ]
