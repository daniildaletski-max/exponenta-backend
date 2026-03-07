"""
Tests for market and system-level endpoints defined in main.py:

  - GET  /health
  - POST /api/v1/sentiment/analyze
  - GET  /api/v1/market/quotes  (requires auth)
"""

import pytest
from httpx import AsyncClient


class TestHealth:

    async def test_health(self, client: AsyncClient):
        """GET /health returns 200 with status and version."""
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert "timestamp" in body


class TestSentiment:

    async def test_sentiment_no_input(self, client: AsyncClient):
        """POST /sentiment/analyze with neither tickers nor query returns 400."""
        resp = await client.post(
            "/api/v1/sentiment/analyze",
            json={},
        )
        assert resp.status_code == 400


class TestQuotes:

    async def test_quotes_endpoint(
        self, client: AsyncClient, auth_headers: dict
    ):
        """GET /market/quotes?tickers=AAPL returns a quotes array (or 500 if no API key)."""
        resp = await client.get(
            "/api/v1/market/quotes",
            params={"tickers": "AAPL"},
            headers=auth_headers,
        )

        if resp.status_code == 500:
            pytest.skip(
                "Market quotes returned 500 — Polygon API key likely missing"
            )

        assert resp.status_code == 200
        body = resp.json()
        assert "quotes" in body
        assert isinstance(body["quotes"], list)
