"""
Tests for system-level endpoints:

  - GET /api/v1/system/status
  - GET /docs  (OpenAPI / Swagger UI)
"""

import pytest
from httpx import AsyncClient


class TestSystemStatus:

    async def test_system_status(self, client: AsyncClient):
        """GET /system/status returns 200."""
        resp = await client.get("/api/v1/system/status")
        assert resp.status_code == 200
        body = resp.json()
        # The response should at minimum contain the app version
        assert "version" in body


class TestOpenAPIDocs:

    async def test_openapi_docs(self, client: AsyncClient):
        """GET /docs (Swagger UI) returns 200."""
        resp = await client.get("/docs")
        assert resp.status_code == 200
