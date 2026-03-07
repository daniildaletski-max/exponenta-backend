"""
Shared pytest fixtures for the Exponenta backend test suite.

Uses httpx.AsyncClient + ASGITransport so every test runs against
the real FastAPI app without starting a server process.
The auth layer uses the in-memory _DEMO_USERS dict, so no
database or external services are required for auth tests.
"""

import sys
import os

# Ensure the backend package root is on sys.path so that
# `from main import app` and sibling imports resolve correctly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from httpx import ASGITransport, AsyncClient

from main import app as fastapi_app


# ---------------------------------------------------------------------------
# Reset the in-memory user store between tests so registrations in one test
# don't leak into another.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_demo_users():
    """Reset _DEMO_USERS to its original state before each test."""
    from api.routes_auth import _DEMO_USERS, _pwd

    original_demo = {
        "demo@exponenta.ai": {
            "id": "usr_demo_001",
            "email": "demo@exponenta.ai",
            "display_name": "Demo Investor",
            "password_hash": _pwd.hash("exponenta2026"),
            "created_at": __import__("datetime").datetime(
                2026, 1, 15, tzinfo=__import__("datetime").timezone.utc
            ),
        }
    }
    _DEMO_USERS.clear()
    _DEMO_USERS.update(original_demo)
    yield
    # cleanup after the test
    _DEMO_USERS.clear()
    _DEMO_USERS.update(original_demo)


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    """Return the FastAPI application instance."""
    return fastapi_app


@pytest.fixture
async def client(app):
    """Async httpx client wired to the ASGI app (no real server needed)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def auth_token(client: AsyncClient) -> str:
    """Register a fresh test user and return a valid JWT access token."""
    resp = await client.post(
        "/api/v1/auth/register",
        json={
            "name": "Test User",
            "email": "testuser@example.com",
            "password": "securepassword123",
        },
    )
    assert resp.status_code == 201, f"Registration failed: {resp.text}"
    data = resp.json()
    return data["access_token"]


@pytest.fixture
async def auth_headers(auth_token: str) -> dict[str, str]:
    """Return an Authorization header dict with a valid Bearer token."""
    return {"Authorization": f"Bearer {auth_token}"}
