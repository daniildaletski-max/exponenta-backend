"""
Tests for authentication endpoints: /api/v1/auth/*

Covers registration, login, token validation, and the /me endpoint.
All tests use the in-memory _DEMO_USERS store (no database required).
"""

import pytest
from httpx import AsyncClient


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegister:

    async def test_register_success(self, client: AsyncClient):
        """POST /register with valid data returns 201 and an access_token."""
        resp = await client.post(
            "/api/v1/auth/register",
            json={
                "name": "Alice Doe",
                "email": "alice@example.com",
                "password": "strongpass99",
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"
        assert body["expires_in"] > 0

    async def test_register_duplicate(self, client: AsyncClient):
        """Registering the same email twice returns 409 Conflict."""
        payload = {
            "name": "Bob",
            "email": "bob@example.com",
            "password": "password1234",
        }
        resp1 = await client.post("/api/v1/auth/register", json=payload)
        assert resp1.status_code == 201

        resp2 = await client.post("/api/v1/auth/register", json=payload)
        assert resp2.status_code == 409

    async def test_register_invalid_email(self, client: AsyncClient):
        """An obviously invalid email returns 422."""
        resp = await client.post(
            "/api/v1/auth/register",
            json={
                "name": "Bad Email",
                "email": "not-an-email",
                "password": "password1234",
            },
        )
        assert resp.status_code == 422

    async def test_register_short_password(self, client: AsyncClient):
        """A password shorter than 8 characters returns 422."""
        resp = await client.post(
            "/api/v1/auth/register",
            json={
                "name": "Short Pass",
                "email": "short@example.com",
                "password": "abc",
            },
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

class TestLogin:

    async def test_login_success(self, client: AsyncClient):
        """Login with a registered user returns 200 and a token."""
        # Register first
        await client.post(
            "/api/v1/auth/register",
            json={
                "name": "Login Tester",
                "email": "logintest@example.com",
                "password": "mypassword88",
            },
        )

        resp = await client.post(
            "/api/v1/auth/login",
            json={
                "email": "logintest@example.com",
                "password": "mypassword88",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"

    async def test_login_invalid_password(self, client: AsyncClient):
        """Wrong password returns 401."""
        resp = await client.post(
            "/api/v1/auth/login",
            json={
                "email": "demo@exponenta.ai",
                "password": "wrongpassword",
            },
        )
        assert resp.status_code == 401

    async def test_login_nonexistent_user(self, client: AsyncClient):
        """Login with an unknown email returns 401."""
        resp = await client.post(
            "/api/v1/auth/login",
            json={
                "email": "ghost@nowhere.com",
                "password": "irrelevant123",
            },
        )
        assert resp.status_code == 401

    async def test_demo_user_login(self, client: AsyncClient):
        """The pre-seeded demo user can log in with known credentials."""
        resp = await client.post(
            "/api/v1/auth/login",
            json={
                "email": "demo@exponenta.ai",
                "password": "exponenta2026",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body


# ---------------------------------------------------------------------------
# /me endpoint
# ---------------------------------------------------------------------------

class TestMe:

    async def test_me_authenticated(self, client: AsyncClient, auth_headers: dict):
        """GET /me with a valid token returns user info."""
        resp = await client.get("/api/v1/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "email" in body
        assert body["email"] == "testuser@example.com"
        assert "display_name" in body
        assert "id" in body

    async def test_me_no_token(self, client: AsyncClient):
        """GET /me without an Authorization header returns 401 or 403."""
        resp = await client.get("/api/v1/auth/me")
        assert resp.status_code in (401, 403)
