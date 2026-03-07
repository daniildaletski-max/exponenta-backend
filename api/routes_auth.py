from __future__ import annotations

import hashlib
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel, Field

from api.dependencies import get_current_user
from api.schemas import LoginRequest, TokenResponse, UserResponse
from config import Settings, get_settings

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

router = APIRouter()
_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

# In-memory user store for beta — replaced by DB in production
_DEMO_USERS: dict[str, dict] = {
    "demo@exponenta.ai": {
        "id": "usr_demo_001",
        "email": "demo@exponenta.ai",
        "display_name": "Demo Investor",
        "password_hash": _pwd.hash("exponenta2026"),
        "created_at": datetime(2026, 1, 15, tzinfo=timezone.utc),
    }
}

# In-memory refresh token store for beta — replaced by DB table in production
_REFRESH_TOKENS: dict[str, dict] = {}


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(min_length=5, max_length=255)
    password: str = Field(min_length=8, max_length=128)


class RefreshRequest(BaseModel):
    refresh_token: str


def _create_access_token(user: dict, settings: Settings) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    return jwt.encode(
        {"sub": user["id"], "email": user["email"], "exp": expire, "type": "access"},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )


def _create_refresh_token(user: dict, settings: Settings) -> str:
    token = secrets.token_urlsafe(48)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    expires_at = datetime.now(timezone.utc) + timedelta(days=settings.jwt_refresh_expire_days)

    _REFRESH_TOKENS[token_hash] = {
        "user_id": user["id"],
        "email": user["email"],
        "expires_at": expires_at,
        "revoked": False,
    }
    return token


def _issue_tokens(user: dict, settings: Settings) -> TokenResponse:
    access = _create_access_token(user, settings)
    refresh = _create_refresh_token(user, settings)
    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        expires_in=settings.jwt_expire_minutes * 60,
    )


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, settings: Settings = Depends(get_settings)):
    if not _EMAIL_RE.match(body.email):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Invalid email address")

    if body.email.lower() in _DEMO_USERS:
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")

    user = {
        "id": f"usr_{uuid.uuid4().hex[:12]}",
        "email": body.email.lower(),
        "display_name": body.name,
        "password_hash": _pwd.hash(body.password),
        "created_at": datetime.now(timezone.utc),
    }
    _DEMO_USERS[user["email"]] = user

    return _issue_tokens(user, settings)


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, settings: Settings = Depends(get_settings)):
    user = _DEMO_USERS.get(body.email.lower())
    if not user or not _pwd.verify(body.password, user["password_hash"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")

    return _issue_tokens(user, settings)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(body: RefreshRequest, settings: Settings = Depends(get_settings)):
    token_hash = hashlib.sha256(body.refresh_token.encode()).hexdigest()
    stored = _REFRESH_TOKENS.get(token_hash)

    if not stored:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid refresh token")

    if stored["revoked"]:
        # Possible token reuse attack — revoke all tokens for this user
        for th, data in _REFRESH_TOKENS.items():
            if data["user_id"] == stored["user_id"]:
                data["revoked"] = True
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Refresh token revoked")

    if stored["expires_at"] < datetime.now(timezone.utc):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Refresh token expired")

    # Revoke old refresh token (rotation)
    stored["revoked"] = True

    user = _DEMO_USERS.get(stored["email"])
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found")

    return _issue_tokens(user, settings)


@router.get("/me", response_model=UserResponse)
async def me(current_user: dict = Depends(get_current_user)):
    user = _DEMO_USERS.get(current_user["email"])
    if not user:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    return UserResponse(
        id=user["id"],
        email=user["email"],
        display_name=user["display_name"],
        created_at=user["created_at"],
    )
