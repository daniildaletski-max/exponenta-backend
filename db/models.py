"""
SQLAlchemy ORM models for Exponenta.

Replaces the JSON-file storage with proper PostgreSQL tables.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSON, JSONB
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class PortfolioHolding(Base):
    __tablename__ = "portfolio_holdings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), nullable=False, index=True)
    ticker = Column(String(10), nullable=False)
    quantity = Column(Float, nullable=False)
    avg_price = Column(Float, nullable=False)
    asset_class = Column(String(20), default="stock")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("user_id", "ticker", name="uq_user_ticker"),
    )


class UserSetting(Base):
    __tablename__ = "user_settings"

    user_id = Column(String(64), primary_key=True)
    key = Column(String(100), primary_key=True)
    value = Column(JSONB, nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class SignalHistory(Base):
    __tablename__ = "signal_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False)
    signal_type = Column(String(50), nullable=False)
    direction = Column(String(10))
    confidence = Column(Float)
    actual_result = Column(Float)
    metadata_ = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_signal_ticker_created", "ticker", "created_at"),
    )


class PredictionCache(Base):
    __tablename__ = "prediction_cache"

    symbol = Column(String(10), primary_key=True)
    horizon = Column(String(10), primary_key=True)
    prediction = Column(JSONB, nullable=False)
    model_version = Column(String(50))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_prediction_expires", "expires_at"),
    )


class User(Base):
    __tablename__ = "users"

    id = Column(String(64), primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    display_name = Column(String(100), nullable=False)
    password_hash = Column(Text, nullable=False)
    risk_profile = Column(String(20), default="moderate")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(String(64), primary_key=True)
    user_id = Column(String(64), ForeignKey("users.id"), nullable=False, index=True)
    token_hash = Column(String(128), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    revoked = Column(Boolean, default=False)


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id = Column(String(64), primary_key=True, default=lambda: f"snap_{uuid4().hex[:12]}")
    user_id = Column(String(64), ForeignKey("users.id"), nullable=False, index=True)
    total_value = Column(Float, nullable=False)
    holdings_json = Column(JSON, nullable=False)
    metrics_json = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String(64), primary_key=True, default=lambda: f"msg_{uuid4().hex[:12]}")
    user_id = Column(String(64), ForeignKey("users.id"), nullable=False, index=True)
    session_id = Column(String(64), nullable=False, index=True)
    role = Column(String(16), nullable=False)  # "user" | "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(String(64), primary_key=True, default=lambda: f"mdl_{uuid4().hex[:12]}")
    model_name = Column(String(64), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    metrics_json = Column(JSON, default=dict)
    training_date = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=True)
    file_path = Column(String(256), nullable=True)


class PriceAlert(Base):
    __tablename__ = "price_alerts"

    id = Column(String(64), primary_key=True, default=lambda: f"alt_{uuid4().hex[:12]}")
    user_id = Column(String(64), ForeignKey("users.id"), nullable=False, index=True)
    ticker = Column(String(16), nullable=False)
    condition = Column(String(16), nullable=False)  # "above" | "below" | "pct_change"
    target_value = Column(Float, nullable=False)
    triggered = Column(Boolean, default=False)
    triggered_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
