"""Initial schema — portfolio, signals, predictions, users.

Revision ID: 001
Revises: None
Create Date: 2026-03-06
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("display_name", sa.String(100), nullable=False),
        sa.Column("password_hash", sa.Text, nullable=False),
        sa.Column("risk_profile", sa.String(20), server_default="moderate"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_users_email", "users", ["email"])

    op.create_table(
        "portfolio_holdings",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("ticker", sa.String(10), nullable=False),
        sa.Column("quantity", sa.Float, nullable=False),
        sa.Column("avg_price", sa.Float, nullable=False),
        sa.Column("asset_class", sa.String(20), server_default="stock"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("user_id", "ticker", name="uq_user_ticker"),
    )
    op.create_index("ix_portfolio_user_id", "portfolio_holdings", ["user_id"])

    op.create_table(
        "user_settings",
        sa.Column("key", sa.String(100), primary_key=True),
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("value", JSONB, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_user_settings_user_id", "user_settings", ["user_id"])

    op.create_table(
        "signal_history",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("ticker", sa.String(10), nullable=False),
        sa.Column("signal_type", sa.String(50), nullable=False),
        sa.Column("direction", sa.String(10)),
        sa.Column("confidence", sa.Float),
        sa.Column("actual_result", sa.Float),
        sa.Column("metadata", JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_signal_ticker_created", "signal_history", ["ticker", sa.text("created_at DESC")])

    op.create_table(
        "prediction_cache",
        sa.Column("symbol", sa.String(10), primary_key=True),
        sa.Column("horizon", sa.String(10), primary_key=True),
        sa.Column("prediction", JSONB, nullable=False),
        sa.Column("model_version", sa.String(50)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("idx_prediction_expires", "prediction_cache", ["expires_at"])


def downgrade() -> None:
    op.drop_table("prediction_cache")
    op.drop_table("signal_history")
    op.drop_table("user_settings")
    op.drop_table("portfolio_holdings")
    op.drop_table("users")
