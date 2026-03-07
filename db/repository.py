"""
Database repository -- async CRUD operations replacing in-memory stores.

Usage in routes:
    from db.database import get_db
    from db.repository import holdings_repo

    @router.get("")
    async def list_holdings(user=Depends(get_current_user), db=Depends(get_db)):
        return await holdings_repo.list_by_user(db, user["sub"])
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import (
    ChatMessage,
    ModelVersion,
    PortfolioHolding,
    PortfolioSnapshot,
    PriceAlert,
    RefreshToken,
    SignalHistory,
    User,
)


# ── Holdings ─────────────────────────────────────────────────────────


class HoldingsRepository:
    async def list_by_user(self, db: AsyncSession, user_id: str) -> list[PortfolioHolding]:
        result = await db.execute(
            select(PortfolioHolding)
            .where(PortfolioHolding.user_id == user_id)
            .order_by(PortfolioHolding.created_at)
        )
        return list(result.scalars().all())

    async def get(self, db: AsyncSession, user_id: str, holding_id: int) -> PortfolioHolding | None:
        result = await db.execute(
            select(PortfolioHolding).where(
                PortfolioHolding.id == holding_id,
                PortfolioHolding.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_by_ticker(self, db: AsyncSession, user_id: str, ticker: str) -> PortfolioHolding | None:
        result = await db.execute(
            select(PortfolioHolding).where(
                PortfolioHolding.user_id == user_id,
                PortfolioHolding.ticker == ticker.upper(),
            )
        )
        return result.scalar_one_or_none()

    async def create(
        self, db: AsyncSession, user_id: str, ticker: str,
        quantity: float, avg_price: float, asset_class: str = "stock",
    ) -> PortfolioHolding:
        holding = PortfolioHolding(
            user_id=user_id,
            ticker=ticker.upper(),
            quantity=quantity,
            avg_price=avg_price,
            asset_class=asset_class,
        )
        db.add(holding)
        await db.flush()
        await db.refresh(holding)
        return holding

    async def update(
        self, db: AsyncSession, user_id: str, holding_id: int, **kwargs
    ) -> PortfolioHolding | None:
        holding = await self.get(db, user_id, holding_id)
        if not holding:
            return None
        for key, value in kwargs.items():
            if value is not None and hasattr(holding, key):
                setattr(holding, key, value)
        await db.flush()
        await db.refresh(holding)
        return holding

    async def delete(self, db: AsyncSession, user_id: str, holding_id: int) -> bool:
        result = await db.execute(
            delete(PortfolioHolding).where(
                PortfolioHolding.id == holding_id,
                PortfolioHolding.user_id == user_id,
            )
        )
        return result.rowcount > 0


# ── Portfolio Snapshots ──────────────────────────────────────────────


class SnapshotsRepository:
    async def create(
        self, db: AsyncSession, user_id: str,
        total_value: float, holdings_json: list[dict], metrics_json: dict | None = None,
    ) -> PortfolioSnapshot:
        snap = PortfolioSnapshot(
            id=f"snap_{uuid4().hex[:12]}",
            user_id=user_id,
            total_value=total_value,
            holdings_json=holdings_json,
            metrics_json=metrics_json or {},
        )
        db.add(snap)
        await db.flush()
        await db.refresh(snap)
        return snap

    async def list_by_user(
        self, db: AsyncSession, user_id: str, limit: int = 30, offset: int = 0,
    ) -> tuple[list[PortfolioSnapshot], int]:
        # Count
        count_q = select(PortfolioSnapshot).where(PortfolioSnapshot.user_id == user_id)
        total_result = await db.execute(count_q)
        total = len(total_result.scalars().all())

        # Paginated
        result = await db.execute(
            select(PortfolioSnapshot)
            .where(PortfolioSnapshot.user_id == user_id)
            .order_by(PortfolioSnapshot.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all()), total

    async def get_performance(
        self, db: AsyncSession, user_id: str, days: int = 30,
    ) -> list[PortfolioSnapshot]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        result = await db.execute(
            select(PortfolioSnapshot)
            .where(
                PortfolioSnapshot.user_id == user_id,
                PortfolioSnapshot.created_at >= cutoff,
            )
            .order_by(PortfolioSnapshot.created_at)
        )
        return list(result.scalars().all())

    async def delete(self, db: AsyncSession, user_id: str, snapshot_id: str) -> bool:
        result = await db.execute(
            delete(PortfolioSnapshot).where(
                PortfolioSnapshot.id == snapshot_id,
                PortfolioSnapshot.user_id == user_id,
            )
        )
        return result.rowcount > 0


# ── Chat Messages ────────────────────────────────────────────────────


class ChatRepository:
    async def save_message(
        self, db: AsyncSession, user_id: str,
        session_id: str, role: str, content: str,
    ) -> ChatMessage:
        msg = ChatMessage(
            id=f"msg_{uuid4().hex[:12]}",
            user_id=user_id,
            session_id=session_id,
            role=role,
            content=content,
        )
        db.add(msg)
        await db.flush()
        await db.refresh(msg)
        return msg

    async def list_sessions(self, db: AsyncSession, user_id: str) -> list[dict]:
        """Get unique sessions with last message and count."""
        result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.user_id == user_id)
            .order_by(ChatMessage.created_at.desc())
        )
        messages = result.scalars().all()

        sessions: dict[str, dict] = {}
        for msg in messages:
            if msg.session_id not in sessions:
                preview = msg.content[:80] + ("..." if len(msg.content) > 80 else "")
                sessions[msg.session_id] = {
                    "session_id": msg.session_id,
                    "message_count": 0,
                    "last_message_at": msg.created_at,
                    "preview": preview,
                }
            sessions[msg.session_id]["message_count"] += 1
        return list(sessions.values())

    async def get_session_messages(
        self, db: AsyncSession, user_id: str, session_id: str,
        limit: int = 50, offset: int = 0,
    ) -> tuple[list[ChatMessage], int]:
        base = select(ChatMessage).where(
            ChatMessage.user_id == user_id,
            ChatMessage.session_id == session_id,
        )
        count_result = await db.execute(base)
        total = len(count_result.scalars().all())

        result = await db.execute(
            base.order_by(ChatMessage.created_at)
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all()), total

    async def delete_session(self, db: AsyncSession, user_id: str, session_id: str) -> int:
        result = await db.execute(
            delete(ChatMessage).where(
                ChatMessage.user_id == user_id,
                ChatMessage.session_id == session_id,
            )
        )
        return result.rowcount


# ── Price Alerts ─────────────────────────────────────────────────────


class AlertsRepository:
    async def list_by_user(self, db: AsyncSession, user_id: str) -> list[PriceAlert]:
        result = await db.execute(
            select(PriceAlert)
            .where(PriceAlert.user_id == user_id)
            .order_by(PriceAlert.created_at.desc())
        )
        return list(result.scalars().all())

    async def create(
        self, db: AsyncSession, user_id: str,
        ticker: str, condition: str, target_value: float,
    ) -> PriceAlert:
        alert = PriceAlert(
            id=f"alt_{uuid4().hex[:12]}",
            user_id=user_id,
            ticker=ticker.upper(),
            condition=condition,
            target_value=target_value,
        )
        db.add(alert)
        await db.flush()
        await db.refresh(alert)
        return alert

    async def count_by_user(self, db: AsyncSession, user_id: str) -> int:
        result = await db.execute(
            select(PriceAlert).where(PriceAlert.user_id == user_id)
        )
        return len(result.scalars().all())

    async def delete(self, db: AsyncSession, user_id: str, alert_id: str) -> bool:
        result = await db.execute(
            delete(PriceAlert).where(
                PriceAlert.id == alert_id,
                PriceAlert.user_id == user_id,
            )
        )
        return result.rowcount > 0

    async def trigger(self, db: AsyncSession, alert_id: str) -> bool:
        result = await db.execute(
            update(PriceAlert)
            .where(PriceAlert.id == alert_id)
            .values(triggered=True, triggered_at=datetime.now(timezone.utc))
        )
        return result.rowcount > 0


# ── Refresh Tokens ───────────────────────────────────────────────────


class RefreshTokenRepository:
    async def create(
        self, db: AsyncSession, user_id: str, token: str, expires_days: int = 7,
    ) -> str:
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        rt = RefreshToken(
            id=f"rt_{uuid4().hex[:12]}",
            user_id=user_id,
            token_hash=token_hash,
            expires_at=datetime.now(timezone.utc) + timedelta(days=expires_days),
        )
        db.add(rt)
        await db.flush()
        return token_hash

    async def validate(self, db: AsyncSession, token: str) -> RefreshToken | None:
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        result = await db.execute(
            select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        )
        return result.scalar_one_or_none()

    async def revoke(self, db: AsyncSession, token_hash: str) -> None:
        await db.execute(
            update(RefreshToken)
            .where(RefreshToken.token_hash == token_hash)
            .values(revoked=True)
        )

    async def revoke_all_for_user(self, db: AsyncSession, user_id: str) -> int:
        result = await db.execute(
            update(RefreshToken)
            .where(RefreshToken.user_id == user_id)
            .values(revoked=True)
        )
        return result.rowcount


# ── Signal History ───────────────────────────────────────────────────


class SignalRepository:
    async def record(
        self, db: AsyncSession, ticker: str, signal_type: str,
        direction: str, confidence: float, metadata: dict | None = None,
    ) -> SignalHistory:
        signal = SignalHistory(
            ticker=ticker.upper(),
            signal_type=signal_type,
            direction=direction,
            confidence=confidence,
            metadata_=metadata or {},
        )
        db.add(signal)
        await db.flush()
        await db.refresh(signal)
        return signal

    async def list_recent(self, db: AsyncSession, limit: int = 100) -> list[SignalHistory]:
        result = await db.execute(
            select(SignalHistory)
            .order_by(SignalHistory.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


# ── Singletons ───────────────────────────────────────────────────────

holdings_repo = HoldingsRepository()
snapshots_repo = SnapshotsRepository()
chat_repo = ChatRepository()
alerts_repo = AlertsRepository()
refresh_token_repo = RefreshTokenRepository()
signal_repo = SignalRepository()
