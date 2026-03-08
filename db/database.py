"""
Async database engine and session factory (PostgreSQL via asyncpg).

Falls back to SQLite for local development when DATABASE_URL is not configured.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import get_settings

DATABASE_URL = get_settings().database_url

if DATABASE_URL.startswith("sqlite"):
    async_engine = create_async_engine(DATABASE_URL, echo=False)
else:
    async_engine = create_async_engine(
        DATABASE_URL,
        echo=False,
        pool_size=20,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=1800,
        connect_args={"timeout": 10},
    )

# Backward-compat alias
engine = async_engine

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        try:
            yield session
            if session.in_transaction():
                await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    from db.models import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
