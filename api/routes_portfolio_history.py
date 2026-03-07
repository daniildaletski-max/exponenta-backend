from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_current_user
from db.database import get_db
from db.repository import snapshots_repo

router = APIRouter()


class SnapshotCreate(BaseModel):
    total_value: float
    holdings: list[dict]
    metrics: dict = Field(default_factory=dict)


class SnapshotOut(BaseModel):
    id: str
    total_value: float
    holdings: list[dict]
    metrics: dict
    created_at: datetime


class SnapshotListResponse(BaseModel):
    snapshots: list[SnapshotOut]
    count: int


class PerformanceSummary(BaseModel):
    current_value: float
    initial_value: float
    total_return_pct: float
    period_days: int
    data_points: list[dict]


@router.post("", response_model=SnapshotOut, status_code=201)
async def create_snapshot(
    body: SnapshotCreate,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Save a portfolio snapshot for performance tracking."""
    user_id = user["id"]
    snap = await snapshots_repo.create(
        db,
        user_id=user_id,
        total_value=body.total_value,
        holdings_json=body.holdings,
        metrics_json=body.metrics,
    )
    return SnapshotOut(
        id=snap.id,
        total_value=snap.total_value,
        holdings=snap.holdings_json,
        metrics=snap.metrics_json or {},
        created_at=snap.created_at,
    )


@router.get("", response_model=SnapshotListResponse)
async def list_snapshots(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = Query(default=30, ge=1, le=365),
    offset: int = Query(default=0, ge=0),
):
    """List portfolio snapshots with pagination."""
    user_id = user["id"]
    snaps, total = await snapshots_repo.list_by_user(db, user_id, limit=limit, offset=offset)
    return SnapshotListResponse(
        snapshots=[
            SnapshotOut(
                id=s.id,
                total_value=s.total_value,
                holdings=s.holdings_json,
                metrics=s.metrics_json or {},
                created_at=s.created_at,
            )
            for s in snaps
        ],
        count=total,
    )


@router.get("/performance", response_model=PerformanceSummary)
async def performance_summary(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    days: int = Query(default=30, ge=1, le=365),
):
    """Get portfolio performance over a period."""
    user_id = user["id"]
    period_snaps = await snapshots_repo.get_performance(db, user_id, days=days)

    if not period_snaps:
        raise HTTPException(404, "No snapshots found. Create snapshots first.")

    initial = period_snaps[0].total_value
    current = period_snaps[-1].total_value
    total_return = ((current - initial) / initial * 100) if initial > 0 else 0

    data_points = [
        {"date": s.created_at.isoformat(), "value": s.total_value}
        for s in period_snaps
    ]

    return PerformanceSummary(
        current_value=current,
        initial_value=initial,
        total_return_pct=round(total_return, 4),
        period_days=days,
        data_points=data_points,
    )


@router.delete("/{snapshot_id}", status_code=204)
async def delete_snapshot(
    snapshot_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a specific snapshot."""
    user_id = user["id"]
    deleted = await snapshots_repo.delete(db, user_id, snapshot_id)
    if not deleted:
        raise HTTPException(404, "Snapshot not found")
