from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_current_user
from db.database import get_db
from db.repository import alerts_repo

router = APIRouter()


class AlertCreate(BaseModel):
    ticker: str = Field(min_length=1, max_length=16)
    condition: str = Field(pattern="^(above|below|pct_change)$")
    target_value: float


class AlertOut(BaseModel):
    id: str
    ticker: str
    condition: str
    target_value: float
    triggered: bool
    triggered_at: datetime | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class AlertListResponse(BaseModel):
    alerts: list[AlertOut]
    count: int


@router.get("", response_model=AlertListResponse)
async def list_alerts(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all price alerts for the current user."""
    rows = await alerts_repo.list_by_user(db, user["id"])
    return AlertListResponse(
        alerts=[AlertOut.model_validate(a) for a in rows],
        count=len(rows),
    )


@router.post("", response_model=AlertOut, status_code=status.HTTP_201_CREATED)
async def create_alert(
    body: AlertCreate,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new price alert."""
    user_id = user["id"]

    # Max 20 alerts per user
    count = await alerts_repo.count_by_user(db, user_id)
    if count >= 20:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Maximum 20 alerts allowed. Delete some before creating new ones.",
        )

    alert = await alerts_repo.create(
        db,
        user_id=user_id,
        ticker=body.ticker,
        condition=body.condition,
        target_value=body.target_value,
    )
    return AlertOut.model_validate(alert)


@router.delete("/{alert_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_alert(
    alert_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a price alert."""
    deleted = await alerts_repo.delete(db, user["id"], alert_id)
    if not deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Alert not found")
