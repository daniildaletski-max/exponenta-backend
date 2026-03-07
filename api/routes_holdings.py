from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_current_user
from db.database import get_db
from db.repository import holdings_repo

router = APIRouter()


class HoldingCreate(BaseModel):
    ticker: str = Field(min_length=1, max_length=16)
    quantity: float = Field(gt=0)
    avg_price: float = Field(ge=0)
    asset_class: str = "stock"


class HoldingUpdate(BaseModel):
    quantity: float | None = Field(default=None, gt=0)
    avg_price: float | None = Field(default=None, ge=0)
    asset_class: str | None = None


class HoldingOut(BaseModel):
    id: int
    ticker: str
    quantity: float
    avg_price: float
    asset_class: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class HoldingsListResponse(BaseModel):
    holdings: list[HoldingOut]
    count: int


@router.get("", response_model=HoldingsListResponse)
async def list_holdings(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all holdings for the current user."""
    rows = await holdings_repo.list_by_user(db, user["id"])
    return HoldingsListResponse(
        holdings=[HoldingOut.model_validate(h) for h in rows],
        count=len(rows),
    )


@router.post("", response_model=HoldingOut, status_code=status.HTTP_201_CREATED)
async def create_holding(
    body: HoldingCreate,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Add a new holding to the user's portfolio."""
    user_id = user["id"]

    # Check for duplicate ticker
    existing = await holdings_repo.get_by_ticker(db, user_id, body.ticker)
    if existing:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Holding for {body.ticker.upper()} already exists. Use PATCH to update.",
        )

    holding = await holdings_repo.create(
        db,
        user_id=user_id,
        ticker=body.ticker,
        quantity=body.quantity,
        avg_price=body.avg_price,
        asset_class=body.asset_class,
    )
    return HoldingOut.model_validate(holding)


@router.patch("/{holding_id}", response_model=HoldingOut)
async def update_holding(
    holding_id: int,
    body: HoldingUpdate,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update an existing holding."""
    holding = await holdings_repo.update(
        db,
        user_id=user["id"],
        holding_id=holding_id,
        quantity=body.quantity,
        avg_price=body.avg_price,
        asset_class=body.asset_class,
    )
    if not holding:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Holding not found")
    return HoldingOut.model_validate(holding)


@router.delete("/{holding_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_holding(
    holding_id: int,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Remove a holding from the user's portfolio."""
    deleted = await holdings_repo.delete(db, user["id"], holding_id)
    if not deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Holding not found")
