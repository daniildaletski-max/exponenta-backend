from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_current_user
from db.database import get_db
from db.repository import chat_repo

router = APIRouter()


class MessageCreate(BaseModel):
    session_id: str = "default"
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1)


class MessageOut(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    created_at: datetime


class SessionOut(BaseModel):
    session_id: str
    message_count: int
    last_message_at: datetime
    preview: str


class SessionListResponse(BaseModel):
    sessions: list[SessionOut]
    count: int


class MessageListResponse(BaseModel):
    messages: list[MessageOut]
    session_id: str
    count: int


@router.post("/messages", response_model=MessageOut, status_code=201)
async def save_message(
    body: MessageCreate,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Save a chat message to history."""
    user_id = user["id"]
    msg = await chat_repo.save_message(
        db,
        user_id=user_id,
        session_id=body.session_id,
        role=body.role,
        content=body.content,
    )
    return MessageOut(
        id=msg.id,
        session_id=msg.session_id,
        role=msg.role,
        content=msg.content,
        created_at=msg.created_at,
    )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all chat sessions for the current user."""
    user_id = user["id"]
    sessions = await chat_repo.list_sessions(db, user_id)
    return SessionListResponse(
        sessions=[SessionOut(**s) for s in sessions],
        count=len(sessions),
    )


@router.get("/sessions/{session_id}", response_model=MessageListResponse)
async def get_session_messages(
    session_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """Get messages for a specific chat session."""
    user_id = user["id"]
    messages, total = await chat_repo.get_session_messages(
        db, user_id, session_id, limit=limit, offset=offset,
    )

    if total == 0:
        raise HTTPException(404, "Session not found")

    return MessageListResponse(
        messages=[
            MessageOut(
                id=m.id,
                session_id=m.session_id,
                role=m.role,
                content=m.content,
                created_at=m.created_at,
            )
            for m in messages
        ],
        session_id=session_id,
        count=total,
    )


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a chat session and all its messages."""
    user_id = user["id"]
    deleted_count = await chat_repo.delete_session(db, user_id, session_id)
    if deleted_count == 0:
        raise HTTPException(404, "Session not found")
