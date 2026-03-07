"""
AI Chat assistant endpoint with SSE streaming.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.dependencies import get_current_user

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


@router.post("/message")
async def chat_message(req: ChatRequest, user: dict = Depends(get_current_user)):
    from prediction.chat_assistant import process_chat_message

    portfolio_data = {"holdings": [], "total_value": 0, "has_portfolio": False}

    async def event_stream():
        try:
            async for chunk in process_chat_message(req.message, req.session_id, portfolio_data):
                yield chunk
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
