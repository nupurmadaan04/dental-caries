"""
FastAPI Backend Server for Dental Caries Clinical AI & Gemini Assistant
Provides secure server-side Gemini API mediation, health checks, and case-aware chat endpoints.
"""

import os
import json
import logging
import asyncio
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load server environment variables
load_dotenv()

from backend.gemini_service import gemini_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("dental_ai.server")

app = FastAPI(
    title="Dental Caries Clinical AI Backend",
    version="2.4.0",
    description="Secure backend server for Dental Caries Segmentation & Gemini Clinical AI Assistant"
)

# Enable CORS for frontend development and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request / Response Schemas
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User prompt or question")
    session_id: Optional[str] = Field(default="default_session", description="Unique session identifier for multi-turn history")
    mode: Optional[str] = Field(default="standard", description="'standard', 'simple', or 'technical'")
    case_context: Optional[Dict[str, Any]] = Field(default=None, description="Current MLUA case results and model metadata (PHI-free)")

class ClearSessionRequest(BaseModel):
    session_id: str

@app.get("/api/health")
@app.get("/api/chat/health")
async def health_check():
    """Health check endpoint confirming server status and Gemini readiness."""
    has_api_key = bool(os.getenv("GEMINI_API_KEY", "").strip() and os.getenv("GEMINI_API_KEY") != "your_gemini_api_key_here")
    return {
        "status": "ONLINE",
        "service": "Dental Caries Clinical AI Service",
        "version": "2.4.0",
        "model_checkpoint": "EXP-MLUA-003_E75_BEST.pth",
        "selected_epoch": 75,
        "completed_training_epochs": 78,
        "production_threshold": 0.50,
        "gemini_assistant": {
            "configured": has_api_key,
            "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            "api_approach": "google-genai Interactions & Models API",
            "status": "READY" if has_api_key else "STANDBY (Set GEMINI_API_KEY in server .env)"
        }
    }

@app.post("/api/chat")
@app.post("/api/assistant/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Context-aware conversational assistant endpoint.
    Securely routes user questions to Gemini with dynamic MLUA case context.
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        response = await gemini_service.chat(
            message=request.message.strip(),
            session_id=request.session_id or "default_session",
            case_context=request.case_context,
            mode=request.mode or "standard"
        )
        return response
    except Exception as e:
        logger.error("Chat endpoint error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error while processing assistant query.")

@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Streaming chat endpoint for progressive token delivery.
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    async def event_generator():
        try:
            response = await gemini_service.chat(
                message=request.message.strip(),
                session_id=request.session_id or "default_session",
                case_context=request.case_context,
                mode=request.mode or "standard"
            )
            full_text = response.get("text", "")
            # Stream words with slight delay for smooth interactive typing
            words = full_text.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield f"data: {json.dumps({'chunk': chunk, 'done': False}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.015)
            yield f"data: {json.dumps({'done': True, 'full_text': full_text}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error("Stream error: %s", e)
            yield f"data: {json.dumps({'error': 'Streaming interrupted', 'done': True})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/chat/clear")
async def clear_session_endpoint(request: ClearSessionRequest):
    """Clears history and context for the specified chat session."""
    gemini_service.clear_session(request.session_id)
    return {"status": "cleared", "session_id": request.session_id}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info("Starting Dental Caries AI server on %s:%d", host, port)
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)
