"""
API v1 Router Endpoints: Comprehensive Multimodal, Streaming & Agentic Suite
"""

import json
import asyncio
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status
from fastapi.responses import StreamingResponse
from typing import Optional
from app.models.schemas import ChatRequest, ChatResponse, DocumentSummaryResponse, HealthResponse
from app.services.agent_service import agent_router
from app.services.rag_service import rag_service
from app.services.vision_service import vision_service
from app.services.speech_service import speech_service
from app.services.document_service import summarize_text_document
from app.core.config import settings
from app.core.logging import logger
from app.core.errors import CustomGPTException

router = APIRouter()


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def health_check():
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        environment=settings.ENV,
        model_loaded=True,
        device=settings.DEVICE,
    )


@router.post("/upload", response_model=DocumentSummaryResponse, status_code=status.HTTP_200_OK)
async def upload_and_summarize(file: UploadFile = File(...)):
    try:
        content_bytes = await file.read()
        rag_service.ingest_document(file.filename, content_bytes)
        summary_report = summarize_text_document(file.filename, content_bytes)
        return DocumentSummaryResponse(filename=file.filename, summary=summary_report)
    except CustomGPTException as e:
        logger.error(f"Custom Exception processing upload for {file.filename}: {e.message}")
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Error processing upload for {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=f"File processing error: {str(e)}")


@router.post("/vision/analyze", status_code=status.HTTP_200_OK)
async def analyze_image_endpoint(file: UploadFile = File(...), prompt: Optional[str] = Form(None)):
    try:
        image_bytes = await file.read()
        analysis = vision_service.analyze_image(file.filename, image_bytes, prompt)
        return {"filename": file.filename, "analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")


@router.post("/speech/transcribe", status_code=status.HTTP_200_OK)
async def transcribe_audio_endpoint(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        result = speech_service.transcribe_audio(file.filename, audio_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio processing error: {str(e)}")


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
def generate_chat_response(request: ChatRequest):
    try:
        response_text, tool_used, trace, citations = agent_router.route_and_execute(
            prompt=request.prompt,
            engine_mode=request.engine_mode or "flash_reasoning",
            session_id=request.session_id or "default_session"
        )
        return ChatResponse(
            prompt=request.prompt,
            response=response_text,
            full_text=response_text,
            source_tool=tool_used,
            engine_mode=request.engine_mode or "flash_reasoning",
            session_id=request.session_id or "default_session",
            reasoning_trace=trace,
            citations=citations
        )
    except Exception as e:
        logger.error(f"Agent Execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream", status_code=status.HTTP_200_OK)
async def generate_chat_stream(request: ChatRequest):
    """
    Phase D SSE Streaming Endpoint:
    1. Executes tools synchronously and emits discrete reasoning trace + citations event.
    2. Streams final response tokens live over Server-Sent Events.
    """
    async def event_generator():
        try:
            # Step 1: Execute agent tool routing
            response_text, tool_used, trace, citations = agent_router.route_and_execute(
                prompt=request.prompt,
                engine_mode=request.engine_mode or "flash_reasoning",
                session_id=request.session_id or "default_session"
            )

            # Step 2: Emit discrete reasoning trace and citations SSE event
            trace_payload = json.dumps({
                "trace": trace,
                "tool": tool_used,
                "citations": [c.model_dump() for c in citations]
            })
            yield f"event: trace\ndata: {trace_payload}\n\n"

            # Step 3: Stream final answer tokens
            words = response_text.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                token_payload = json.dumps({"token": chunk})
                yield f"event: token\ndata: {token_payload}\n\n"
                await asyncio.sleep(0.03)

            # Step 4: Emit end event
            yield f"event: end\ndata: {json.dumps({'status': 'complete'})}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            err_payload = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {err_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
