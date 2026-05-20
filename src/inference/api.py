"""
src/inference/api.py

FastAPI inference endpoint for the fine-tuned AeroSense ChartQA model.
Supports both local Ollama inference and direct model loading.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from src.inference.ollama_client import OllamaClient

# ── Schemas ───────────────────────────────────────────────────────────────────

class InferenceRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)
    context: str = Field(default="", max_length=2000)
    max_tokens: int = Field(default=512, ge=64, le=2048)
    model: str = Field(default="aerosense-chartqa")

class InferenceResponse(BaseModel):
    answer: str
    model: str
    latency_ms: float
    tokens_generated: int | None = None
    safety_flagged: bool = False

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0.0"


# ── Safety filter ─────────────────────────────────────────────────────────────

SAFETY_PATTERNS = [
    "how to bypass",
    "disable transponder",
    "fly without clearance",
    "ignore atc",
    "exceed limits",
]

def is_safety_flagged(question: str) -> bool:
    q = question.lower()
    return any(pattern in q for pattern in SAFETY_PATTERNS)


# ── App lifecycle ─────────────────────────────────────────────────────────────

client: OllamaClient | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    logger.info("Starting AeroSense ChartQA API...")
    client = OllamaClient(model="aerosense-chartqa")
    logger.success("Ollama client initialized ✓")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="AeroSense ChartQA API",
    description="Fine-tuned aviation chart Q&A — Qwen2.5-3B + LoRA/QLoRA",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=client is not None,
    )


@app.post("/infer", response_model=InferenceResponse)
async def infer(req: InferenceRequest) -> InferenceResponse:
    if client is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    safety_flagged = is_safety_flagged(req.question)

    start = time.perf_counter()
    try:
        answer = client.generate(
            question=req.question,
            context=req.context,
            max_tokens=req.max_tokens,
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    latency_ms = (time.perf_counter() - start) * 1000

    if safety_flagged:
        answer = (
            "⚠️ SAFETY NOTICE: This question may relate to unsafe aviation practices. "
            "Always consult official FAA publications and qualified aviation professionals. "
            "\n\n" + answer
        )

    logger.info(f"Inference: {latency_ms:.0f}ms | safety_flagged={safety_flagged}")

    return InferenceResponse(
        answer=answer,
        model=req.model,
        latency_ms=round(latency_ms, 1),
        safety_flagged=safety_flagged,
    )


@app.get("/examples", response_model=list[dict[str, Any]])
async def examples() -> list[dict[str, Any]]:
    """Return sample questions for the demo UI."""
    return [
        {"category": "chart_symbology",    "question": "What does a dashed magenta circle on a VFR sectional chart indicate?"},
        {"category": "approach_procedures", "question": "What is the difference between decision altitude and minimum descent altitude?"},
        {"category": "airspace",            "question": "What are the pilot certification requirements to fly in Class B airspace?"},
        {"category": "navigation_aids",     "question": "How does GPS RAIM work and why does it matter for IFR operations?"},
        {"category": "faa_regulations",     "question": "What are the fuel requirements for an IFR flight?"},
    ]
