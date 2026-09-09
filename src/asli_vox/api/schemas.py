"""Pydantic schemas for API request and response payloads."""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class DetectionResponse(BaseModel):
    """Detection pipeline output schema."""
    verdict: str = Field(..., description="'bonafide' (genuine) or 'spoof' (deepfake)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    heatmap_base64: str = Field(..., description="Base64 PNG data URI of Grad-CAM / Spectrogram heatmap")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of detection")
    audio_duration_sec: Optional[float] = Field(None, description="Analyzed audio duration in seconds")
    is_speech: Optional[bool] = Field(None, description="Whether speech activity was detected via VAD")
    details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Diagnostic metrics")


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = "ok"
    timestamp: str
    model_loaded: bool
    use_embeddings: bool


class HistoryItem(BaseModel):
    """Session history item schema."""
    id: str
    verdict: str
    confidence: float
    timestamp: str
    source: str = "upload"  # "live" or "upload"
    audio_duration_sec: Optional[float] = None
