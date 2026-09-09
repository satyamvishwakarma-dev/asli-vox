"""REST endpoints for file upload, detection, history, and diagnostics."""
import uuid
from datetime import datetime, timezone
from typing import List
import numpy as np

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from asli_vox.config import settings
from asli_vox.utils.logger import logger
from asli_vox.utils.audio import load_audio_from_bytes, is_silent_or_empty, vad_chunk_audio
from asli_vox.features.spectrogram import extract_logmel
from asli_vox.features.embeddings import extract_xlsr_embedding
from asli_vox.models.lightgbm_baseline import classify
from asli_vox.explainability.gradcam import generate_gradcam_heatmap
from asli_vox.api.schemas import DetectionResponse, HistoryItem


router = APIRouter(tags=["Detection"])

# In-memory session history
_SESSION_HISTORY: List[HistoryItem] = []


def add_history_entry(verdict: str, confidence: float, source: str, duration: float) -> HistoryItem:
    """Store detection in session history."""
    item = HistoryItem(
        id=str(uuid.uuid4())[:8],
        verdict=verdict,
        confidence=confidence,
        timestamp=datetime.now(timezone.utc).isoformat(),
        source=source,
        audio_duration_sec=round(duration, 2),
    )
    _SESSION_HISTORY.insert(0, item)
    if len(_SESSION_HISTORY) > settings.MAX_HISTORY:
        _SESSION_HISTORY.pop()
    return item


@router.post("/detect", response_model=DetectionResponse)
async def detect_audio_file(file: UploadFile = File(...)):
    """
    Upload an audio file (.wav, .mp3, .ogg, .flac) for spoof/deepfake detection.
    Runs log-mel extraction, classification, and Grad-CAM heatmap generation.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    logger.info(f"Received audio upload: {file.filename} (content-type: {file.content_type})")

    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(status_code=400, detail="Could not read uploaded audio file.")

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 1. Decode & resample to 16kHz mono
    audio, sr = load_audio_from_bytes(content, target_sr=settings.SAMPLE_RATE)
    duration = float(len(audio) / sr) if len(audio) > 0 else 0.0

    # 2. Check for empty or pure silence
    if is_silent_or_empty(audio):
        logger.info("Audio is silent or empty.")
        ts = datetime.now(timezone.utc).isoformat()
        dummy_mel = extract_logmel(np.zeros(sr, dtype=np.float32))
        heatmap_b64 = generate_gradcam_heatmap(dummy_mel, verdict="bonafide")
        return DetectionResponse(
            verdict="bonafide",
            confidence=0.50,
            heatmap_base64=heatmap_b64,
            timestamp=ts,
            audio_duration_sec=duration,
            is_speech=False,
            details={"note": "Audio was silent or contained no detectable voice signal."},
        )

    # 3. Voice Activity Detection
    intervals = vad_chunk_audio(audio, sr=sr, aggressiveness=settings.VAD_AGGRESSIVENESS)
    has_speech = len(intervals) > 0

    # 4. Feature Extraction: Log-mel spectrogram
    logmel = extract_logmel(
        audio,
        sr=sr,
        n_mels=settings.N_MELS,
        n_fft=settings.N_FFT,
        hop_length=settings.HOP_LENGTH,
    )

    # Optional XLSR embeddings (gated by config)
    xlsr_emb = extract_xlsr_embedding(audio, sr=sr)

    # 5. Model Inference (LightGBM or heuristic baseline stub)
    classification = classify(logmel, audio, sr=sr)
    verdict = classification["verdict"]
    confidence = classification["confidence"]

    # 6. Explainability: Grad-CAM heatmap
    heatmap_b64 = generate_gradcam_heatmap(logmel, verdict=verdict)

    # 7. Record to session history
    add_history_entry(verdict, confidence, source="upload", duration=duration)

    now_iso = datetime.now(timezone.utc).isoformat()
    return DetectionResponse(
        verdict=verdict,
        confidence=confidence,
        heatmap_base64=heatmap_b64,
        timestamp=now_iso,
        audio_duration_sec=round(duration, 2),
        is_speech=has_speech,
        details={
            "model_source": classification["model_source"],
            "spoof_probability": classification["spoof_probability"],
            "has_embeddings": xlsr_emb is not None,
            "filename": file.filename,
        },
    )


@router.get("/history", response_model=List[HistoryItem])
async def get_session_history():
    """Get the recent session detection history."""
    return _SESSION_HISTORY


@router.delete("/history")
async def clear_session_history():
    """Clear session history."""
    _SESSION_HISTORY.clear()
    return {"message": "History cleared"}
