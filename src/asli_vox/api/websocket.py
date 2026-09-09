"""WebSocket endpoint for real-time live microphone streaming detection."""
import json
from datetime import datetime, timezone
from typing import Optional
import numpy as np

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from asli_vox.config import settings
from asli_vox.utils.logger import logger
from asli_vox.utils.audio import load_audio_from_bytes, is_silent_or_empty, vad_chunk_audio
from asli_vox.features.spectrogram import extract_logmel
from asli_vox.features.embeddings import extract_xlsr_embedding
from asli_vox.models.lightgbm_baseline import classify
from asli_vox.explainability.gradcam import generate_gradcam_heatmap
from asli_vox.api.routes import add_history_entry


router = APIRouter(tags=["WebSocket Detection"])


def _process_audio_chunk(raw_bytes: bytes) -> Optional[dict]:
    """
    Safely process accumulated audio bytes through the full detection pipeline.
    Never raises an exception — returns safe neutral payload on error or silence.
    """
    ts = datetime.now(timezone.utc).isoformat()

    if not raw_bytes or len(raw_bytes) < 320:  # < 10ms
        return None

    try:
        audio, sr = load_audio_from_bytes(raw_bytes, target_sr=settings.SAMPLE_RATE)
        duration = float(len(audio) / sr) if len(audio) > 0 else 0.0

        # Handle silence or near-zero energy
        if is_silent_or_empty(audio):
            dummy_mel = extract_logmel(np.zeros(sr, dtype=np.float32))
            heatmap_b64 = generate_gradcam_heatmap(dummy_mel, verdict="bonafide")
            return {
                "verdict": "bonafide",
                "confidence": 0.50,
                "heatmap_base64": heatmap_b64,
                "timestamp": ts,
                "audio_duration_sec": round(duration, 2),
                "is_speech": False,
                "details": {"note": "Silent or non-vocal audio frame."},
            }

        # VAD segmentation
        intervals = vad_chunk_audio(audio, sr=sr, aggressiveness=settings.VAD_AGGRESSIVENESS)
        has_speech = len(intervals) > 0

        # Log-mel Spectrogram
        logmel = extract_logmel(
            audio,
            sr=sr,
            n_mels=settings.N_MELS,
            n_fft=settings.N_FFT,
            hop_length=settings.HOP_LENGTH,
        )

        # Classification (LightGBM or heuristic baseline stub)
        res = classify(logmel, audio, sr=sr)
        verdict = res["verdict"]
        confidence = res["confidence"]

        # Grad-CAM Heatmap
        heatmap_b64 = generate_gradcam_heatmap(logmel, verdict=verdict)

        # Record to session history
        add_history_entry(verdict, confidence, source="live", duration=duration)

        return {
            "verdict": verdict,
            "confidence": confidence,
            "heatmap_base64": heatmap_b64,
            "timestamp": ts,
            "audio_duration_sec": round(duration, 2),
            "is_speech": has_speech,
            "details": {
                "model_source": res["model_source"],
                "spoof_probability": res["spoof_probability"],
            },
        }

    except Exception as exc:
        logger.error(f"Error processing streaming audio chunk: {exc}", exc_info=True)
        # Never crash the websocket connection on bad input
        dummy_mel = extract_logmel(np.zeros(settings.SAMPLE_RATE, dtype=np.float32))
        return {
            "verdict": "bonafide",
            "confidence": 0.50,
            "heatmap_base64": generate_gradcam_heatmap(dummy_mel, verdict="bonafide"),
            "timestamp": ts,
            "audio_duration_sec": 0.0,
            "is_speech": False,
            "details": {"error": str(exc)},
        }


@router.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """
    WebSocket endpoint accepting live audio data from browser microphone.
    Supports:
      - Binary audio frames (PCM 16-bit 16kHz or WAV chunks)
      - Text control messages: {"type": "finish"}, {"type": "reset"}
    Returns structured JSON: verdict, confidence, heatmap_base64, timestamp.
    Guaranteed not to crash the connection on bad or silent audio.
    """
    await websocket.accept()
    logger.info("WebSocket client connected to /ws/detect")

    buffer = bytearray()
    # 1.0 second of 16-bit mono 16kHz PCM = 32000 bytes
    min_stream_bytes = int(settings.SAMPLE_RATE * 2 * settings.STREAM_WINDOW_SEC)

    try:
        while True:
            msg = await websocket.receive()

            if "bytes" in msg and msg["bytes"] is not None:
                data = msg["bytes"]
                buffer.extend(data)

                # If enough audio accumulated for live window
                if len(buffer) >= min_stream_bytes:
                    payload = _process_audio_chunk(bytes(buffer))
                    if payload is not None:
                        await websocket.send_json(payload)
                    # Keep a 50% sliding overlap window
                    keep_bytes = min_stream_bytes // 2
                    buffer = buffer[-keep_bytes:]

            elif "text" in msg and msg["text"] is not None:
                try:
                    cmd = json.loads(msg["text"])
                except Exception:
                    cmd = {"type": msg["text"].strip().lower()}

                action = cmd.get("type", "")

                if action == "finish":
                    # Client finished recording: process whatever remains in buffer
                    if len(buffer) > 0:
                        payload = _process_audio_chunk(bytes(buffer))
                        if payload is not None:
                            await websocket.send_json(payload)
                        buffer.clear()

                elif action == "reset":
                    buffer.clear()
                    await websocket.send_json({"status": "buffer_cleared"})

                elif action == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.now(timezone.utc).isoformat()})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected normally.")
    except Exception as exc:
        logger.warning(f"WebSocket session error: {exc}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
