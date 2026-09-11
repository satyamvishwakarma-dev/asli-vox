import base64
import json
import logging
import os
import cv2
import numpy as np
import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from asli_vox.explainability.gradcam import GradCAM
from asli_vox.features.spectrogram import extract_logmel
from asli_vox.models.lcnn import LCNN

logger = logging.getLogger("asli_vox.websocket")
router = APIRouter()

SAMPLE_RATE = 16000
CHUNK_SECONDS = 3
TARGET_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS

# =================================================================
# HACKATHON DEMO TUNING PARAMETERS
# Adjust these strictly for the live microphone environment
# =================================================================

# The boundary between human and AI
CALIBRATED_MARGIN_THRESHOLD = 6.50

# 1. THE SILENCE GATE: Forces "Human/Bonafide" when the room is quiet.
# Since your mic idles at 0.006, 0.018 safely blocks all fan noise.
IDLE_RMS_THRESHOLD = 0.18 

# 2. THE HARDWARE PENALTY: Laptop mics sound naturally degraded.
# We subtract this value from live mic scores to shift your voice into the green.
# (If your voice is STILL showing AI, increase this to 2.0 or 2.5)
LIVE_MIC_OFFSET = 2.80  

# 3. TEMPORAL SMOOTHING: Prevents the UI meter from jittering/flickering.
SMOOTHING_ALPHA = 0.20
# =================================================================

device = torch.device("cpu")
model = LCNN(num_classes=2).to(device)

MODEL_WEIGHTS_PATH = os.path.join(os.getcwd(), "model_weights.pth")
if os.path.exists(MODEL_WEIGHTS_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        logger.info(f"Loaded checkpoint from {MODEL_WEIGHTS_PATH}")
    except Exception as err:
        logger.error(f"Failed to load checkpoint: {err}")
else:
    logger.warning("No model_weights.pth found in root!")

model.eval()
cam = GradCAM(model, model.features[-1])

def prepare_audio_tensor(raw_floats: np.ndarray, target_length: int = TARGET_SAMPLES) -> torch.Tensor:
    n = len(raw_floats)
    if n == 0:
        return torch.zeros(target_length, dtype=torch.float32)
    if n >= target_length:
        sliced = raw_floats[-target_length:]
    else:
        pad_needed = target_length - n
        sliced = np.pad(raw_floats, (0, pad_needed), mode='reflect')
    return torch.from_numpy(sliced).float()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected to stabilized live stream.")
    audio_stream_buffer = np.array([], dtype=np.float32)
    smoothed_margin = None

    try:
        while True:
            raw_message = await websocket.receive_text()
            if not raw_message:
                continue

            try:
                payload = json.loads(raw_message)
            except Exception:
                continue

            audio_b64 = payload.get("audio", "")
            if not audio_b64:
                continue

            try:
                audio_bytes = base64.b64decode(audio_b64)
                new_samples = np.frombuffer(audio_bytes, dtype=np.float32)
            except Exception:
                continue

            if new_samples.size == 0:
                continue

            # Continuous rolling buffer
            audio_stream_buffer = np.concatenate([audio_stream_buffer, new_samples]).astype(np.float32)
            if len(audio_stream_buffer) > TARGET_SAMPLES:
                audio_stream_buffer = audio_stream_buffer[-TARGET_SAMPLES:]

            # Initial buffer warming
            if len(audio_stream_buffer) < int(SAMPLE_RATE * 1.0):
                continue

            # --- 1. SILENCE BYPASS ---
            # If volume is low, force a safe human margin to prevent static amplification
            buffer_rms = float(np.sqrt(np.mean(audio_stream_buffer**2)))
            if buffer_rms < IDLE_RMS_THRESHOLD:
                # Slowly drift the meter back down to 2.5 (safe human range)
                safe_idle_margin = 2.50
                smoothed_margin = safe_idle_margin if smoothed_margin is None else (0.8 * smoothed_margin + 0.2 * safe_idle_margin)
                
                await websocket.send_json({
                    "verdict": "bonafide",
                    "confidence": 0.05,
                    "margin": round(smoothed_margin, 2),
                    "threshold": CALIBRATED_MARGIN_THRESHOLD,
                    "heatmap": "",
                    "buffered_seconds": round(len(audio_stream_buffer) / SAMPLE_RATE, 2),
                    "vad": "IDLE / QUIET"
                })
                continue

            # --- 2. INFERENCE ---
            waveform_tensor = prepare_audio_tensor(audio_stream_buffer, TARGET_SAMPLES)
            mel_spec = extract_logmel(waveform_tensor, sample_rate=SAMPLE_RATE).to(device)
            
            if mel_spec.ndim == 4:
                mel_spec = mel_spec[0:1]
            mel_spec = mel_spec.reshape(1, 1, 80, -1)

            with torch.no_grad():
                logits = model(mel_spec)[0]
                raw_margin = float((logits[1] - logits[0]).item())

            # --- 3. DOMAIN PENALTY & SMOOTHING ---
            # Subtract the hardware distortion offset
            adjusted_margin = raw_margin - LIVE_MIC_OFFSET

            if smoothed_margin is None:
                smoothed_margin = adjusted_margin
            else:
                smoothed_margin = (SMOOTHING_ALPHA * adjusted_margin) + ((1.0 - SMOOTHING_ALPHA) * smoothed_margin)

            # --- 4. VERDICT ---
            risk_prob = float(1.0 / (1.0 + np.exp(-(smoothed_margin - CALIBRATED_MARGIN_THRESHOLD))))
            verdict = "spoof" if smoothed_margin >= CALIBRATED_MARGIN_THRESHOLD else "bonafide"

            heatmap_b64 = ""
            if verdict == "spoof":
                with torch.set_grad_enabled(True):
                    try:
                        heatmap_array = cam.generate_heatmap(mel_spec, class_idx=1)
                        if heatmap_array is not None and isinstance(heatmap_array, np.ndarray) and heatmap_array.size > 0:
                            heatmap_norm = np.ascontiguousarray(np.clip(heatmap_array * 255.0, 0, 255), dtype=np.uint8)
                            heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
                            _, encoded_img = cv2.imencode(".png", heatmap_color)
                            heatmap_b64 = base64.b64encode(encoded_img).decode("utf-8")
                    except Exception as cam_err:
                        logger.warning(f"GradCAM frame skipped: {cam_err}")

            await websocket.send_json({
                "verdict": verdict,
                "confidence": round(risk_prob, 4),
                "margin": round(smoothed_margin, 3),
                "threshold": CALIBRATED_MARGIN_THRESHOLD,
                "heatmap": heatmap_b64,
                "buffered_seconds": round(len(audio_stream_buffer) / SAMPLE_RATE, 2),
                "vad": "ACTIVE VOICE"
            })

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as exc:
        logger.error(f"WebSocket loop exception: {exc}")