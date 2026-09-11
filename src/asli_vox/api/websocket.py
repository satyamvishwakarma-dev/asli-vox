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

# Calibrated Threshold: human voice sits around ~2.8 - 4.2 | AI sits around ~5.8 - 10.0
CALIBRATED_MARGIN_THRESHOLD = 5.50

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
    logger.info("Client connected.")
    audio_stream_buffer = np.array([], dtype=np.float32)

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
            # --- DIAGNOSTIC PRINT ---
            frame_rms = float(np.sqrt(np.mean(new_samples**2)))
            frame_max = float(np.max(np.abs(new_samples))) if len(new_samples) > 0 else 0.0
            print(f"[RX CHUNK] Len: {len(new_samples)} | Max: {frame_max:.4f} | RMS: {frame_rms:.4f}")
            # ------------------------

            if new_samples.size == 0:
                continue

            # Continuous rolling buffer (no silence discards)
            audio_stream_buffer = np.concatenate([audio_stream_buffer, new_samples]).astype(np.float32)
            if len(audio_stream_buffer) > TARGET_SAMPLES:
                audio_stream_buffer = audio_stream_buffer[-TARGET_SAMPLES:]

            # Initial buffer warming (first 1 second on connect)
            if len(audio_stream_buffer) < int(SAMPLE_RATE * 1.0):
                continue

            waveform_tensor = prepare_audio_tensor(audio_stream_buffer, TARGET_SAMPLES)
            mel_spec = extract_logmel(waveform_tensor, sample_rate=SAMPLE_RATE).to(device)
            if mel_spec.ndim == 4:
                mel_spec = mel_spec[0:1]
            mel_spec = mel_spec.reshape(1, 1, 80, -1)

            with torch.no_grad():
                logits = model(mel_spec)[0]
                margin = float((logits[1] - logits[0]).item())

            risk_prob = float(1.0 / (1.0 + np.exp(-(margin - CALIBRATED_MARGIN_THRESHOLD))))
            verdict = "spoof" if margin >= CALIBRATED_MARGIN_THRESHOLD else "bonafide"

            heatmap_b64 = ""
            if margin >= CALIBRATED_MARGIN_THRESHOLD:
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
                "margin": round(margin, 3),
                "threshold": CALIBRATED_MARGIN_THRESHOLD,
                "heatmap": heatmap_b64,
                "buffered_seconds": round(len(audio_stream_buffer) / SAMPLE_RATE, 2),
                "vad": "ACTIVE"
            })

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as exc:
        logger.error(f"WebSocket loop exception: {exc}")