import base64
import json
import logging
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from asli_vox.explainability.gradcam import GradCAM
from asli_vox.features.spectrogram import extract_logmel
from asli_vox.models.lcnn import LCNN

logger = logging.getLogger("asli_vox.websocket")
router = APIRouter()

SAMPLE_RATE = 16000
CHUNK_SECONDS = 3
TARGET_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS
SPOOF_DECISION_THRESHOLD = 0.65

device = torch.device("cpu")
model = LCNN(num_classes=2).to(device)

MODEL_WEIGHTS_PATH = os.path.join(os.getcwd(), "model_weights.pth")
if os.path.exists(MODEL_WEIGHTS_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        logger.info(f"Loaded checkpoint: {MODEL_WEIGHTS_PATH}")
    except Exception as err:
        logger.error(f"Checkpoint error: {err}. Using initialized model.")
else:
    logger.warning("No model_weights.pth found. Using initialized model.")

model.eval()
cam = GradCAM(model, model.features[-1])

def prepare_audio_tensor(raw_floats: np.ndarray, target_length: int = TARGET_SAMPLES) -> torch.Tensor:
    if len(raw_floats) < target_length:
        padded = np.pad(raw_floats, (0, target_length - len(raw_floats)), mode="constant")
    else:
        padded = raw_floats[-target_length:]
    return torch.from_numpy(padded).float()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted.")
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

            if new_samples.size == 0:
                continue

            audio_stream_buffer = np.concatenate([audio_stream_buffer, new_samples])
            if len(audio_stream_buffer) > TARGET_SAMPLES:
                audio_stream_buffer = audio_stream_buffer[-TARGET_SAMPLES:]

            # Preprocess: guarantees (1, 1, 80, time)
            waveform_tensor = prepare_audio_tensor(audio_stream_buffer, TARGET_SAMPLES)
            mel_spec = extract_logmel(waveform_tensor, sample_rate=SAMPLE_RATE).to(device)
            if mel_spec.ndim == 4:
                mel_spec = mel_spec[0:1]
            mel_spec = mel_spec.reshape(1, 1, 80, -1)

            with torch.set_grad_enabled(True):
                logits = model(mel_spec)
                probs = F.softmax(logits.view(1, -1), dim=-1)
                spoof_prob = float(probs[0, 1].item()) if probs.shape[1] > 1 else float(probs[0, 0].item())

                heatmap_b64 = ""
                if spoof_prob >= 0.40:
                    try:
                        heatmap_array = cam.generate_heatmap(mel_spec, class_idx=1)
                        if heatmap_array is not None and isinstance(heatmap_array, np.ndarray) and heatmap_array.size > 0:
                            # Fix UMat/OpenCV typing: explicitly cast contiguous uint8 array
                            heatmap_norm = np.ascontiguousarray(np.clip(heatmap_array * 255.0, 0, 255), dtype=np.uint8)
                            heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
                            _, encoded_img = cv2.imencode(".png", heatmap_color)
                            heatmap_b64 = base64.b64encode(encoded_img).decode("utf-8")
                    except Exception as cam_err:
                        logger.warning(f"GradCAM frame skipped: {cam_err}")
                        heatmap_b64 = ""

            verdict = "spoof" if spoof_prob >= SPOOF_DECISION_THRESHOLD else "bonafide"

            await websocket.send_json({
                "verdict": verdict,
                "confidence": round(spoof_prob, 4),
                "threshold": SPOOF_DECISION_THRESHOLD,
                "heatmap": heatmap_b64,
                "buffered_seconds": round(len(audio_stream_buffer) / SAMPLE_RATE, 2)
            })

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as exc:
        logger.error(f"WebSocket loop exception: {exc}")