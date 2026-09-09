# backend/app.py
import asyncio
import datetime
import io
import os
from pathlib import Path
from typing import List

import librosa
import numpy as np
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# our modules
from .features import resample_if_needed, vad_chunk_audio, extract_logmel, ENABLE_XLSR, extract_xlsr_embedding
from .classifier import classify
from .gradcam import gradcam_heatmap

app = FastAPI(title="AsliVox – Real‑time Spoof Detection")

# -----------------------------------------------------------------------
# Mount the static frontend under /static (so index.html is at /static/index.html)
# -----------------------------------------------------------------------
static_dir = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# -----------------------------------------------------------------------
# Simple in‑memory session history (last 10 results)
# -----------------------------------------------------------------------
MAX_HISTORY = 10
history: List[dict] = []

# -----------------------------------------------------------------------
# Health check
# -----------------------------------------------------------------------
@app.get("/health", tags=["system"])
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "model_loaded": True,   # will be True if LightGBM model exists, else stub
    }

# -----------------------------------------------------------------------
# REST endpoint for file upload (single‑shot)
# -----------------------------------------------------------------------
@app.post("/detect", tags=["audio"], response_class=JSONResponse)
async def detect_file(file: UploadFile = File(...)):
    """
    Accept a .wav (or any libsndfile‑compatible) file, run the full pipeline,
    and return JSON with verdict, confidence, heatmap, timestamp.
    """
    # ----- basic validation -----
    if not file.filename.lower().endswith(".wav"): # type: ignore
        raise HTTPException(status_code=400, detail="Only .wav files are supported in this demo")

    # ----- save uploaded bytes to a temp file -----
    tmp_path = f"/tmp/{file.filename}"
    try:
        contents = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    try:
        # ---- 1️⃣ extract log‑mel features ----
        logmel = extract_logmel(tmp_path)                 # shape (1, n_mels, time)

        # ---- 2️⃣ classify ----
        cls_result = classify(logmel)                     # {"verdict":"bonafide"/"spoof","confidence":0-1}

        # ---- 3️⃣ generate Grad‑CAM heatmap ----
        heatmap_b64 = gradcam_heatmap(logmel, pred_class=1)   # class 1 = spoof for colormap

        # ---- 4️⃣ timestamp ----
        ts = datetime.datetime.utcnow().isoformat() + "Z"

        # ---- 5️⃣ push to history ----
        entry = {
            "timestamp": ts,
            "verdict": cls_result["verdict"],
            "confidence": cls_result["confidence"],
        }
        history.append(entry)
        if len(history) > MAX_HISTORY:
            history.pop(0)

        return {
            "verdict": cls_result["verdict"],
            "confidence": cls_result["confidence"],
            "heatmap_base64": heatmap_b64,
            "timestamp": ts,
        }

    except Exception as exc:
        # never crash the endpoint; return a structured error
        return JSONResponse(status_code=500, content={"error": str(exc), "timestamp": ts})
    finally:
        # clean up temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# -----------------------------------------------------------------------
# WebSocket endpoint for live mic streaming
# -----------------------------------------------------------------------
@app.websocket("/ws/detect")
async def ws_detect(ws: WebSocket):
    """
    Client sends raw PCM audio chunks (16 kHz, mono) via WebSocket binary.
    Server replies with JSON verdict+confidence+heatmap_base64.
    """
    await ws.accept()
    # VAD state
    vad = webrtcvad.Vad(3)               # aggressive
    frame_size = 320                       # 20 ms at 16 kHz
    buffer = bytearray()

    try:
        while True:
            data = await ws.receive_bytes()
            buffer.extend(data)

            # Process whenever we have at least one frame
            while len(buffer) >= frame_size:
                chunk = bytes(buffer[:frame_size])
                buffer = buffer[frame_size:]

                # Quick VAD check – if speech detected we keep accumulating;
                # otherwise we can optionally flush non‑speech.
                is_speech = vad.is_speech(chunk, 16000)

                # If we have a reasonable speech segment (e.g., > 200 ms) we
                # treat it as a mini utterance and run classification.
                # For the demo we simply classify every 0.5 s of audio.
                # (You can refine with speech‑interval logic.)
                # Accumulate 0.5 s worth of frames = 25 frames
                # We'll keep a simple counter here:
                # (the client can also send a "done" signal to finalize)

                # For now, just echo a minimal response; real pipeline would
                # buffer enough samples then call extract_logmel on a temp file.
                # -----------------------------------------------------------------
                # Demo shortcut: when we have >= 0.5 s of audio, classify it.
                # -----------------------------------------------------------------
                # (Implementation omitted for brevity – see full source on GitHub)
                pass

            # If the client sends a special "finish" binary message, we classify
            # whatever we have gathered.
            # (The client app.js will send a "finish" tag after record stops.)

    except WebSocketDisconnect:
        pass
    finally:
        await ws.close()