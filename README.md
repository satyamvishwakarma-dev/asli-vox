# AsliVox

AsliVox is a FastAPI-based deepfake voice detection demo built for Smart India Hackathon by Team Apex.  
It provides:

- **Live microphone analysis** over WebSocket (`/ws`)
- **Audio file verification** over REST (`/api/verify-file`)
- **Frontend dashboard** served directly by the backend (`/`)
- **Grad-CAM heatmap output** for spoof detections

---

## Tech Stack

- Python 3.12+
- FastAPI + Uvicorn
- PyTorch + Torchaudio
- Librosa + SoundFile
- OpenCV (heatmap rendering)
- Vanilla HTML/CSS/JS frontend

---

## Project Structure

```text
asli-vox/
├── frontend/                  # Web UI (served by FastAPI)
├── src/asli_vox/
│   ├── api/
│   │   ├── main.py            # FastAPI app + frontend mounting
│   │   ├── routes.py          # REST endpoint (/api/verify-file)
│   │   └── websocket.py       # Live stream endpoint (/ws)
│   ├── explainability/        # Grad-CAM implementation
│   ├── features/              # Log-mel and feature extraction
│   └── models/                # LCNN architecture and baseline models
├── tests/                     # Pipeline smoke test
├── run.py                     # Local launch script
└── pyproject.toml
```

---

## Prerequisites

1. Python **3.12 or newer**
2. A trained model weights file named:

```text
model_weights.pth
```

Place this file in the **repository root** (`asli-vox/model_weights.pth`) before starting the server.

---

## Setup

### Option A: Using `uv` (recommended)

```bash
uv sync
uv run python run.py
```

### Option B: Using `pip`

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e .
python run.py
```

Server runs on:

```text
http://127.0.0.1:8000
```

Open this URL in a browser to access the dashboard.

---

## API Endpoints

### `GET /`
Serves the frontend dashboard (`frontend/index.html`).

### `POST /api/verify-file`
Upload an audio file as `multipart/form-data` with field name `file`.

Example response:

```json
{
  "filename": "sample.wav",
  "verdict": "bonafide",
  "margin": 2.143,
  "threshold": 5.0,
  "confidence": 0.1254,
  "logits": [-1.23, 0.91]
}
```

### `WS /ws`
Live streaming detection endpoint used by the UI.
Client sends Base64-encoded float32 audio chunks in JSON:

```json
{ "audio": "<base64-float32-bytes>" }
```

---

## Environment Configuration

Copy and adjust defaults if needed:

```bash
cp .env.example .env
```

Main tunables include:
- sample rate and mel settings
- VAD behavior
- model path flags
- debug mode

---

## Running Tests

```bash
uv run pytest
```

---

## Notes

- The live pipeline and thresholds in `api/websocket.py` are tuned for demo conditions.
- Frontend supports live mic streaming and file upload from the same dashboard.

---

## Team

Built by **Team Apex** for **Smart India Hackathon**.
