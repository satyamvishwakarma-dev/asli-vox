"""AsliVox: Real-Time AI Voice Deepfake / Spoof Detection."""
import uvicorn

__version__ = "0.1.0"


def main():
    """CLI entrypoint to launch the AsliVox server."""
    uvicorn.run("asli_vox.api.main:app", host="0.0.0.0", port=8000, reload=True)
