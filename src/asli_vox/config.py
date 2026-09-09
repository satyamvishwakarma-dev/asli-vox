"""Configuration settings for AsliVox."""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root directory (asli-vox/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application configuration loaded from environment or defaults."""
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "AsliVox"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Audio preprocessing
    SAMPLE_RATE: int = 16000
    N_MELS: int = 64
    N_FFT: int = 1024
    HOP_LENGTH: int = 512
    VAD_AGGRESSIVENESS: int = 3  # 0 (least aggressive) to 3 (most aggressive)
    VAD_FRAME_MS: int = 20       # 10, 20, or 30 ms frames for WebRTC VAD
    MIN_AUDIO_LEN_SEC: float = 0.25

    # Heavy model feature toggle (kept False by default for instant demo response)
    USE_EMBEDDINGS: bool = False

    # Model paths
    MODEL_PATH: Path = BASE_DIR / "src" / "asli_vox" / "models" / "lgb_spoof_model.joblib"
    FALLBACK_MODEL_PATH: Path = BASE_DIR / "src" / "asli_vox" / "backend" / "data" / "lgb_spoof_model.joblib"

    # Static frontend directory
    FRONTEND_DIR: Path = BASE_DIR / "frontend"

    # History capacity
    MAX_HISTORY: int = 20

    # Live streaming buffer
    STREAM_WINDOW_SEC: float = 1.0


settings = Settings()
