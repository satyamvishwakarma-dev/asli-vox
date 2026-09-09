"""wav2vec2-XLSR representation extraction, gated behind configuration flag."""
from typing import Optional
import numpy as np

from asli_vox.config import settings
from asli_vox.utils.logger import logger

# Lazy model cache
_PROCESSOR = None
_MODEL = None


def is_embeddings_enabled() -> bool:
    """Check if XLSR embeddings are enabled in configuration."""
    return bool(settings.USE_EMBEDDINGS)


def extract_xlsr_embedding(audio: np.ndarray, sr: int = 16000) -> Optional[np.ndarray]:
    """
    Extract wav2vec2-XLSR embeddings from audio array.
    Gated behind USE_EMBEDDINGS config flag (default=False).
    When disabled, returns None immediately with zero latency and zero RAM overhead.
    """
    global _PROCESSOR, _MODEL

    if not is_embeddings_enabled():
        return None

    try:
        import torch
        from transformers import Wav2Vec2Model, Wav2Vec2Processor

        if _PROCESSOR is None or _MODEL is None:
            logger.info("USE_EMBEDDINGS=True: Lazy loading Wav2Vec2-XLSR-300M model...")
            model_name = "facebook/wav2vec2-xls-r-300m"
            _PROCESSOR = Wav2Vec2Processor.from_pretrained(model_name)
            _MODEL = Wav2Vec2Model.from_pretrained(model_name)
            _MODEL.eval()

        inputs = _PROCESSOR(audio, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            outputs = _MODEL(**inputs)
            # Pool over time dimension: (batch, time, hidden) -> (hidden,)
            hidden_states = outputs.last_hidden_state
            pooled = torch.mean(hidden_states, dim=1).squeeze(0).cpu().numpy()
            return pooled.astype(np.float32)

    except ImportError:
        logger.warning("transformers package not installed; skipping XLSR embedding extraction.")
        return None
    except Exception as exc:
        logger.error(f"Failed to extract wav2vec2-XLSR embedding: {exc}")
        return None
