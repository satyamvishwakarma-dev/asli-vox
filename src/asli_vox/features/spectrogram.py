"""Log-mel spectrogram extraction and spectral feature analysis."""
import numpy as np
import librosa
from typing import Dict, Any

from asli_vox.config import settings
from asli_vox.utils.logger import logger


def extract_logmel(
    audio: np.ndarray,
    sr: int = settings.SAMPLE_RATE,
    n_mels: int = settings.N_MELS,
    n_fft: int = settings.N_FFT,
    hop_length: int = settings.HOP_LENGTH,
) -> np.ndarray:
    """
    Extract log-mel spectrogram from 1D audio array.
    Returns:
        np.ndarray with shape (1, n_mels, time_frames), float32.
    """
    if audio is None or audio.size == 0:
        return np.zeros((1, n_mels, 1), dtype=np.float32)

    # Ensure 1D mono float32
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0 if audio.shape[0] < audio.shape[1] else 1)
    audio = audio.astype(np.float32)

    # Handle very short audio by padding if less than n_fft
    if len(audio) < n_fft:
        pad_width = n_fft - len(audio)
        audio = np.pad(audio, (0, pad_width), mode="constant")

    try:
        # Compute mel power spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0,
        )

        # Convert to decibel scale (log-mel)
        logmel = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)

        # Reshape to (1, n_mels, time_steps)
        if logmel.ndim == 2:
            logmel = logmel[np.newaxis, :, :]

        return logmel.astype(np.float32)

    except Exception as e:
        logger.error(f"Error extracting log-mel spectrogram: {e}")
        # Safe fallback shape
        return np.zeros((1, n_mels, max(1, len(audio) // hop_length)), dtype=np.float32)


def extract_acoustic_features(audio: np.ndarray, sr: int = settings.SAMPLE_RATE) -> Dict[str, float]:
    """
    Extract key acoustic/spectral statistics for spoof heuristic analysis:
    - Spectral centroid: center of mass of frequency spectrum
    - Spectral rolloff: frequency below which 85% of energy resides
    - Spectral flatness: measure of noisiness vs tonality
    - Zero-crossing rate: rate of sign-changes
    - High-frequency energy ratio: artifact density indicator
    """
    if audio is None or audio.size < 320:
        return {
            "spectral_centroid_mean": 2000.0,
            "spectral_rolloff_mean": 4000.0,
            "spectral_flatness_mean": 0.01,
            "zcr_mean": 0.05,
            "high_freq_ratio": 0.15,
            "energy_variance": 0.01,
        }

    try:
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)
        flatness = librosa.feature.spectral_flatness(y=audio)
        zcr = librosa.feature.zero_crossing_rate(y=audio)

        # Frequency energy ratio (energy above 4kHz vs total energy)
        # Synthetic voices often show unnatural high-frequency energy cutoffs or artifacts
        fft_vals = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
        high_mask = freqs >= 4000
        total_energy = float(np.sum(fft_vals ** 2) + 1e-9)
        high_energy = float(np.sum(fft_vals[high_mask] ** 2))
        high_ratio = high_energy / total_energy

        # Temporal energy variation
        frames = librosa.util.frame(audio, frame_length=512, hop_length=256)
        frame_energies = np.sum(frames ** 2, axis=0)
        energy_var = float(np.var(frame_energies) / (np.mean(frame_energies) + 1e-6))

        return {
            "spectral_centroid_mean": float(np.mean(centroid)),
            "spectral_rolloff_mean": float(np.mean(rolloff)),
            "spectral_flatness_mean": float(np.mean(flatness)),
            "zcr_mean": float(np.mean(zcr)),
            "high_freq_ratio": float(high_ratio),
            "energy_variance": float(energy_var),
        }
    except Exception as e:
        logger.warning(f"Error computing acoustic features: {e}")
        return {
            "spectral_centroid_mean": 2000.0,
            "spectral_rolloff_mean": 4000.0,
            "spectral_flatness_mean": 0.01,
            "zcr_mean": 0.05,
            "high_freq_ratio": 0.15,
            "energy_variance": 0.01,
        }
