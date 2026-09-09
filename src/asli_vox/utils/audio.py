"""Audio processing utilities: loading, resampling, normalization, and VAD."""
import io
import wave
import numpy as np
import librosa
import soundfile as sf
from typing import List, Tuple, Optional

from asli_vox.utils.logger import logger


# Try webrtcvad; fallback to direct _webrtcvad or energy-based VAD
class _WebRtcVadShim:
    """Shim for webrtcvad when pkg_resources is missing."""
    def __init__(self, mode: int = 3):
        try:
            import _webrtcvad
            self._vad = _webrtcvad.create()
            _webrtcvad.init(self._vad)
            _webrtcvad.set_mode(self._vad, mode)
            self._available = True
        except Exception:
            self._available = False

    def is_speech(self, buf: bytes, sample_rate: int) -> bool:
        if not self._available:
            return True
        import _webrtcvad
        length = len(buf) // 2
        return bool(_webrtcvad.process(self._vad, sample_rate, buf, length))


def create_vad(aggressiveness: int = 3):
    """Create a VAD instance using webrtcvad or the shim."""
    try:
        import webrtcvad
        return webrtcvad.Vad(aggressiveness)
    except Exception:
        try:
            return _WebRtcVadShim(aggressiveness)
        except Exception as e:
            logger.warning(f"Failed to initialize webrtcvad: {e}. Falling back to energy VAD.")
            return None


def load_audio_from_bytes(data: bytes, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Safely load audio from raw bytes (WAV, MP3, OGG, or raw PCM).
    Returns (audio_1d_float32, sample_rate).
    """
    if not data or len(data) == 0:
        return np.zeros(0, dtype=np.float32), target_sr

    # 1. Try soundfile (handles WAV, FLAC, OGG, etc.)
    try:
        buf = io.BytesIO(data)
        audio, sr = sf.read(buf, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = resample_audio(audio, sr, target_sr)
        return audio.astype(np.float32), target_sr
    except Exception:
        pass

    # 2. Try standard library wave module
    try:
        buf = io.BytesIO(data)
        with wave.open(buf, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

            if sampwidth == 2:
                raw = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 1:
                raw = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            elif sampwidth == 4:
                raw = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raw = np.frombuffer(frames, dtype=np.float32)

            if n_channels > 1:
                raw = raw.reshape(-1, n_channels).mean(axis=1)

            audio = resample_audio(raw, framerate, target_sr)
            return audio.astype(np.float32), target_sr
    except Exception:
        pass

    # 3. Try interpreting as raw PCM 16-bit 16kHz mono (common for streaming WebSocket)
    try:
        if len(data) % 2 == 0:
            raw_int16 = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            # Sanity check values
            if len(raw_int16) > 0 and not np.isnan(raw_int16).any():
                return raw_int16.astype(np.float32), target_sr
    except Exception:
        pass

    # 4. Try librosa with soundfile/audioread backend
    try:
        buf = io.BytesIO(data)
        audio, sr = librosa.load(buf, sr=target_sr, mono=True)
        return audio.astype(np.float32), target_sr
    except Exception as e:
        logger.error(f"Failed to decode audio bytes: {e}")
        return np.zeros(0, dtype=np.float32), target_sr


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample 1D audio to target sample rate."""
    if audio.size == 0:
        return audio

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1 if audio.shape[1] < audio.shape[0] else 0)

    if orig_sr == target_sr:
        return audio.astype(np.float32)

    try:
        resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        return resampled.astype(np.float32)
    except Exception as e:
        logger.warning(f"Librosa resample failed ({e}), using linear interpolation fallback.")
        num_samples = int(round(len(audio) * float(target_sr) / orig_sr))
        x_orig = np.linspace(0, len(audio), len(audio), endpoint=False)
        x_target = np.linspace(0, len(audio), num_samples, endpoint=False)
        return np.interp(x_target, x_orig, audio).astype(np.float32)


def is_silent_or_empty(audio: np.ndarray, threshold_rms: float = 0.003) -> bool:
    """Check if audio is empty, too short, or near-complete silence."""
    if audio is None or len(audio) < 160:  # < 10ms at 16kHz
        return True
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return rms < threshold_rms


def vad_chunk_audio(
    audio: np.ndarray,
    sr: int = 16000,
    aggressiveness: int = 3,
    frame_ms: int = 20,
) -> List[Tuple[float, float]]:
    """
    Perform Voice Activity Detection (VAD) and return list of (start_sec, end_sec).
    Falls back gracefully to energy-based VAD if WebRTC VAD is unavailable.
    """
    if audio.size == 0:
        return []

    # Frame size must be 10, 20, or 30 ms for webrtcvad at 16kHz:
    # 10ms -> 160 samples, 20ms -> 320 samples, 30ms -> 480 samples
    frame_size = int(sr * (frame_ms / 1000.0))
    if frame_size not in (160, 320, 480):
        frame_size = 320

    vad = create_vad(aggressiveness)

    # Convert audio to int16 PCM
    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767.0).astype(np.int16)

    num_frames = len(audio_int16) // frame_size
    if num_frames == 0:
        return [(0.0, float(len(audio) / sr))]

    frames_speech = []
    if vad is not None:
        try:
            for i in range(num_frames):
                chunk = audio_int16[i * frame_size : (i + 1) * frame_size].tobytes()
                speech = vad.is_speech(chunk, sr)
                frames_speech.append(speech)
        except Exception as e:
            logger.warning(f"WebRTC VAD execution error: {e}. Using energy VAD.")
            frames_speech = []

    # Energy-based VAD fallback if needed
    if not frames_speech:
        rms_threshold = 0.008
        for i in range(num_frames):
            frame_float = audio[i * frame_size : (i + 1) * frame_size]
            rms = np.sqrt(np.mean(frame_float ** 2))
            frames_speech.append(rms > rms_threshold)

    # Aggregate frames into intervals
    intervals: List[Tuple[float, float]] = []
    in_speech = False
    start_time = 0.0

    for i, is_active in enumerate(frames_speech):
        t = (i * frame_size) / sr
        if is_active and not in_speech:
            in_speech = True
            start_time = t
        elif not is_active and in_speech:
            in_speech = False
            end_time = t
            if (end_time - start_time) >= 0.06:  # filter negligible clicks
                intervals.append((round(start_time, 3), round(end_time, 3)))

    if in_speech:
        end_time = (num_frames * frame_size) / sr
        intervals.append((round(start_time, 3), round(end_time, 3)))

    # If no speech intervals found but audio has some energy, return full audio span
    if not intervals and not is_silent_or_empty(audio):
        intervals.append((0.0, round(float(len(audio) / sr), 3)))

    return intervals
