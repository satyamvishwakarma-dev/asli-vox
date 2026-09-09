# backend/features.py
import numpy as np
import librosa
import webrtcvad

# ---------------------------------------------------------------------------
# 1️⃣  Resample + VAD chunking
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000  # target for VAD & model input
N_MFCC = 64  # number of mel bins for our fallback features
HOP_LENGTH = 512
N_FFT = 1024


def resample_if_needed(y: np.ndarray, orig_sr: int) -> np.ndarray:
    """Librosa resample to 16 kHz if needed."""
    if orig_sr == SAMPLE_RATE:
        return y
    return librosa.resample(y, orig_sr=orig_sr, target_sr=SAMPLE_RATE)


def vad_chunk_audio(
    raw_path: str, aggressiveness: int = 3
) -> list[tuple[float, float]]:
    """
    Use webrtcvad to detect speech intervals.
    Returns list of (start_sec, end_sec) tuples.
    """
    # Read original wav (librosa loads float32, 0‑1) – we need the raw PCM for VAD
    import soundfile as sf

    data, sr = sf.read(raw_path)
    data = resample_if_needed(data, sr)
    # Convert to int16 (VAD expects 16‑kHz, 16‑bit mono)
    import array

    if data.dtype != np.int16:
        data_int16 = (data * np.iinfo(np.int16).max).astype(np.int16)
    else:
        data_int16 = data

    vad = webrtcvad.Vad(aggressiveness)
    # webrtcvad.process expects frames of 20 ms (320 samples at 16 kHz)
    frame_size = 320
    channels = 1
    # mono already
    num_frames = len(data_int16) // frame_size
    frames = []
    for i in range(num_frames):
        chunk = data_int16[i * frame_size : (i + 1) * frame_size]
        is_speech = vad.is_speech(chunk.tobytes(), SAMPLE_RATE)
        frames.append(is_speech)

    # simple run‑length to get intervals
    intervals = []
    in_speech = False
    start = 0
    for i, speech in enumerate(frames):
        if speech and not in_speech:
            in_speech = True
            start = i * frame_size / SAMPLE_RATE
        elif not speech and in_speech:
            in_speech = False
            end = i * frame_size / SAMPLE_RATE
            intervals.append((start, end))
    if in_speech:
        end = num_frames * frame_size / SAMPLE_RATE
        intervals.append((start, end))
    return intervals


# ---------------------------------------------------------------------------
# 2️⃣  Feature extraction (fallback: log‑mel spectrogram)
# ---------------------------------------------------------------------------


def extract_logmel(audio_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load audio, resample, compute log‑mel spectrogram.
    Shape: (1, n_mels, time) ready for a lightweight classifier.
    """
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    # add channel dim & batch dim -> (1, n_mels, time)
    logmel = logmel[np.newaxis, :, :]
    return logmel.astype(np.float32)


# ---------------------------------------------------------------------------
# 3️⃣  Optional wav2vec2‑XLSR embedding (toggle in config)
# ---------------------------------------------------------------------------
# If you later want to use a large pretrained model, set ENABLE_XLSR=True.
# The function below is a *placeholder* that returns a dummy vector so the
# rest of the pipeline never crashes.

ENABLE_XLSR = False  # keep False for the demo


def extract_xlsr_embedding(audio_path: str) -> np.ndarray | None:
    if not ENABLE_XLSR:
        return None
    # In a real deployment you would load the model once at startup:
    #   from transformers import Wav2Vec2Model, Wav2Vec2Processor
    #   processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xls-r-300m")
    #   model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")
    #   ... extract embeddings ...
    return None  # stub: nothing loaded -> None
