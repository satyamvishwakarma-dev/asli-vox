"""End-to-end smoke test for AsliVox detection pipeline."""
import io
import wave
import numpy as np

from asli_vox.utils.audio import load_audio_from_bytes, vad_chunk_audio, is_silent_or_empty
from asli_vox.features.spectrogram import extract_logmel
from asli_vox.models.lightgbm_baseline import classify, is_model_loaded
from asli_vox.explainability.gradcam import generate_gradcam_heatmap
from asli_vox.api.schemas import DetectionResponse


def create_mock_wav(duration_sec: float = 1.0, sr: int = 16000, freq: float = 440.0) -> bytes:
    """Create in-memory PCM 16-bit WAV bytes."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Fundamental + harmonics + slight noise
    signal = 0.5 * np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(2 * np.pi * 2 * freq * t)
    signal_int16 = (signal * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(signal_int16.tobytes())
    return buf.getvalue()


def test_full_pipeline_mock_audio():
    """Test entire detection flow from audio bytes to schema output."""
    wav_bytes = create_mock_wav(duration_sec=1.5, sr=16000, freq=300.0)

    # 1. Load audio
    audio, sr = load_audio_from_bytes(wav_bytes, target_sr=16000)
    assert len(audio) > 0
    assert sr == 16000
    assert not is_silent_or_empty(audio)

    # 2. VAD
    intervals = vad_chunk_audio(audio, sr=sr)
    assert len(intervals) > 0

    # 3. Log-mel Spectrogram
    logmel = extract_logmel(audio, sr=sr)
    assert logmel.ndim == 3
    assert logmel.shape[0] == 1
    assert logmel.shape[1] == 64  # n_mels
    assert logmel.shape[2] > 0

    # 4. Classification
    res = classify(logmel, audio, sr=sr)
    assert res["verdict"] in ("bonafide", "spoof")
    assert 0.0 <= res["confidence"] <= 1.0
    assert 0.0 <= res["spoof_probability"] <= 1.0
    assert "model_source" in res

    # 5. Grad-CAM Heatmap
    heatmap_b64 = generate_gradcam_heatmap(logmel, verdict=res["verdict"])
    assert heatmap_b64.startswith("data:image/png;base64,")
    assert len(heatmap_b64) > 100

    # 6. Response Schema
    detection = DetectionResponse(
        verdict=res["verdict"],
        confidence=res["confidence"],
        heatmap_base64=heatmap_b64,
        timestamp="2026-09-09T12:00:00Z",
        audio_duration_sec=1.5,
        is_speech=True,
        details=res,
    )
    assert detection.verdict in ("bonafide", "spoof")
    assert detection.confidence >= 0.5


def test_silence_handling():
    """Verify that silent audio does not crash and returns safe payload."""
    silent_audio = np.zeros(16000, dtype=np.float32)
    assert is_silent_or_empty(silent_audio)

    logmel = extract_logmel(silent_audio)
    res = classify(logmel, silent_audio)
    assert res["verdict"] in ("bonafide", "spoof")
    heatmap_b64 = generate_gradcam_heatmap(logmel, verdict=res["verdict"])
    assert heatmap_b64.startswith("data:image/png;base64,")


if __name__ == "__main__":
    print("Running pipeline test with mock audio...")
    test_full_pipeline_mock_audio()
    print("Running silence handling test...")
    test_silence_handling()
    print("All tests passed successfully!")
