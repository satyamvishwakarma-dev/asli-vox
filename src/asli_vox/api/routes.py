import io
import torch
import numpy as np
import soundfile as sf
import librosa
from fastapi import APIRouter, File, UploadFile
from asli_vox.models.lcnn import LCNN
from asli_vox.features.spectrogram import extract_logmel

router = APIRouter()

device = torch.device("cpu")
model = LCNN(num_classes=2).to(device)
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval()

CALIBRATED_MARGIN_THRESHOLD = 5.00

@router.post("/api/verify-file")
async def verify_file(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Decode audio in memory
    try:
        data, sr = sf.read(io.BytesIO(contents), dtype='float32')
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != 16000:
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    except Exception:
        data, _ = librosa.load(io.BytesIO(contents), sr=16000, mono=True)

    # Standardize to 3 seconds
    target_samples = 48000
    if len(data) >= target_samples:
        waveform = data[:target_samples]
    else:
        waveform = np.pad(data, (0, target_samples - len(data)), mode='reflect')

    wave_tensor = torch.from_numpy(waveform).float()
    mel_spec = extract_logmel(wave_tensor, sample_rate=16000).to(device)

    with torch.no_grad():
        logits = model(mel_spec)[0]
        margin = float((logits[1] - logits[0]).item())
        risk_score = float(1.0 / (1.0 + np.exp(-(margin - CALIBRATED_MARGIN_THRESHOLD))))

    verdict = "spoof" if margin >= CALIBRATED_MARGIN_THRESHOLD else "bonafide"

    return {
        "filename": file.filename,
        "verdict": verdict,
        "margin": round(margin, 3),
        "threshold": CALIBRATED_MARGIN_THRESHOLD,
        "confidence": round(risk_score, 4),
        "logits": [round(float(logits[0]), 3), round(float(logits[1]), 3)]
    }