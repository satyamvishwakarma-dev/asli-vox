import numpy as np
import torch
import torchaudio

_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    win_length=400,
    hop_length=160,
    n_mels=80
)
_amp_to_db = torchaudio.transforms.AmplitudeToDB()

def extract_logmel(waveform: torch.Tensor, sample_rate: int = 16000, **kwargs) -> torch.Tensor:
    """Extracts peak-scaled, CMVN-normalized log-mel spectrogram matching Kaggle training.
    
    Accepts 1D (T,), 2D (C, T), or 3D (B, C, T).
    Guarantees output shape: (1, 1, 80, time_frames).
    """
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # (1, T)
    elif waveform.ndim == 2:
        if waveform.shape[0] > waveform.shape[1] and waveform.shape[1] <= 2:
            waveform = waveform.transpose(0, 1)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
    elif waveform.ndim == 3:
        if waveform.shape[1] > 1:
            waveform = torch.mean(waveform, dim=1, keepdim=False)
        else:
            waveform = waveform.squeeze(1)
        waveform = waveform[0:1]
    elif waveform.ndim > 3:
        waveform = waveform.reshape(1, -1)

    # 1. Peak Normalization
    max_amp = torch.max(torch.abs(waveform))
    if max_amp > 1e-4:
        waveform = waveform / max_amp * 0.95

    with torch.no_grad():
        mel = _mel_transform(waveform)      # (1, 80, time)
        log_mel = _amp_to_db(mel)           # (1, 80, time)
        
        # 2. Instance Normalization (CMVN) along time frames
        mean = log_mel.mean(dim=-1, keepdim=True)
        std = log_mel.std(dim=-1, keepdim=True) + 1e-6
        norm_log_mel = (log_mel - mean) / std

    time_frames = norm_log_mel.shape[-1]
    return norm_log_mel.reshape(1, 1, 80, time_frames)

get_mel_spectrogram = extract_logmel

def extract_acoustic_features(waveform, sample_rate: int = 16000) -> np.ndarray:
    """Acoustic feature pooling for baseline models (LightGBM)."""
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()
    
    spec = extract_logmel(waveform, sample_rate=sample_rate).squeeze()  # (80, time)
    mean = torch.mean(spec, dim=-1)
    std = torch.std(spec, dim=-1)
    return torch.cat([mean, std], dim=0).cpu().numpy()