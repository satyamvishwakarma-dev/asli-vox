import os
import torch
import torchaudio
from torch.utils.data import Dataset
from src.asli_vox.features.spectrogram import get_mel_spectrogram

class ASVSpoof2019Dataset(Dataset):
    def __init__(self, protocol_file, audio_dir, chunk_seconds=3, sample_rate=16000):
        self.audio_dir = audio_dir
        self.chunk_samples = chunk_seconds * sample_rate
        self.sample_rate = sample_rate
        self.data = []
        
        # ASVspoof 2019 LA protocol format: SPEAKER_ID AUDIO_ID - SYSTEM_ID LABEL(bonafide/spoof)
        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    audio_id = parts[1]
                    label = 1 if parts[4] == 'spoof' else 0
                    self.data.append((audio_id, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_id, label = self.data[idx]
        file_path = os.path.join(self.audio_dir, f"{audio_id}.flac")
        
        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
        # Force exact chunk size via zero-pad or truncate
        if waveform.shape[1] > self.chunk_samples:
            waveform = waveform[:, :self.chunk_samples]
        else:
            pad_amount = self.chunk_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            
        mel_spec = get_mel_spectrogram(waveform[0], self.sample_rate)
        return mel_spec, torch.tensor(label, dtype=torch.long)