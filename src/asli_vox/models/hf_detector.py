# src/asli_vox/models/hf_detector.py
import torch
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_ID = "MelodyMachine/Deepfake-audio-detection-V2"

class PretrainedAudioSpoofDetector:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        print(f"[AsliVox] Loading pretrained anti-spoofing checkpoint: {MODEL_ID}...")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
        self.model = AutoModelForAudioClassification.from_pretrained(MODEL_ID).to(self.device)
        self.model.eval()
        
        # Determine index of fake/spoof vs real/bonafide
        self.id2label = self.model.config.id2label
        print(f"[AsliVox] Model labels: {self.id2label}")
        
        self.fake_idx = 1
        for idx, label in self.id2label.items():
            if "fake" in label.lower() or "spoof" in label.lower():
                self.fake_idx = int(idx)
                break
        self.real_idx = 0 if self.fake_idx == 1 else 1

    def predict(self, audio_array: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Accepts 1D float32 numpy array (16kHz).
        Returns dict with probabilities and discrete verdict.
        """
        if len(audio_array) == 0:
            return {"verdict": "silent", "confidence": 0.0, "fake_prob": 0.0, "real_prob": 0.0}

        # Normalize amplitude safely
        max_val = np.max(np.abs(audio_array))
        if max_val > 1e-4:
            norm_audio = audio_array / max_val * 0.95
        else:
            norm_audio = audio_array

        inputs = self.feature_extractor(
            norm_audio, 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        fake_prob = float(probs[self.fake_idx])
        real_prob = float(probs[self.real_idx])
        verdict = "spoof" if fake_prob > 0.50 else "bonafide"

        return {
            "verdict": verdict,
            "confidence": round(fake_prob if verdict == "spoof" else real_prob, 4),
            "fake_prob": round(fake_prob, 4),
            "real_prob": round(real_prob, 4),
            "raw_logits": [round(float(x), 3) for x in logits[0].tolist()]
        }

# Global singleton
detector = PretrainedAudioSpoofDetector()