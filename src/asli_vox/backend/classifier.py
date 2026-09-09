# backend/classifier.py
import numpy as np
import joblib
import pathlib

MODEL_PATH = pathlib.Path(__file__).parent / "data" / "lgb_spoof_model.joblib"

# ---------------------------------------------------------------------------
# 1️⃣  Try loading a real LightGBM model trained on ASVspoof 2019 LA
# ---------------------------------------------------------------------------
try:
    clf = joblib.load(MODEL_PATH)
    MODEL_AVAILABLE = True
except Exception:          # no trained file yet – we fall back to a deterministic stub
    clf = None
    MODEL_AVAILABLE = False

# ---------------------------------------------------------------------------
# 2️⃣  Deterministic stub that mimics a realistic score distribution
# ---------------------------------------------------------------------------
def spoof_score_logmel(logmel: np.ndarray) -> float:
    """
    Very small “model” that simply returns a value based on the mean magnitude
    of the log‑mel.  This keeps the pipeline moving without any heavy ML.
    In a real app you would call `clf.predict_proba(logmel)` etc.
    """
    # logmel shape: (1, n_mels, time) -> mean over all entries
    mean_val = np.mean(logmel)
    # map roughly: louder → higher bonafide probability
    # we squash into [0,1] with a sigmoid‑like function
    score = 1.0 / (1.0 + np.exp(-(mean_val + 2.0)))   # bias -2 centers around 0.5
    return float(np.clip(score, 0.0, 1.0))

def classify(logmel: np.ndarray) -> dict:
    """
    Return dict: {"verdict": "bonafide"|"spoof", "confidence": float}
    """
    if MODEL_AVAILABLE and clf is not None:
        # real model path – uncomment & adjust when you have a model
        # proba = clf.predict_proba(logmel)[0]       # [[bonafide, spoof]]
        # confidence = float(proba[1])  # probability of spoof
        # verdict = "spoof" if confidence > 0.5 else "bonafide"
        pass  # placeholder – we use the stub below

    # ---- stub --------------------------------------------------------------
    score = spoof_score_logmel(logmel)
    # convert score to verdict; treat >0.55 as spoof for a slightly imbalanced demo
    if score > 0.55:
        verdict = "spoof"
        confidence = score
    else:
        verdict = "bonafide"
        confidence = 1.0 - score
    # ----------------------------------------------------------------------
    return {"verdict": verdict, "confidence": round(confidence, 4)}
# ---------------------------------------------------------------------------