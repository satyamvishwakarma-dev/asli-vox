import time
import numpy as np
import torch
from huggingface_hub import snapshot_download

MODEL_REPO = "SpeechAntiSpoofingBenchmarks/AASIST-L"
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 64600  # ~4.04s, matches the model's expected input length
N_WARMUP = 3
N_TRIALS = 20


def main():
    print(f"Downloading {MODEL_REPO} from Hugging Face...")
    local_dir = snapshot_download(repo_id=MODEL_REPO)
    print(f"Downloaded to: {local_dir}")

    import sys
    sys.path.insert(0, local_dir)
    from importlib import import_module

    try:
        AASIST_L = import_module("aasist_l").AASIST_L
    except ModuleNotFoundError as e:
        print(f"Module not found: {e}")
        print(f"Please ensure that the 'speech_spoof_bench' module is included in the repository.")
        return

    print("Loading model...")
    torch.set_num_threads(4)  # match expected demo laptop core count
    model = AASIST_L()
    model.load()

    # Synthetic audio for timing purposes -- real waveform values don't affect latency
    dummy_wav = np.random.uniform(-1, 1, WINDOW_SAMPLES).astype(np.float32)

    print(f"\nWarming up ({N_WARMUP} runs)...")
    for _ in range(N_WARMUP):
        _ = model.score_batch([dummy_wav], [SAMPLE_RATE])

    print(f"Timing {N_TRIALS} inference runs (single-sample, batch=1)...")
    latencies = []
    for _ in range(N_TRIALS):
        start = time.perf_counter()
        _ = model.score_batch([dummy_wav], [SAMPLE_RATE])
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    latencies = np.array(latencies)
    print(f"\n--- AASIST-L CPU Latency (batch=1, {WINDOW_SAMPLES/SAMPLE_RATE:.2f}s window) ---")
    print(f"Mean:   {latencies.mean():.1f} ms")
    print(f"Median: {np.median(latencies):.1f} ms")
    print(f"P95:    {np.percentile(latencies, 95):.1f} ms")
    print(f"Min/Max: {latencies.min():.1f} / {latencies.max():.1f} ms")

    print("\n--- Verdict ---")
    if latencies.mean() < 100:
        print("PASS: comfortably real-time. Proceed with AASIST-L, fine-tune if time allows.")
    elif latencies.mean() < 200:
        print("MARGINAL: usable but tight, especially with WebSocket overhead + preprocessing on top.")
        print("Consider: shorter audio windows, or keep LightGBM as a guaranteed fallback.")
    else:
        print("FAIL: too slow for real-time CPU demo. Fall back to LightGBM primary now.")


if __name__ == "__main__":
    main()