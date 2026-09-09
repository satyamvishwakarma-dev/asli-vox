# backend/gradcam.py
import numpy as np
import base64
from io import BytesIO
import matplotlib

matplotlib.use("Agg")  # head‑less backend
import matplotlib.pyplot as plt


def gradcam_heatmap(logmel: np.ndarray, pred_class: int = 1) -> str:
    """
    Produce a pseudo‑Grad‑CAM heatmap from the log‑mel tensor.
    The function pretends the network's last conv layer is the mel‑filter bank,
    computes “gradients” as the absolute mean across time, and colours the
    result with a jet colormap.  Returns a base64‑encoded PNG data URI.

    In a real model you would:
      1. Register forward/backward hooks on the last conv layer.
      2. Compute pooled gradients of the target class score.
      3. Multiply feature maps by pooled gradients.
      4. ReLU, upsample, overlay on original spectrogram.
    Here we fake step 3 with a simple magnitude‑based weighting.
    """

    def resize_nearest(image: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize an array using NumPy, avoiding an OpenCV dependency."""
        y = np.linspace(0, image.shape[0] - 1, height).astype(int)
        x = np.linspace(0, image.shape[1] - 1, width).astype(int)
        return image[np.ix_(y, x)]

    # logmel shape: (1, n_mels, time) – squeeze batch & channel dims
    spec = logmel[0]  # (n_mels, time)

    # ---- fake gradient computation -----------------------------------------
    # “gradient” = how much each filter response contributes to the output.
    # We'll use absolute mean across time for each mel bin.
    grads = np.abs(spec).mean(axis=1)  # (n_mels,)
    # Normalise to [0,1]
    grads = (grads - grads.min()) / (grads.max() - grads.min() + 1e-7)

    # ---- colourise with jet colormap ---------------------------------------
    # matplotlib's jet maps 0→blue, 1→red
    cmap = plt.get_cmap("jet")
    # Expand grads to (n_mels, 1) for broadcasting
    colored = cmap(grads)[:, :3]  # drop alpha, shape (n_mels, 3) RGB
    # Upsample to a fixed image size (e.g., 256 × time_pixels)
    target_h, target_w = 256, spec.shape[1] * 4  # simple scaling
    colored_hw = resize_nearest(colored[:, np.newaxis, :], target_w, target_h)
    # Convert float [0,1] to uint8 [0,255]
    colored_u8 = (colored_hw * 255).astype(np.uint8)

    # ---- overlay on the original spectrogram --------------------------------
    # Log‑mel in dB is negative; we'll shift & normalize for display
    spec_db = spec - spec.min()
    spec_db = spec_db / (spec_db.max() + 1e-7)
    spec_u8 = (spec_db * 255).astype(np.uint8)
    spec_u8 = resize_nearest(spec_u8, target_w, target_h)

    # Heatmap overlay: add weighted heatmap to gray spectrogram
    gray_rgb = np.repeat(spec_u8[..., np.newaxis], 3, axis=2)
    overlay = (gray_rgb * 0.6 + colored_u8 * 0.4).astype(np.uint8)

    # ---- save to PNG in memory & base64‑encode -------------------------------
    buf = BytesIO()
    plt.imsave(buf, overlay, format="png")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"
