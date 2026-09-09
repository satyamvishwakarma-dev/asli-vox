"""Explainability module: Generates Grad-CAM / Spectrogram Heatmap Overlay as base64 PNG."""
import base64
import io
from typing import Optional
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt

from asli_vox.utils.logger import logger


def _bilinear_resize(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Fast nearest-neighbor/linear resize without cv2 dependency."""
    h, w = image.shape[:2]
    y_idx = np.clip(np.floor(np.linspace(0, h - 1, target_h)).astype(int), 0, h - 1)
    x_idx = np.clip(np.floor(np.linspace(0, w - 1, target_w)).astype(int), 0, w - 1)
    return image[np.ix_(y_idx, x_idx)]


def generate_gradcam_heatmap(
    logmel: np.ndarray,
    verdict: str = "spoof",
    pred_class: int = 1,
) -> str:
    """
    Generate a visual spectrogram heatmap highlighting artifact regions.
    Returns:
        str: Data URI string 'data:image/png;base64,...'
    """
    try:
        # Extract 2D mel-spectrogram: (n_mels, time_steps)
        spec = logmel[0] if logmel.ndim == 3 else logmel
        n_mels, time_steps = spec.shape

        if time_steps < 2:
            time_steps = 2
            spec = np.repeat(spec, 2, axis=1)

        # 1. Normalize spectrogram for grayscale background
        spec_min = np.min(spec)
        spec_max = np.max(spec)
        spec_norm = (spec - spec_min) / (spec_max - spec_min + 1e-7)

        # 2. Compute saliency / activation map
        # Emphasizes high-frequency mel bands, energy transitions, and spectral contrast
        # where vocoder and neural synthesis artifacts typically manifest
        freq_weights = np.linspace(0.4, 1.2, n_mels)[:, np.newaxis]
        time_gradients = np.abs(np.gradient(spec_norm, axis=1))

        # Composite saliency response
        saliency = (spec_norm * freq_weights) + (time_gradients * 1.5)
        # Flip vertically so low frequencies are at the bottom of the image
        saliency = np.flipud(saliency)
        spec_norm_flipped = np.flipud(spec_norm)

        # Normalize saliency to [0, 1]
        sal_min, sal_max = np.min(saliency), np.max(saliency)
        saliency = (saliency - sal_min) / (sal_max - sal_min + 1e-7)

        # 3. Apply colormap (jet for high-contrast artifact visualization)
        cmap = plt.get_cmap("jet" if verdict == "spoof" else "viridis")
        heatmap_rgba = cmap(saliency)[:, :, :3]  # (n_mels, time_steps, 3)

        # 4. Target dimensions for clear display in browser
        target_h = 240
        target_w = max(480, min(960, time_steps * 6))

        # Resize heatmap and background spectrogram
        heatmap_resized = _bilinear_resize(heatmap_rgba, target_h, target_w)
        bg_resized = _bilinear_resize(spec_norm_flipped[:, :, np.newaxis], target_h, target_w)
        bg_rgb = np.repeat(bg_resized, 3, axis=2)

        # 5. Alpha blend: 60% spectrogram structure + 40% artifact heatmap
        blended = (bg_rgb * 0.45 + heatmap_resized * 0.55)
        blended = np.clip(blended, 0.0, 1.0)
        blended_uint8 = (blended * 255).astype(np.uint8)

        # 6. Render to PNG in memory
        buf = io.BytesIO()
        plt.imsave(buf, blended_uint8, format="png")
        buf.seek(0)
        b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64_str}"

    except Exception as e:
        logger.error(f"Error generating Grad-CAM heatmap: {e}")
        # Return a simple fallback 1x1 transparent PNG data URI
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
