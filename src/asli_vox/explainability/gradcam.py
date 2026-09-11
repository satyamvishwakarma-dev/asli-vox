import cv2
import numpy as np
import torch

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._hook_forward)
        self.target_layer.register_full_backward_hook(self._hook_backward)

    def _hook_forward(self, module, input, output):
        self.activations = output

    def _hook_backward(self, module, grad_input, grad_output):
        if grad_output and len(grad_output) > 0 and grad_output[0] is not None:
            self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        # Expected input shape: (1, 1, H, W)
        target_h, target_w = input_tensor.shape[-2], input_tensor.shape[-1]
        fallback_map = np.zeros((target_h, target_w), dtype=np.float32)

        self.model.zero_grad()
        output = self.model(input_tensor)

        if output.ndim == 1:
            output = output.unsqueeze(0)

        # Select target class logit
        idx = class_idx if output.shape[1] > class_idx else 0
        target_score = output[:, idx]
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return fallback_map

        grads = self.gradients
        acts = self.activations

        # If somehow flattened, return fallback
        if acts.ndim < 3 or grads.ndim < 3:
            return fallback_map

        # Standardize to 4D: (Batch, Channels, Height, Width)
        if acts.ndim == 3:
            acts = acts.unsqueeze(0)
        if grads.ndim == 3:
            grads = grads.unsqueeze(0)

        # Global average pool the gradients over H and W: shape (B, C, 1, 1)
        weights = torch.mean(grads, dim=[-2, -1], keepdim=True)

        # Weighted combination: sum over channel dimension -> shape (B, 1, H, W)
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam_np = cam.squeeze().detach().cpu().numpy()

        # Check if cam_np is 2D
        if cam_np.ndim != 2:
            return fallback_map

        max_val = np.max(cam_np)
        if max_val > 1e-8:
            cam_np = cam_np / max_val
        else:
            return fallback_map

        # Resize to original input spectrogram width and height
        cam_resized = cv2.resize(cam_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return cam_resized

def generate_gradcam_heatmap(model, input_tensor: torch.Tensor, target_layer=None, class_idx: int = 1) -> np.ndarray:
    if target_layer is None:
        target_layer = model.features[-1]
    cam = GradCAM(model, target_layer)
    return cam.generate_heatmap(input_tensor, class_idx=class_idx)