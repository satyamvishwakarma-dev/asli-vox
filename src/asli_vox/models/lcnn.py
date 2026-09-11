import torch
import torch.nn as nn

class MFM(nn.Module):
    """Max-Feature-Map activation: halves channel dimension by taking max of two halves."""
    def forward(self, x):
        half = x.size(1) // 2
        if x.ndim == 2:
            return torch.max(x[:, :half], x[:, half:])
        return torch.max(x[:, :half, :, :], x[:, half:, :, :])

class LCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Input: (Batch, 1, 80, Time)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), MFM(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 48, kernel_size=3, padding=1), MFM(), nn.MaxPool2d(2, 2),
            nn.Conv2d(24, 64, kernel_size=3, padding=1), MFM(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Target layer for Grad-CAM
        )
        self.mfm_final = MFM()
        # Adaptive pool handles variable-length audio chunks cleanly
        self.pool = nn.AdaptiveAvgPool2d((5, 5))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(32 * 5 * 5, 64), MFM(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.mfm_final(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x