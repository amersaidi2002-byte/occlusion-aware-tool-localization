"""Small encoder-decoder initialized randomly: 1 mask + 2 ordered keypoint maps."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def block(cin: int, cout: int):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.ReLU(),
    )


class ToolNet(nn.Module):
    def __init__(self, width: int = 12):
        super().__init__()
        self.enc1 = block(3, width)
        self.enc2 = block(width, width * 2)
        self.middle = block(width * 2, width * 4)
        self.dec2 = block(width * 6, width * 2)
        self.dec1 = block(width * 3, width)
        self.head = nn.Conv2d(width, 3, 1)

    def forward(self, image):
        e1 = self.enc1(image)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        mid = self.middle(F.max_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([F.interpolate(mid, scale_factor=2, mode="nearest"), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode="nearest"), e1], 1))
        return self.head(d1)


def objective(logits, targets, heatmap_weight: float = 1.0):
    seg = F.binary_cross_entropy_with_logits(logits[:, :1], targets[:, :1])
    probability = logits[:, :1].sigmoid()
    dice = 1 - (2 * (probability * targets[:, :1]).sum((1, 2, 3)) + 1) / (
        probability.sum((1, 2, 3)) + targets[:, :1].sum((1, 2, 3)) + 1
    )
    # Spatial cross entropy avoids the all-zero solution of sparse heatmap MSE.
    maps = targets[:, 1:].flatten(2)
    maps = maps / maps.sum(-1, keepdim=True).clamp_min(1e-8)
    localization = -(maps * F.log_softmax(logits[:, 1:].flatten(2), dim=-1)).sum(-1).mean()
    return seg + dice.mean() + heatmap_weight * localization


def decode(logits: np.ndarray):
    """Decode CHW logits without dependencies on PyTorch or ONNX Runtime."""
    if logits.shape != (3, 64, 64) or not np.isfinite(logits).all():
        raise ValueError("Expected finite [3,64,64] logits")
    maps = logits[1:]
    flat = maps.reshape(2, -1)
    index = flat.argmax(-1)
    coordinates = np.stack([index % 64, index // 64], axis=-1).astype(np.float32)
    return logits[0] > 0, coordinates
