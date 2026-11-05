"""Deterministic 2-D rendering, with scene IDs disjoint across dataset splits.

Keypoints are ordered [tip, base]. Segmentation targets the *visible* instrument;
keypoints target their full positions, including when covered by an occluder.
These stylized images are not a simulation of surgical imaging physics.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset
import torch

SIZE = 64
SCENES = {"train": tuple(range(8)), "val": (8, 9), "test": (10, 11), "shape_shift": (12, 13)}
SPLIT_CODES = {name: i for i, name in enumerate(SCENES)}


@dataclass
class Sample:
    image: np.ndarray
    mask: np.ndarray
    keypoints: np.ndarray
    visible: np.ndarray
    occluded_fraction: float
    scene_id: int


def render(index: int, split: str, seed: int = 17, augment: bool = True) -> Sample:
    if split not in SCENES or index < 0:
        raise ValueError("Unknown split or negative sample index")
    scene = SCENES[split][index % len(SCENES[split])]
    rng = np.random.default_rng(np.random.SeedSequence([seed, SPLIT_CODES[split], index]))
    style = np.random.default_rng(np.random.SeedSequence([seed, 1000, scene]))
    yy, xx = np.mgrid[:SIZE, :SIZE]
    color = style.uniform(25, 70, 3)
    gradient = (xx / SIZE - 0.5) * style.uniform(-25, 25)
    bg = color + gradient[..., None] + rng.normal(0, 4, (SIZE, SIZE, 3))
    bg = Image.fromarray(np.uint8(np.clip(bg, 0, 255)))
    draw_bg = ImageDraw.Draw(bg)
    for _ in range(3):
        x, y = rng.integers(0, SIZE, 2)
        r = int(rng.integers(2, 8))
        shade = tuple(int(v) for v in rng.integers(30, 100, 3))
        draw_bg.ellipse((x - r, y - r, x + r, y + r), fill=shade)

    center = rng.uniform(25, 39, 2)
    angle = rng.uniform(-np.pi, np.pi)
    direction = np.array([np.cos(angle), np.sin(angle)])
    side = np.array([-direction[1], direction[0]])
    length = rng.uniform(26, 37) if split != "shape_shift" else rng.uniform(36, 43)
    width = rng.uniform(2.0, 3.2) if split != "shape_shift" else rng.uniform(3.2, 4.5)
    tip, base = center + direction * length / 2, center - direction * length / 2
    full = Image.new("L", (SIZE, SIZE))
    full_draw = ImageDraw.Draw(full)
    polygon = [
        tuple(base + side * width),
        tuple(tip - direction * 4 + side * width),
        tuple(tip),
        tuple(tip - direction * 4 - side * width),
        tuple(base - side * width),
    ]
    full_draw.polygon(polygon, fill=255)
    r = width * 1.7
    full_draw.ellipse((base[0] - r, base[1] - r, base[0] + r, base[1] + r), fill=255)
    full_mask = np.asarray(full) > 0
    material = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    along = np.clip(((xx - base[0]) * direction[0] + (yy - base[1]) * direction[1]) / length, 0, 1)
    material[:] = (150 + 65 * along)[..., None]
    material += rng.normal(0, 3, material.shape)
    image = np.asarray(bg).copy()
    image[full_mask] = np.uint8(np.clip(material[full_mask], 0, 255))
    occluder = np.zeros((SIZE, SIZE), dtype=bool)
    # Index cycling guarantees coverage of 3 requested severities in each split.
    severity = index % 3 if augment else 0
    if severity:
        location = center + direction * rng.uniform(-length * 0.4, length * 0.4)
        radius = rng.uniform(3, 6) if severity == 1 else rng.uniform(7, 12)
        occluder = (np.abs(xx - location[0]) < radius) & (np.abs(yy - location[1]) < radius)
        oc_color = rng.integers(30, 125, 3)
        image[occluder] = oc_color
    mask = full_mask & ~occluder
    keypoints = np.stack([tip, base]).astype(np.float32)
    visible = np.array(
        [not occluder[int(round(y)), int(round(x))] for x, y in keypoints], dtype=bool
    )
    if augment:
        image = np.uint8(np.clip(image.astype(float) * rng.uniform(0.8, 1.2), 0, 255))
        if rng.random() < 0.2:
            image = np.array(Image.fromarray(image).filter(ImageFilter.GaussianBlur(0.5)))
    fraction = float((full_mask & occluder).sum() / max(full_mask.sum(), 1))
    return Sample(image, mask.astype(np.float32), keypoints, visible, fraction, scene)


def heatmaps(points: np.ndarray, sigma: float = 1.8) -> np.ndarray:
    yy, xx = np.mgrid[:SIZE, :SIZE]
    return np.exp(
        -((xx[None] - points[:, 0, None, None]) ** 2 + (yy[None] - points[:, 1, None, None]) ** 2)
        / (2 * sigma**2)
    ).astype(np.float32)


class ToolDataset(Dataset):
    """Cache generated tensors; train/val/test never share a scene ID or seed stream."""

    def __init__(self, count: int, split: str, seed: int = 17, augment: bool = True):
        if count < 1:
            raise ValueError("Dataset must contain at least one sample")
        self.samples = [render(i, split, seed, augment) for i in range(count)]
        self.x = (
            torch.from_numpy(np.stack([s.image.transpose(2, 0, 1) for s in self.samples])).float()
            / 255
        )
        self.y = torch.from_numpy(
            np.stack([np.concatenate([s.mask[None], heatmaps(s.keypoints)]) for s in self.samples])
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
