"""Per-image metrics and a fixed threshold/PCA baseline (no learned parameters)."""

import numpy as np
from .model import decode


def geometric_baseline(image: np.ndarray):
    intensity = image.mean(-1)
    mask = (intensity > 140) & ((image.max(-1).astype(float) - image.min(-1)) < 30)
    y, x = np.where(mask)
    if len(x) < 3:
        return mask, np.full((2, 2), 31.5, dtype=np.float32)
    points = np.stack([x, y], 1)
    center = points.mean(0)
    _, _, vectors = np.linalg.svd(points - center, full_matrices=False)
    direction = vectors[0]
    along = (points - center) @ direction
    a, b = np.quantile(along, [0.02, 0.98])
    # The synthetic material is brighter toward the tip; this assumption is explicit.
    if (
        intensity[y[along > b - 3], x[along > b - 3]].mean()
        < intensity[y[along < a + 3], x[along < a + 3]].mean()
    ):
        a, b = b, a
    return mask, np.stack([center + direction * b, center + direction * a]).astype(np.float32)


def measure(mask, points, sample):
    truth = sample.mask > 0.5
    union = (mask | truth).sum()
    error = np.linalg.norm(points - sample.keypoints, axis=-1)
    return {
        "iou": float((mask & truth).sum() / max(union, 1)),
        "error_px": float(error.mean()),
        "pck_5px": float((error <= 5).mean()),
        "visible_error_sum": float(error[sample.visible].sum()),
        "visible_count": int(sample.visible.sum()),
        "occluded_fraction": sample.occluded_fraction,
        "scene_id": sample.scene_id,
    }


def aggregate(rows):
    if not rows:
        return {"n": 0}
    visible = sum(r["visible_count"] for r in rows)
    return {
        "n": len(rows),
        "iou": float(np.mean([r["iou"] for r in rows])),
        "mean_keypoint_error_px": float(np.mean([r["error_px"] for r in rows])),
        "pck_5px": float(np.mean([r["pck_5px"] for r in rows])),
        "visible_keypoint_error_px": sum(r["visible_error_sum"] for r in rows) / max(visible, 1),
    }


def evaluate(dataset, predictions):
    rows = [
        measure(*decode(prediction), sample)
        for sample, prediction in zip(dataset.samples, predictions)
    ]
    return summarize(rows), rows


def summarize(rows):
    return {
        "overall": aggregate(rows),
        "occlusion_bins": {
            "none": aggregate([r for r in rows if r["occluded_fraction"] < 0.01]),
            "partial": aggregate([r for r in rows if 0.01 <= r["occluded_fraction"] < 0.3]),
            "heavy": aggregate([r for r in rows if r["occluded_fraction"] >= 0.3]),
        },
    }
