import numpy as np
import pytest
import torch

from toolloc.synthetic import SCENES, ToolDataset, heatmaps, render
from toolloc.model import ToolNet, decode, objective
from toolloc.evaluation import aggregate, geometric_baseline, measure


def test_scene_ids_are_disjoint():
    splits = list(SCENES.values())
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            assert not set(left) & set(right)


def test_render_reproducible_and_labels_valid():
    for split in SCENES:
        for index in range(9):
            a, b = render(index, split), render(index, split)
            np.testing.assert_array_equal(a.image, b.image)
            assert a.image.shape == (64, 64, 3)
            assert a.mask.shape == (64, 64)
            assert 0 <= a.occluded_fraction <= 1
            assert ((a.keypoints >= 0) & (a.keypoints < 64)).all()


def test_heatmap_coordinate_convention():
    points = np.array([[13, 27], [42, 9]], dtype=np.float32)
    maps = heatmaps(points)
    fake = np.concatenate([np.ones((1, 64, 64)), maps])
    _, recovered = decode(fake)
    np.testing.assert_array_equal(points, recovered)


def test_finite_loss_and_gradient():
    torch.set_num_threads(1)
    data = ToolDataset(2, "train")
    model = ToolNet()
    output = model(data.x)
    assert output.shape == (2, 3, 64, 64)
    loss = objective(output, data.y)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_perfect_prediction_metrics():
    s = render(2, "test")
    m = measure(s.mask > 0.5, s.keypoints, s)
    assert m["iou"] == 1 and m["pck_5px"] == 1 and m["error_px"] == 0
    assert aggregate([m])["n"] == 1


def test_empty_geometric_input_is_defined():
    mask, points = geometric_baseline(np.zeros((64, 64, 3), np.uint8))
    assert not mask.any() and np.isfinite(points).all()


def test_reject_nonfinite_logits():
    with pytest.raises(ValueError):
        decode(np.full((3, 64, 64), np.nan))


def test_reject_bad_split_and_count():
    with pytest.raises(ValueError):
        render(0, "invalid")
    with pytest.raises(ValueError):
        ToolDataset(0, "train")
