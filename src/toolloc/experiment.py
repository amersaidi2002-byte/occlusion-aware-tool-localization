"""One command to train, validate, evaluate, export and generate a visual report."""

import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

from .synthetic import SCENES, ToolDataset
from .model import ToolNet, decode, objective
from .evaluation import evaluate, geometric_baseline, measure, summarize


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit(args, output, train, validation):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    model = ToolNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    history, best = [], float("inf")
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for image, target in loader:
            optimizer.zero_grad()
            loss = objective(model(image), target, args.heatmap_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            values = [
                (len(x), float(objective(model(x), y, args.heatmap_weight)))
                for x, y in DataLoader(validation, batch_size=args.batch_size)
            ]
        val_loss = sum(n * loss for n, loss in values) / len(validation)
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": val_loss}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "config": vars(args),
                    "scene_ids": SCENES,
                    "model": "ToolNet(width=12)",
                    "data": "procedural synthetic v1",
                },
                output / "best.pt",
            )
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, history, checkpoint["epoch"]


def predictions(model, dataset):
    with torch.inference_mode():
        return np.concatenate([model(x).numpy() for x, _ in DataLoader(dataset, batch_size=32)])


def confidence(logits):
    """Minimum spatial-softmax peak across the two keypoints; not a probability of safety."""
    maps = logits[:, 1:].reshape(len(logits), 2, -1)
    values = np.exp(maps - maps.max(-1, keepdims=True))
    return (values / values.sum(-1, keepdims=True)).max(-1).min(-1)


def preview(dataset, outputs, path):
    canvas = Image.new("RGB", (64 * 6, 100 * 3), "#17232c")
    for i in range(18):
        index = i * len(dataset) // 18
        sample = dataset.samples[index]
        tile = Image.fromarray(sample.image).convert("RGB")
        draw = ImageDraw.Draw(tile)
        _, points = decode(outputs[index])
        for truth, pred in zip(sample.keypoints, points):
            x, y = truth
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline="#33ff66")
            x, y = pred
            draw.line((x - 2, y, x + 2, y), fill="#ff44ff")
            draw.line((x, y - 2, x, y + 2), fill="#ff44ff")
        x, y = i % 6 * 64, i // 6 * 100
        canvas.paste(tile, (x, y))
        ImageDraw.Draw(canvas).text(
            (x + 2, y + 66), f"occ {sample.occluded_fraction:.0%}", fill="white"
        )
    canvas.resize((768, 600)).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/main"))
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--train", type=int, default=1200)
    parser.add_argument("--val", type=int, default=240)
    parser.add_argument("--test", type=int, default=480)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--heatmap-weight", type=float, default=1.0)
    parser.add_argument("--no-augmentation", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.train, args.val, args.test, args.threads, args.batch_size) < 1:
        parser.error("Counts, epochs and threads must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    # Paths in saved config must be JSON/pickle portable.
    output = args.output
    args.output = str(output)
    torch.set_num_threads(args.threads)
    start = time.perf_counter()
    train = ToolDataset(args.train, "train", args.seed, not args.no_augmentation)
    val = ToolDataset(args.val, "val", args.seed)
    model, history, selected_epoch = fit(args, output, train, val)
    # Fix a coverage threshold on validation only, before inspecting held-out metrics.
    val_predictions = predictions(model, val)
    confidence_threshold = float(np.quantile(confidence(val_predictions), 0.2))
    results = {}
    for split in ["test", "shape_shift"]:
        data = ToolDataset(args.test, split, args.seed)
        pred = predictions(model, data)
        learned, rows = evaluate(data, pred)
        keep = confidence(pred) >= confidence_threshold
        gated = [row for row, accepted in zip(rows, keep) if accepted]
        learned["confidence_gating"] = {
            "threshold_source": "20th percentile of validation confidence",
            "threshold": confidence_threshold,
            "coverage": float(keep.mean()),
            "retained": summarize(gated)["overall"],
            "warning": "uncalibrated confidence heuristic; not a safety guarantee",
        }
        baseline = [measure(*geometric_baseline(s.image), s) for s in data.samples]
        results[split] = {"model": learned, "threshold_pca": summarize(baseline)}
        (output / f"{split}_per_image.json").write_text(json.dumps(rows, indent=2) + "\n")
        if split == "test":
            preview(data, pred, output / "preview.png")
            np.save(output / "example_input.npy", data.x[0:1].numpy())
            np.save(output / "example_logits.npy", pred[0:1])
    torch.onnx.export(
        model,
        train.x[:1],
        str(output / "model.onnx"),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(output / "model.onnx"))
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(
        str(output / "model.onnx"), options, providers=["CPUExecutionProvider"]
    )
    # Export parity uses validation samples, not training examples.
    torch_out = val_predictions[:16]
    ort_out = session.run(None, {"image": val.x[:16].numpy()})[0]
    np.testing.assert_allclose(torch_out, ort_out, atol=1e-4, rtol=1e-4)
    report = {
        "data": "synthetic only",
        "config": vars(args),
        "split_scene_ids": SCENES,
        "selected_epoch": selected_epoch,
        "parameters": sum(p.numel() for p in model.parameters()),
        "history": history,
        "evaluation": results,
        "onnx_parity_max_abs_error": float(np.max(np.abs(torch_out - ort_out))),
        "checkpoint_sha256": sha256(output / "best.pt"),
        "onnx_sha256": sha256(output / "model.onnx"),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "onnxruntime": ort.__version__,
            "platform": platform.platform(),
            "threads": args.threads,
        },
        "elapsed_seconds": time.perf_counter() - start,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "results": results,
                "parity_max_abs_error": report["onnx_parity_max_abs_error"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
