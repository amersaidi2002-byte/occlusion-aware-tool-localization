"""Package executed runs, regenerate README measurements, and show worst failures."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

from toolloc.model import decode
from toolloc.synthetic import render


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--main", type=Path, default=Path("runs/main"))
    p.add_argument("--ablation", type=Path, default=Path("runs/no_augmentation"))
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    results, artifacts = root / "results", root / "artifacts/main"
    results.mkdir(exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    main_report = json.loads((args.main / "report.json").read_text())
    ablation = json.loads((args.ablation / "report.json").read_text())
    shutil.copy2(args.main / "report.json", results / "main_report.json")
    shutil.copy2(args.ablation / "report.json", results / "no_augmentation_report.json")
    for filename in ["best.pt", "model.onnx", "example_input.npy", "example_logits.npy"]:
        shutil.copy2(args.main / filename, artifacts / filename)
    Image.open(args.main / "preview.png").quantize(colors=128).save(results / "preview.png")
    source = render(0, "test", main_report["config"]["seed"])
    np.save(artifacts / "input_rgb.npy", source.image)
    Image.fromarray(source.image).save(results / "example_input.png")
    rows = []
    for split in ["test", "shape_shift"]:
        for label, report, key in [
            ("threshold/PCA", main_report, "threshold_pca"),
            ("custom network", main_report, "model"),
            ("no augmentation", ablation, "model"),
        ]:
            metrics = report["evaluation"][split][key]["overall"]
            rows.append({"split": split, "method": label, **metrics})
        measurements = json.loads((args.main / f"{split}_per_image.json").read_text())
        with (results / f"{split}_per_image.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["index", *measurements[0]])
            writer.writeheader()
            writer.writerows({"index": i, **row} for i, row in enumerate(measurements))
    with (results / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = [
        "| Split | Method | Mean IoU | PCK@5 px | Mean keypoint error |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        table.append(
            f"| {row['split']} | {row['method']} | {row['iou']:.3f} | "
            f"{row['pck_5px']:.1%} | {row['mean_keypoint_error_px']:.2f} px |"
        )
    table += [
        "",
        "Each split contains 480 synthetic images. One training seed, 16 epochs per trial;",
        "checkpoint selection uses validation loss only. The network improves segmentation",
        "and PCK on the standard split, but its mean keypoint error is slightly worse than",
        "the simple baseline because some predictions fail badly. See the failure gallery.",
        "",
        f"PyTorch/ONNX validation parity: max absolute logit error "
        f"`{main_report['onnx_parity_max_abs_error']:.3g}` over 16 images.",
        "",
        "Full metrics, occlusion bins and confidence coverage: [main report](results/main_report.json).",
        "The confidence threshold is validation-derived and is not a calibrated safety score.",
    ]
    readme = root / "README.md"
    before, tail = readme.read_text().split("<!-- RESULTS_START -->")
    _, after = tail.split("<!-- RESULTS_END -->")
    readme.write_text(
        before + "<!-- RESULTS_START -->\n" + "\n".join(table) + "\n<!-- RESULTS_END -->" + after
    )
    errors = json.loads((args.main / "test_per_image.json").read_text())
    worst = np.argsort([r["error_px"] for r in errors])[-8:][::-1]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(artifacts / "model.onnx"), options, providers=["CPUExecutionProvider"]
    )
    canvas = Image.new("RGB", (4 * 128, 2 * 160), "#17232c")
    for position, index in enumerate(worst):
        sample = render(int(index), "test", main_report["config"]["seed"])
        image = Image.fromarray(sample.image)
        tensor = np.ascontiguousarray(sample.image.transpose(2, 0, 1)[None], dtype=np.float32) / 255
        _, points = decode(session.run(None, {"image": tensor})[0][0])
        draw = ImageDraw.Draw(image)
        for truth, pred in zip(sample.keypoints, points):
            x, y = truth
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline="lime")
            x, y = pred
            draw.line((x - 2, y, x + 2, y), fill="magenta")
            draw.line((x, y - 2, x, y + 2), fill="magenta")
        x, y = (position % 4) * 128, (position // 4) * 160
        canvas.paste(image.resize((128, 128)), (x, y))
        ImageDraw.Draw(canvas).text(
            (x + 2, y + 130), f"#{index} error {errors[index]['error_px']:.1f}px", fill="white"
        )
    canvas.quantize(colors=128).save(results / "worst_failures.png")
    paths = [p for folder in ["src", "tests"] for p in sorted((root / folder).rglob("*.py"))]
    provenance = {
        "source": "personal synthetic experiment, November 2026; AI coding assistance",
        "artifact_sha256": {
            str(p.relative_to(root)): digest(p) for p in sorted(artifacts.iterdir())
        },
        "source_sha256": {str(p.relative_to(root)): digest(p) for p in paths},
        "limitations": "No real-robot or clinical validation; see docs/EXPERIMENT_DESIGN.md",
    }
    (results / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
