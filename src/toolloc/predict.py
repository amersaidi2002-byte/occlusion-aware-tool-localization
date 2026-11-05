"""Run a trained ONNX model on one 64x64 RGB image and save auditable outputs."""

import argparse
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
from .model import decode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("runs/prediction"))
    args = p.parse_args()
    image = Image.open(args.image).convert("RGB")
    if image.size != (64, 64):
        p.error("input must be 64x64; coordinate-changing resizing must be explicit")
    args.output.mkdir(parents=True, exist_ok=True)
    x = np.ascontiguousarray(np.asarray(image).transpose(2, 0, 1)[None], dtype=np.float32) / 255
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(args.model), options, providers=["CPUExecutionProvider"])
    logits = session.run(None, {session.get_inputs()[0].name: x})[0][0]
    mask, points = decode(logits)
    Image.fromarray(np.uint8(mask) * 255).save(args.output / "mask.png")
    draw = ImageDraw.Draw(image)
    for point, color in zip(points, ["magenta", "cyan"]):
        x, y = point
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=color)
    image.resize((384, 384)).save(args.output / "overlay.png")
    (args.output / "prediction.json").write_text(
        json.dumps(
            {
                "keypoints_tip_base_xy": points.tolist(),
                "source": "synthetic research model; no real-world accuracy claim",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
