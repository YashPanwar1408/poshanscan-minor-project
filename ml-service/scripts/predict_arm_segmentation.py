"""Run the trained upper-arm segmenter on a completely new image."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipeline.segment import configure_segmentation_model, predict_yolo_image  # noqa: E402

DEFAULT_WEIGHTS = ROOT / "app" / "models" / "arm_segmentation" / "best.pt"
DEFAULT_OUTPUT = ROOT / "results" / "arm_segmentation" / "predictions"


def predict_image(
    image_path: Path,
    weights: Path = DEFAULT_WEIGHTS,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    imgsz: int = 640,
    confidence: float = 0.25,
) -> dict[str, object]:
    """Predict, save the binary mask/overlay and return JSON-safe metadata."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not weights.is_file():
        raise FileNotFoundError(f"Trained YOLO checkpoint not found: {weights}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {image_path}")
    if image.shape[0] * image.shape[1] > 20_000_000:
        raise ValueError("Images must contain 20 megapixels or fewer.")
    if not configure_segmentation_model("yolo", weights):
        raise RuntimeError("Could not load YOLO segmentation weights. Install the optional segmentation dependencies and verify the checkpoint.")

    started = time.perf_counter()
    prediction = predict_yolo_image(
        image,
        imgsz=imgsz,
        confidence_threshold=confidence,
    )
    inference_seconds = time.perf_counter() - started

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    mask_path = output_dir / f"{stem}_mask.png"
    overlay_path = output_dir / f"{stem}_overlay.png"
    report_path = output_dir / f"{stem}_prediction.json"
    if not cv2.imwrite(str(mask_path), prediction["mask"]):
        raise OSError(f"Could not save predicted mask: {mask_path}")
    if not cv2.imwrite(str(overlay_path), prediction["visualization"]):
        raise OSError(f"Could not save prediction overlay: {overlay_path}")

    report: dict[str, object] = {
        "image": str(image_path.resolve()),
        "weights": str(weights.resolve()),
        "detected": prediction["detected"],
        "width": prediction["width"],
        "height": prediction["height"],
        "instances": prediction["instances"],
        "inference_seconds": inference_seconds,
        "mask_path": str(mask_path.resolve()),
        "visualization_path": str(overlay_path.resolve()),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to a new image not used for training.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args(argv)
    try:
        report = predict_image(
            args.image,
            args.weights,
            args.output_dir,
            imgsz=args.imgsz,
            confidence=args.confidence,
        )
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"Prediction failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    if not report["detected"]:
        print("No upper-arm mask was detected. Review the image and model output before using it.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
