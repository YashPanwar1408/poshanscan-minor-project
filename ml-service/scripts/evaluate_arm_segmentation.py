"""Evaluate a trained arm-segmentation checkpoint on the reserved test split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_annotations import validate_annotations  # noqa: E402
from scripts.yolo_dataset import build_yolo_layout  # noqa: E402

DEFAULT_WEIGHTS = ROOT / "app" / "models" / "arm_segmentation" / "best.pt"
DEFAULT_OUTPUT = ROOT / "results" / "arm_segmentation" / "evaluation"


def binary_mask_metrics(predicted: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    pred = np.asarray(predicted) > 0
    truth = np.asarray(ground_truth) > 0
    if pred.shape != truth.shape:
        raise ValueError("Predicted and ground-truth masks must have the same shape.")
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    union = tp + fp + fn
    return {
        "iou": float(tp / union) if union else 1.0,
        "dice": float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 1.0,
        "pixel_precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "pixel_recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "true_positive_pixels": float(tp),
        "false_positive_pixels": float(fp),
        "false_negative_pixels": float(fn),
    }


def ground_truth_mask(label_path: Path, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
        if not fields:
            continue
        coordinates = np.asarray([float(value) for value in fields[1:]], dtype=np.float64).reshape(-1, 2)
        points = np.round(coordinates * np.array([width - 1, height - 1])).astype(np.int32)
        cv2.fillPoly(mask, [points], 255)
    return mask


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _finite_metric(value: Any) -> Optional[float]:
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return None
    return metric if np.isfinite(metric) else None


def predicted_mask(result: Any, width: int, height: int) -> np.ndarray:
    output = np.zeros((height, width), dtype=np.uint8)
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is None or boxes is None or getattr(masks, "data", None) is None:
        return output
    data = _to_numpy(masks.data)
    classes = _to_numpy(getattr(boxes, "cls", np.zeros((len(data),), dtype=np.int32))).reshape(-1)
    for index, mask_data in enumerate(data):
        if index >= len(classes) or int(classes[index]) != 0:
            continue
        resized = cv2.resize(mask_data.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        output[resized >= 0.5] = 255
    return output


def _comparison_image(image: np.ndarray, truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    def panel(base: np.ndarray, mask: Optional[np.ndarray], title: str, color: tuple[int, int, int]) -> np.ndarray:
        canvas = base.copy()
        if mask is not None:
            overlay = np.zeros_like(canvas)
            overlay[mask > 0] = color
            canvas = cv2.addWeighted(canvas, 0.72, overlay, 0.28, 0.0)
        scale = min(420 / canvas.shape[1], 420 / canvas.shape[0], 1.0)
        canvas = cv2.resize(canvas, (max(1, int(canvas.shape[1] * scale)), max(1, int(canvas.shape[0] * scale))))
        cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return canvas

    panels = [
        panel(image, None, "Original", (255, 255, 255)),
        panel(image, truth, "Ground truth", (0, 220, 0)),
        panel(image, pred, "Prediction", (0, 0, 255)),
    ]
    height = max(panel_image.shape[0] for panel_image in panels)
    aligned = [cv2.copyMakeBorder(p, 0, height - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24)) for p in panels]
    return cv2.hconcat(aligned)


def evaluate_test_split(weights: Path, output_dir: Path, imgsz: int = 416) -> dict[str, Any]:
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = output_dir / "visual_comparisons"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    test_entries = [
        line.strip()
        for line in (ROOT / "dataset" / "splits" / "test.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    model = YOLO(str(weights))
    with TemporaryDirectory(prefix="poshanscan-yolo-eval-") as temp_name:
        data_yaml = build_yolo_layout(Path(temp_name))
        val_results = model.val(
            data=str(data_yaml),
            split="test",
            imgsz=imgsz,
            plots=True,
            project=str(output_dir / "ultralytics"),
            name="test",
            exist_ok=True,
        )
        seg_metrics = getattr(val_results, "seg", None)
        map_metrics = {
            "mask_precision": _finite_metric(getattr(seg_metrics, "mp", None)),
            "mask_recall": _finite_metric(getattr(seg_metrics, "mr", None)),
            "map50": _finite_metric(getattr(seg_metrics, "map50", None)),
            "map50_95": _finite_metric(getattr(seg_metrics, "map", None)),
        }
        per_image: list[dict[str, Any]] = []
        all_tp = all_fp = all_fn = 0
        inference_seconds: list[float] = []
        failures: list[dict[str, Any]] = []
        label_dir = ROOT / "dataset" / "annotations" / "labels"
        for entry in test_entries:
            image_path = ROOT / "dataset" / Path(entry)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                failures.append({"image": image_path.name, "reason": "image_decode_failed"})
                continue
            height, width = image.shape[:2]
            truth = ground_truth_mask(label_dir / f"{image_path.stem}.txt", width, height)
            started = time.perf_counter()
            prediction_results = model.predict(source=image, imgsz=imgsz, verbose=False)
            elapsed = time.perf_counter() - started
            inference_seconds.append(elapsed)
            pred = predicted_mask(prediction_results[0], width, height) if prediction_results else np.zeros_like(truth)
            metrics = binary_mask_metrics(pred, truth)
            all_tp += int(metrics["true_positive_pixels"])
            all_fp += int(metrics["false_positive_pixels"])
            all_fn += int(metrics["false_negative_pixels"])
            per_image.append({"image": image_path.name, "inference_seconds": elapsed, **metrics})
            if metrics["iou"] < 0.5 or not np.any(pred):
                failures.append({"image": image_path.name, "reason": "low_iou_or_empty_prediction", "iou": metrics["iou"]})
            comparison = _comparison_image(image, truth, pred)
            cv2.imwrite(str(visuals_dir / f"{image_path.stem}_comparison.jpg"), comparison)

    pixel_metrics = binary_mask_metrics_from_counts(all_tp, all_fp, all_fn)
    validation_result_dir = getattr(val_results, "save_dir", None)
    validation_result_dir = Path(validation_result_dir) if validation_result_dir else None
    confusion_matrix_path = None
    if validation_result_dir:
        for candidate in ("confusion_matrix.png", "BoxConfusionMatrix.png", "MaskConfusionMatrix.png"):
            path = validation_result_dir / candidate
            if path.is_file():
                confusion_matrix_path = str(path)
                break
    report = {
        "evaluation_type": "held_out_test_split",
        "dataset_provenance": "AI-generated synthetic images with manually reviewed polygon labels required",
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "test_image_count": len(test_entries),
        "weights": str(weights),
        "imgsz": imgsz,
        "ultralytics_mask_metrics": map_metrics,
        "validation_metrics_report": str(ROOT / "results" / "arm_segmentation" / "training_report.json"),
        "pixel_metrics_aggregate": pixel_metrics,
        "mean_image_iou": float(np.mean([row["iou"] for row in per_image])) if per_image else None,
        "mean_image_dice": float(np.mean([row["dice"] for row in per_image])) if per_image else None,
        "inference_speed": {
            "mean_seconds_per_image": float(np.mean(inference_seconds)) if inference_seconds else None,
            "median_seconds_per_image": float(np.median(inference_seconds)) if inference_seconds else None,
            "images_per_second": float(1.0 / np.mean(inference_seconds)) if inference_seconds and np.mean(inference_seconds) > 0 else None,
            "timing_scope": "model.predict per image; excludes model loading and image decoding",
        },
        "confusion_matrix": confusion_matrix_path,
        "failure_cases": failures,
        "per_image_metrics_csv": str(output_dir / "per_image_metrics.csv"),
        "visual_comparisons_directory": str(visuals_dir),
        "limitation": "Synthetic test metrics do not establish performance on real children or clinical MUAC accuracy.",
    }
    (output_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["image", "inference_seconds", "iou", "dice", "pixel_precision", "pixel_recall", "true_positive_pixels", "false_positive_pixels", "false_negative_pixels"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_image)
    return report


def binary_metrics_from_counts(tp: int, fp: int, fn: int) -> dict[str, float]:
    denom = tp + fp + fn
    return {
        "iou": float(tp / denom) if denom else 1.0,
        "dice": float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 1.0,
        "pixel_precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "pixel_recall": float(tp / (tp + fn)) if tp + fn else 0.0,
    }


def binary_mask_metrics_from_counts(tp: int, fp: int, fn: int) -> dict[str, float]:
    return binary_metrics_from_counts(tp, fp, fn)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=416)
    args = parser.parse_args(argv)
    weights = args.weights if args.weights.is_absolute() else ROOT / args.weights
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir

    validation, _ = validate_annotations(split_dir=ROOT / "dataset" / "splits")
    if not validation["training_ready"]:
        print(
            f"Evaluation stopped: reviewed annotations are incomplete or invalid "
            f"({validation['valid_annotated_images']}/{validation['image_count']} labeled images).",
            file=sys.stderr,
        )
        return 2
    if not weights.is_file():
        print(f"Evaluation stopped: trained checkpoint not found: {weights}", file=sys.stderr)
        return 2
    try:
        report = evaluate_test_split(weights, output_dir, args.imgsz)
    except ImportError:
        print("Ultralytics is not installed; install requirements-segmentation.txt first.", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
