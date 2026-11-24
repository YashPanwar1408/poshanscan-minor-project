"""Evaluate an existing PoshanScan YOLO segmentation checkpoint in Colab.

This is evaluation-only: it never calls model.train(), edits the dataset, or
writes to the checkpoint/training-run directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

EXPECTED_COUNTS = {"train": 84, "val": 24, "test": 12}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_DATA = Path("/content/poshanscan_yolo_dataset")
DEFAULT_RUN = Path(
    "/content/poshanscan_colab_output/runs/"
    "yolov8n-seg-upper-arm-20261123T073138Z"
)
DEFAULT_WEIGHTS = DEFAULT_RUN / "weights" / "best.pt"
DEFAULT_OUTPUT = Path(
    "/content/poshanscan_colab_output/evaluation_existing_best_fixed_batch"
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _class_names(config: dict[str, Any]) -> dict[int, str]:
    names = config.get("names")
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    raise ValueError(f"dataset.yaml has no valid class names: {names!r}")


def _has_proper_self_intersection(points: np.ndarray) -> bool:
    """Detect crossing non-adjacent polygon edges without altering the annotation."""
    count = len(points)

    def cross(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    for first in range(count):
        a, b = points[first], points[(first + 1) % count]
        for second in range(first + 2, count):
            if first == 0 and second == count - 1:
                continue
            c, d = points[second], points[(second + 1) % count]
            c1, c2 = cross(a, b, c), cross(a, b, d)
            c3, c4 = cross(c, d, a), cross(c, d, b)
            if ((c1 > 1e-10 and c2 < -1e-10) or (c1 < -1e-10 and c2 > 1e-10)) and (
                (c3 > 1e-10 and c4 < -1e-10) or (c3 < -1e-10 and c4 > 1e-10)
            ):
                return True
    return False


def validate_dataset(root: Path) -> tuple[Path, dict[str, list[Path]], dict[str, Any]]:
    config_path = root / "dataset.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if _class_names(config) != {0: "upper_arm"}:
        raise ValueError(f"Expected only class 0: upper_arm; got {_class_names(config)!r}")
    for split in EXPECTED_COUNTS:
        if config.get(split) != f"images/{split}":
            raise ValueError(f"dataset.yaml {split} must point to images/{split}")
    configured_root = Path(str(config.get("path", "")))
    if not configured_root.is_absolute():
        configured_root = config_path.parent / configured_root
    if configured_root.resolve() != root.resolve():
        raise ValueError(
            f"dataset.yaml path resolves to {config.get('path')!r}, not {root.resolve()}. "
            "Keep the existing dataset and correct only its Colab path if needed."
        )

    metadata_root = root / "split_metadata"
    split_report_path = metadata_root / "split_report.json"
    split_manifest_path = metadata_root / "split_manifest.csv"
    if not split_report_path.is_file() or not split_manifest_path.is_file():
        raise FileNotFoundError("Dataset ZIP is missing its split report or included split manifest.")
    split_report = json.loads(split_report_path.read_text(encoding="utf-8"))
    if split_report.get("seed") != 2026:
        raise ValueError(f"Expected the fixed split seed 2026, got {split_report.get('seed')!r}")
    profile = split_report.get("dataset_profile", "full_fixed_split")
    if profile not in {"full_fixed_split", "verified_quarantine_subset"}:
        raise ValueError(f"Unsupported dataset profile: {profile!r}")
    if profile == "verified_quarantine_subset":
        if split_report.get("source_split_counts") != EXPECTED_COUNTS:
            raise ValueError("Verified subset does not record the original 84/24/12 split.")
        counts = split_report.get("split_counts")
        if not isinstance(counts, dict) or set(counts) != set(EXPECTED_COUNTS):
            raise ValueError("Verified subset has malformed included split counts.")
        expected_counts = {split: int(counts[split]) for split in EXPECTED_COUNTS}
        if any(count < 1 for count in expected_counts.values()):
            raise ValueError("Verified subset must retain non-empty train, validation and test partitions.")
    else:
        expected_counts = EXPECTED_COUNTS
        if split_report.get("split_counts") != EXPECTED_COUNTS:
            raise ValueError("Full fixed split counts differ from 84/24/12.")

    split_images: dict[str, list[Path]] = {}
    image_hashes: dict[str, str] = {}
    label_rows = 0
    polygon_areas: list[float] = []
    split_names: dict[str, list[str]] = {}
    multiple_polygon_images: list[str] = []
    self_intersections: list[dict[str, Any]] = []
    for split, expected in expected_counts.items():
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing {split} image/label directories under {root}")
        images = sorted(
            (item for item in image_dir.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda item: item.name.lower(),
        )
        labels = sorted(label_dir.glob("*.txt"), key=lambda item: item.name.lower())
        if len(images) != expected or len(labels) != expected:
            raise ValueError(
                f"{split}: expected {expected} images and labels, found {len(images)} and {len(labels)}"
            )
        image_stems = {item.stem for item in images}
        label_stems = {item.stem for item in labels}
        if image_stems != label_stems:
            raise ValueError(
                f"{split} image/label basename mismatch; image-only={sorted(image_stems-label_stems)[:5]}, "
                f"label-only={sorted(label_stems-image_stems)[:5]}"
            )
        split_images[split] = images
        split_names[split] = [item.name for item in images]
        for image_path in images:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Unreadable image: {image_path}")
            image_digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            if image_digest in image_hashes:
                raise ValueError(
                    f"Exact duplicate images detected in existing split data: "
                    f"{image_hashes[image_digest]} and {image_path}"
                )
            image_hashes[image_digest] = str(image_path)
            label_path = label_dir / f"{image_path.stem}.txt"
            polygons_in_image = 0
            for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
                fields = raw_line.split()
                if not fields:
                    continue
                if len(fields) < 7 or (len(fields) - 1) % 2:
                    raise ValueError(f"Malformed polygon row: {label_path}:{line_number}")
                try:
                    class_id = int(fields[0])
                    coords = np.asarray([float(value) for value in fields[1:]], dtype=np.float64).reshape(-1, 2)
                except (ValueError, OverflowError) as exc:
                    raise ValueError(f"Non-numeric polygon row: {label_path}:{line_number}") from exc
                if class_id != 0:
                    raise ValueError(f"Unexpected class {class_id} in {label_path}:{line_number}")
                if not np.isfinite(coords).all() or np.any(coords < 0) or np.any(coords > 1):
                    raise ValueError(f"Non-finite/out-of-range coordinates: {label_path}:{line_number}")
                if len(np.unique(coords, axis=0)) < 3:
                    raise ValueError(f"Polygon has fewer than three unique points: {label_path}:{line_number}")
                x, y = coords[:, 0], coords[:, 1]
                area = 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
                if area <= 1e-8:
                    raise ValueError(f"Zero-area polygon: {label_path}:{line_number}")
                if _has_proper_self_intersection(coords):
                    self_intersections.append({
                        "split": split,
                        "image": image_path.name,
                        "label_row": line_number,
                        "vertices": len(coords),
                    })
                polygon_areas.append(area)
                label_rows += 1
                polygons_in_image += 1
            if polygons_in_image > 1:
                multiple_polygon_images.append(f"{split}/{image_path.name}: {polygons_in_image} polygons")

    if len(image_hashes) != sum(expected_counts.values()):
        raise ValueError(f"Expected {sum(expected_counts.values())} exact-unique included images, got {len(image_hashes)}")
    included_manifest: dict[str, tuple[str, str]] = {}
    included_groups: dict[str, set[str]] = {}
    with split_manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            filename = Path(str(row.get("image_filename", ""))).name
            split = str(row.get("split", ""))
            group = str(row.get("duplicate_group", ""))
            if not filename or split not in EXPECTED_COUNTS or not group or filename in included_manifest:
                raise ValueError(f"Malformed/duplicate included split row: {row!r}")
            included_manifest[filename] = (split, group)
            included_groups.setdefault(group, set()).add(split)
    actual_map = {image.name: split for split, images in split_images.items() for image in images}
    if {name: split for name, (split, _) in included_manifest.items()} != actual_map:
        raise ValueError("Included split manifest does not match the evaluated image directories.")
    if any(len(values) != 1 for values in included_groups.values()):
        raise ValueError("Near-duplicate/source group crosses included split partitions.")
    if profile == "verified_quarantine_subset":
        source_path = metadata_root / "source_split_manifest.csv"
        selection_path = metadata_root / "dataset_selection_manifest.csv"
        if not source_path.is_file() or not selection_path.is_file():
            raise FileNotFoundError("Verified subset requires the original split and selection manifests.")
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if digest != split_report.get("source_split_manifest_sha256"):
            raise ValueError("Original source split manifest hash does not match the report.")
        source: dict[str, tuple[str, str]] = {}
        source_groups: dict[str, set[str]] = {}
        with source_path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                filename = Path(str(row.get("image_filename", ""))).name
                split, group = str(row.get("split", "")), str(row.get("duplicate_group", ""))
                if not filename or split not in EXPECTED_COUNTS or not group or filename in source:
                    raise ValueError(f"Malformed/duplicate source split row: {row!r}")
                source[filename] = (split, group)
                source_groups.setdefault(group, set()).add(split)
        source_counts = {split: sum(value[0] == split for value in source.values()) for split in EXPECTED_COUNTS}
        if len(source) != 120 or source_counts != EXPECTED_COUNTS:
            raise ValueError(f"Original source manifest does not equal 84/24/12: {source_counts}")
        if any(len(values) != 1 for values in source_groups.values()):
            raise ValueError("Original source/near-duplicate group crosses the frozen split.")
        if any(source.get(name) != (split, group) for name, (split, group) in included_manifest.items()):
            raise ValueError("An evaluated image was moved from its original split or duplicate group.")
        selection: dict[str, tuple[str, str, str]] = {}
        with selection_path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                filename = Path(str(row.get("image_filename", ""))).name
                split, group = str(row.get("split", "")), str(row.get("duplicate_group", ""))
                included = str(row.get("included", "")).lower()
                if not filename or split not in EXPECTED_COUNTS or included not in {"true", "false"} or filename in selection:
                    raise ValueError(f"Malformed/duplicate selection row: {row!r}")
                selection[filename] = (split, group, included)
        if len(selection) != 120:
            raise ValueError(f"Expected 120 source selection rows, found {len(selection)}")
        for filename, (split, group) in source.items():
            expected = (split, group, "true" if filename in actual_map else "false")
            if selection.get(filename) != expected:
                raise ValueError(f"Selection/source/included mismatch for {filename}")
    return config_path, split_images, {
        "counts": expected_counts,
        "dataset_profile": profile,
        "class_mapping": {"0": "upper_arm"},
        "polygon_rows": label_rows,
        "polygon_area_fraction_min": min(polygon_areas) if polygon_areas else None,
        "polygon_area_fraction_median": float(np.median(polygon_areas)) if polygon_areas else None,
        "polygon_area_fraction_max": max(polygon_areas) if polygon_areas else None,
        "self_intersecting_polygon_rows": self_intersections,
        "images_with_multiple_polygons": multiple_polygon_images,
        "exact_unique_images": len(image_hashes),
        "image_names_by_split": split_names,
        "split_files_modified": False,
        "note": "Structural validation cannot determine whether a human-drawn polygon follows the intended arm boundary.",
    }


def _read_training_curves(training_run: Path, output_dir: Path) -> dict[str, Any]:
    results_path = training_run / "results.csv"
    if not results_path.is_file():
        raise FileNotFoundError(f"Training results.csv is missing: {results_path}")
    with results_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        rows = [{str(key).strip(): value for key, value in row.items() if key is not None} for row in reader]
    if not rows:
        raise ValueError(f"Training results.csv has no data rows: {results_path}")
    numeric: dict[str, list[float]] = {}
    for key in rows[0]:
        values = []
        for row in rows:
            try:
                value = float(row.get(key, ""))
            except (TypeError, ValueError):
                break
            if not math.isfinite(value):
                break
            values.append(value)
        if len(values) == len(rows):
            numeric[key] = values

    import matplotlib.pyplot as plt

    groups = [
        ("Loss", [key for key in numeric if key.startswith(("train/", "val/"))]),
        ("Metrics", [key for key in numeric if key.startswith("metrics/")]),
        ("Learning rate", [key for key in numeric if key.startswith("lr/")]),
    ]
    fig, axes = plt.subplots(len(groups), 1, figsize=(12, 4 * len(groups)), squeeze=False)
    for axis, (title, keys) in zip(axes[:, 0], groups):
        if keys:
            for key in keys:
                axis.plot(range(1, len(numeric[key]) + 1), numeric[key], label=key)
            axis.legend(fontsize=8, ncol=2)
        else:
            axis.text(0.5, 0.5, "No matching numeric columns", ha="center", va="center")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    curve_path = output_dir / "training_curves_from_results_csv.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)

    def metric_stat(suffix: str) -> dict[str, Any] | None:
        key = next((candidate for candidate in numeric if candidate.endswith(suffix)), None)
        if key is None:
            return None
        values = numeric[key]
        best_index = int(np.argmax(values))
        tail = values[-min(10, len(values)):]
        return {
            "column": key,
            "best": values[best_index],
            "best_epoch": best_index + 1,
            "last": values[-1],
            "last_10_epoch_change": tail[-1] - tail[0] if len(tail) > 1 else 0.0,
        }

    args_path = training_run / "args.yaml"
    args = yaml.safe_load(args_path.read_text(encoding="utf-8")) if args_path.is_file() else {}
    box_map = metric_stat("mAP50(B)")
    mask_map = metric_stat("mAP50(M)")
    if box_map and mask_map and box_map["best"] < 0.05 and mask_map["best"] < 0.05:
        quality_observation = (
            "Best validation box and mask mAP50 are both below 0.05. This checkpoint is not usable "
            "for upper-arm segmentation based on these validation metrics; low values do not by themselves "
            "prove whether the cause is annotation quality, training, or data/model mismatch."
        )
    else:
        quality_observation = "Use the validation metrics and saved overlays; no quality conclusion was inferred from epoch count alone."
    metric_changes = {
        key: value["last_10_epoch_change"]
        for key, value in ((key, metric_stat(suffix)) for key, suffix in (
            ("box_mAP50", "mAP50(B)"),
            ("mask_mAP50", "mAP50(M)"),
            ("box_mAP50_95", "mAP50-95(B)"),
            ("mask_mAP50_95", "mAP50-95(M)"),
        ))
        if value is not None
    }
    if box_map and mask_map and box_map["best"] < 0.05 and mask_map["best"] < 0.05:
        convergence = "No useful convergence: best validation box and mask mAP50 stayed below 0.05."
    elif metric_changes and all(abs(change) < 0.01 for change in metric_changes.values()):
        convergence = "Validation mAP metrics changed by less than 0.01 over the final 10 epochs; they appear plateaued."
    else:
        convergence = "Validation metrics still changed over the final 10 epochs, or the results file lacks mAP columns."
    return {
        "results_csv": str(results_path),
        "epoch_rows": len(rows),
        "last_epoch_value": numeric.get("epoch", list(range(len(rows)))).__getitem__(-1),
        "training_args": args,
        "box_map50": box_map,
        "mask_map50": mask_map,
        "last_10_epoch_mAP_changes": metric_changes,
        "convergence_observation": convergence,
        "training_curves_png": str(curve_path),
        "quality_observation": quality_observation,
        "limitation": "A flat curve can be a plateau at poor performance; it does not make an inaccurate model usable.",
    }


def _metric_block(result: Any) -> dict[str, Any]:
    block: dict[str, Any] = {}
    for key in ("box", "seg"):
        metric = getattr(result, key, None)
        if metric is None:
            continue
        block[key] = {
            name: _json_safe(getattr(metric, name, None))
            for name in ("mp", "mr", "map50", "map75", "map")
        }
    block["results_dict"] = _json_safe(getattr(result, "results_dict", {}))
    block["save_dir"] = str(getattr(result, "save_dir", ""))
    return block


def _ground_truth_mask(label_path: Path, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for raw_line in label_path.read_text(encoding="utf-8-sig").splitlines():
        fields = raw_line.split()
        if not fields:
            continue
        coords = np.asarray([float(value) for value in fields[1:]], dtype=np.float64).reshape(-1, 2)
        points = np.rint(coords * np.array([width, height])).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [points], 255)
    return mask


def _prediction_mask(result: Any, width: int, height: int, confidence_threshold: float) -> tuple[np.ndarray, int]:
    mask = np.zeros((height, width), dtype=np.uint8)
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    data = getattr(masks, "data", None) if masks is not None else None
    if boxes is None or data is None:
        return mask, 0
    classes = boxes.cls.detach().cpu().numpy().reshape(-1).astype(int)
    confidences = boxes.conf.detach().cpu().numpy().reshape(-1)
    mask_data = data.detach().cpu().numpy()
    kept = 0
    for index, instance_mask in enumerate(mask_data):
        if index >= len(classes) or index >= len(confidences):
            continue
        if classes[index] != 0 or float(confidences[index]) < confidence_threshold:
            continue
        resized = cv2.resize(instance_mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        mask[resized >= 0.5] = 255
        kept += 1
    return mask, kept


def _binary_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    pred = predicted > 0
    gt = truth > 0
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    union = tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": int(pred.size - tp - fp - fn),
        "iou": tp / union if union else 1.0,
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "pixel_precision": tp / (tp + fp) if tp + fp else 0.0,
        "pixel_recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def _overlay_panel(image: np.ndarray, mask: np.ndarray, label: str, color: tuple[int, int, int]) -> np.ndarray:
    canvas = image.copy()
    if np.any(mask):
        tint = np.zeros_like(canvas)
        tint[mask > 0] = color
        canvas = cv2.addWeighted(canvas, 0.68, tint, 0.32, 0.0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 2)
    scale = min(360 / canvas.shape[1], 360 / canvas.shape[0])
    canvas = cv2.resize(canvas, (max(1, round(canvas.shape[1] * scale)), max(1, round(canvas.shape[0] * scale))))
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 32), (24, 24, 24), -1)
    cv2.putText(canvas, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _save_comparison(image: np.ndarray, truth: np.ndarray, pred_api: np.ndarray, pred_low: np.ndarray, path: Path) -> None:
    panels = [
        _overlay_panel(image, np.zeros_like(truth), "Image", (255, 255, 255)),
        _overlay_panel(image, truth, "Ground truth", (0, 220, 0)),
        _overlay_panel(image, pred_api, "Prediction conf=0.25", (0, 0, 255)),
        _overlay_panel(image, pred_low, "Prediction conf=0.001", (0, 220, 255)),
    ]
    common_height = max(panel.shape[0] for panel in panels)
    panels = [
        cv2.copyMakeBorder(panel, 0, common_height - panel.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(32, 32, 32))
        for panel in panels
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.hconcat(panels))


def _save_pixel_confusion(matrix: list[list[int]], path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(5.5, 4.5))
    axis.imshow(np.asarray(matrix, dtype=np.int64), cmap="Blues")
    axis.set_xticks([0, 1], labels=["background", "upper_arm"])
    axis.set_yticks([0, 1], labels=["background", "upper_arm"])
    axis.set_xlabel("Predicted pixel class")
    axis.set_ylabel("Ground truth pixel class")
    axis.set_title(title)
    for row in range(2):
        for column in range(2):
            axis.text(column, row, f"{matrix[row][column]:,}", ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_contact_sheet(comparison_paths: list[Path], path: Path, columns: int = 3) -> None:
    from PIL import Image, ImageDraw

    if not comparison_paths:
        return
    tiles = [Image.open(item).convert("RGB") for item in comparison_paths]
    tile_width = 900
    tile_height = max(1, round(tiles[0].height * tile_width / tiles[0].width))
    resized = [tile.resize((tile_width, tile_height)) for tile in tiles]
    rows = math.ceil(len(resized) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), (32, 32, 32))
    draw = ImageDraw.Draw(sheet)
    for index, tile in enumerate(resized):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        sheet.paste(tile, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    for tile in tiles:
        tile.close()


def _evaluate_split(
    model: Any,
    split_images: list[Path],
    data_root: Path,
    split: str,
    artifact_name: str,
    output_dir: Path,
    imgsz: int,
    device: int | str,
) -> dict[str, Any]:
    label_dir = data_root / "labels" / split
    comparisons_dir = output_dir / "comparisons" / artifact_name
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    per_image: list[dict[str, Any]] = []
    total_api = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    comparison_paths: list[Path] = []
    for image_path in split_images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode {image_path}")
        height, width = image.shape[:2]
        truth = _ground_truth_mask(label_dir / f"{image_path.stem}.txt", width, height)
        started = time.perf_counter()
        prediction = model.predict(
            source=image,
            imgsz=imgsz,
            conf=0.001,
            classes=[0],
            device=device,
            verbose=False,
        )
        elapsed = time.perf_counter() - started
        result = prediction[0] if prediction else None
        if result is None:
            pred_api = pred_low = np.zeros_like(truth)
            api_count = low_count = 0
        else:
            pred_api, api_count = _prediction_mask(result, width, height, 0.25)
            pred_low, low_count = _prediction_mask(result, width, height, 0.001)
        metrics = _binary_metrics(pred_api, truth)
        for key in total_api:
            total_api[key] += int(metrics[key])
        per_image.append({
            "image": image_path.name,
            "width": width,
            "height": height,
            "inference_seconds": elapsed,
            "prediction_instances_conf_0_25": api_count,
            "prediction_instances_conf_0_001": low_count,
            **metrics,
        })
        comparison_path = comparisons_dir / f"{image_path.stem}_comparison.jpg"
        _save_comparison(image, truth, pred_api, pred_low, comparison_path)
        comparison_paths.append(comparison_path)

    pixel_matrix = [
        [total_api["tn"], total_api["fp"]],
        [total_api["fn"], total_api["tp"]],
    ]
    _save_pixel_confusion(
        pixel_matrix,
        output_dir / f"{artifact_name}_pixel_confusion_matrix.png",
        f"{artifact_name} pixel confusion (confidence 0.25)",
    )
    with (output_dir / f"{artifact_name}_per_image_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "image", "width", "height", "inference_seconds", "prediction_instances_conf_0_25",
            "prediction_instances_conf_0_001", "iou", "dice", "pixel_precision", "pixel_recall",
            "tp", "fp", "fn", "tn",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_image)
    contact_sheet = output_dir / f"{artifact_name}_visual_contact_sheet.jpg"
    _save_contact_sheet(comparison_paths, contact_sheet)
    tp, fp, fn = total_api["tp"], total_api["fp"], total_api["fn"]
    return {
        "split": split,
        "artifact_name": artifact_name,
        "image_count": len(per_image),
        "imgsz": imgsz,
        "predict_confidence_for_pixel_metrics": 0.25,
        "diagnostic_low_confidence_overlay": 0.001,
        "pixel_metrics_conf_0_25": {
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
            "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
            "pixel_precision": tp / (tp + fp) if tp + fp else 0.0,
            "pixel_recall": tp / (tp + fn) if tp + fn else 0.0,
            "confusion_matrix_background_upper_arm": pixel_matrix,
        },
        "per_image_metrics_csv": str(output_dir / f"{artifact_name}_per_image_metrics.csv"),
        "pixel_confusion_matrix_png": str(output_dir / f"{artifact_name}_pixel_confusion_matrix.png"),
        "visual_comparisons_directory": str(comparisons_dir),
        "visual_contact_sheet": str(contact_sheet),
        "ultralytics_validation": {},
    }


def run_evaluation(
    weights: Path,
    data_root: Path,
    training_run: Path,
    output_dir: Path,
    imgsz: int = 416,
    batch: int = 1,
    api_imgsz: int = 416,
) -> dict[str, Any]:
    if batch <= 0:
        raise ValueError("Evaluation batch must be a positive integer. Use --batch 1 or --batch 4.")
    if imgsz < 64 or api_imgsz < 64:
        raise ValueError("Both evaluation image sizes must be at least 64.")
    if not weights.is_file():
        raise FileNotFoundError(f"Existing checkpoint not found: {weights}")
    if not (training_run / "results.csv").is_file():
        raise FileNotFoundError(f"Training results.csv not found under: {training_run}")
    if output_dir.exists():
        raise FileExistsError(
            f"Evaluation output directory already exists: {output_dir}. Choose a new --output-dir; "
            "the script will not overwrite prior evaluation artifacts."
        )
    config_path, split_images, dataset_audit = validate_dataset(data_root)
    output_dir.mkdir(parents=True)
    training_audit = _read_training_curves(training_run, output_dir)

    import torch
    from ultralytics import YOLO

    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(weights))
    names = getattr(model, "names", {})
    checkpoint_names = (
        {int(key): str(value) for key, value in names.items()}
        if isinstance(names, dict)
        else dict(enumerate(str(value) for value in names))
    )
    task = str(getattr(model, "task", ""))
    if task != "segment" or checkpoint_names != {0: "upper_arm"}:
        raise ValueError(f"Checkpoint mismatch: task={task!r}, names={checkpoint_names!r}")

    evaluation_log_path = output_dir / "evaluation.log"
    eval_report: dict[str, Any] = {}
    with evaluation_log_path.open("w", encoding="utf-8") as log_handle:
        class Tee:
            def __init__(self, stream: Any) -> None:
                self.stream = stream

            def write(self, value: str) -> int:
                log_handle.write(value)
                log_handle.flush()
                return self.stream.write(value)

            def flush(self) -> None:
                log_handle.flush()
                self.stream.flush()

        with redirect_stdout(Tee(sys.stdout)), redirect_stderr(Tee(sys.stderr)):
            print("Evaluation-only run; model.train() is not called.")
            print("Checkpoint:", weights)
            print("Task/class:", task, checkpoint_names)
            print("CUDA available:", bool(torch.cuda.is_available()))
            print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
            print("Fixed positive validation batch:", batch)
            print("Image size:", imgsz)
            print("API default image size comparison:", api_imgsz)
            print("Dataset split counts:", dataset_audit["counts"])
            evaluations = [
                (f"val_imgsz_{imgsz}", "val", imgsz),
                (f"test_imgsz_{imgsz}", "test", imgsz),
            ]
            if api_imgsz != imgsz:
                evaluations.append((f"test_api_imgsz_{api_imgsz}", "test", api_imgsz))
            for artifact_name, split, split_imgsz in evaluations:
                result = model.val(
                    data=str(config_path),
                    split=split,
                    imgsz=split_imgsz,
                    batch=batch,
                    device=device,
                    workers=2 if torch.cuda.is_available() else 0,
                    conf=0.001,
                    plots=True,
                    project=str(output_dir / "ultralytics"),
                    name=f"{artifact_name}_fixed_batch_{batch}",
                    exist_ok=False,
                    verbose=True,
                )
                eval_report[artifact_name] = _metric_block(result)
                print(f"{artifact_name} Ultralytics validation:", json.dumps(_json_safe(eval_report[artifact_name]), indent=2))

            for artifact_name, split, split_imgsz in evaluations:
                visual_metrics = _evaluate_split(
                    model=model,
                    split_images=split_images[split],
                    data_root=data_root,
                    split=split,
                    artifact_name=artifact_name,
                    output_dir=output_dir,
                    imgsz=split_imgsz,
                    device=device,
                )
                visual_metrics["ultralytics_validation"] = eval_report[artifact_name]
                eval_report[artifact_name] = visual_metrics

    report = {
        "evaluation_type": "evaluation_only_existing_checkpoint",
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "training_called": False,
        "checkpoint_modified": False,
        "weights": str(weights),
        "checkpoint_task": task,
        "checkpoint_names": checkpoint_names,
        "dataset_yaml": str(config_path),
        "dataset_audit": dataset_audit,
        "training_curve_audit": training_audit,
        "imgsz": imgsz,
        "api_imgsz_comparison": api_imgsz,
        "fixed_positive_batch": batch,
        "device": str(device),
        "evaluations": eval_report,
        "evaluation_log": str(evaluation_log_path),
        "limitation": "Synthetic image metrics and overlays do not establish real-world or clinical MUAC accuracy.",
    }
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe(report), indent=2))
    print("Evaluation report:", report_path)
    print("Checkpoint remains at:", weights)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--training-run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--api-imgsz", type=int, default=416)
    args = parser.parse_args()
    try:
        run_evaluation(
            weights=args.weights,
            data_root=args.data_root,
            training_run=args.training_run,
            output_dir=args.output_dir,
            imgsz=args.imgsz,
            batch=args.batch,
            api_imgsz=args.api_imgsz,
        )
    except Exception as exc:
        print(f"Evaluation stopped: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
