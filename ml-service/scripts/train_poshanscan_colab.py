"""Colab GPU training workflow for the prepared YOLO images/{train,val,test} ZIP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from importlib import metadata
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

EXPECTED_COUNTS = {"train": 84, "val": 24, "test": 12}
SPLITS = tuple(EXPECTED_COUNTS)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
EXPECTED_SPLIT_SEED = 2026


class _Tee:
    """Copy live training output to both the notebook and a persistent Drive log."""

    def __init__(self, stream: Any, log_handle: Any) -> None:
        self.stream = stream
        self.log_handle = log_handle

    def write(self, value: str) -> int:
        self.log_handle.write(value)
        self.log_handle.flush()
        return self.stream.write(value)

    def flush(self) -> None:
        self.log_handle.flush()
        self.stream.flush()


def _proper_segment_intersection(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def cross(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1, o2 = cross(a, b, c), cross(a, b, d)
    o3, o4 = cross(c, d, a), cross(c, d, b)
    eps = 1e-10
    return (
        ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps))
        and ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps))
    )


def _polygon_has_proper_self_intersection(points: np.ndarray) -> bool:
    count = len(points)
    for first in range(count):
        a, b = points[first], points[(first + 1) % count]
        for second in range(first + 2, count):
            if first == 0 and second == count - 1:
                continue
            if _proper_segment_intersection(a, b, points[second], points[(second + 1) % count]):
                return True
    return False


def _refresh_training_curves(run_dir: Path) -> None:
    """Write a live loss/metric plot after each completed train+validation epoch."""
    results_path = run_dir / "results.csv"
    if not results_path.is_file():
        return
    with results_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        rows = [{str(key).strip(): value for key, value in row.items() if key is not None} for row in reader]
    if not rows:
        return
    numeric: dict[str, list[float]] = {}
    for key in rows[0]:
        values: list[float] = []
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
        ("Validation metrics", [key for key in numeric if key.startswith("metrics/")]),
        ("Learning rate", [key for key in numeric if key.startswith("lr/")]),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), squeeze=False)
    for axis, (title, columns) in zip(axes[:, 0], groups):
        for column in columns:
            axis.plot(range(1, len(numeric[column]) + 1), numeric[column], label=column)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.25)
        if columns:
            axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    destination = run_dir / "training_curves_live.png"
    temporary = run_dir / "training_curves_live.tmp.png"
    fig.savefig(temporary, dpi=120)
    plt.close(fig)
    temporary.replace(destination)


def _load_colab_torch():
    """Use, validate, and preserve the PyTorch/torchvision pair provided by Colab."""
    try:
        import torch
        import torchvision
    except Exception as exc:
        raise RuntimeError(
            "Colab's current PyTorch/torchvision pair failed to import. Restart with a fresh GPU runtime; "
            "this workflow will not reinstall or replace either package."
        ) from exc
    versions = (metadata.version("torch"), metadata.version("torchvision"))
    try:
        torchvision.ops.nms(
            torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0]]),
            torch.tensor([0.9, 0.8]),
            0.5,
        )
    except Exception as exc:
        raise RuntimeError(
            f"torch {versions[0]} and torchvision {versions[1]} cannot run NMS in this runtime. "
            "Reconnect to a fresh compatible Colab runtime; do not pip-install another torch pair."
        ) from exc
    return torch, versions


def _load_yolo_without_torch_updates(torch_versions: tuple[str, str]):
    """Install Ultralytics and missing non-Torch dependencies without dependency upgrades."""
    try:
        metadata.version("ultralytics")
    except PackageNotFoundError:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--no-deps", "ultralytics>=8.2,<9"
        ])

    from packaging.requirements import Requirement

    pending = ["ultralytics"]
    processed: set[str] = set()
    while pending:
        package = pending.pop(0)
        normalized_package = package.lower().replace("_", "-")
        if normalized_package in processed:
            continue
        processed.add(normalized_package)
        requirements = metadata.requires(package) or []
        for requirement_text in requirements:
            requirement = Requirement(requirement_text)
            if requirement.marker and not requirement.marker.evaluate():
                continue
            name = requirement.name
            normalized_name = name.lower().replace("_", "-")
            try:
                installed = metadata.version(name)
            except PackageNotFoundError:
                installed = None

            if normalized_name in {"torch", "torchvision"}:
                expected = torch_versions[0 if normalized_name == "torch" else 1]
                if installed != expected:
                    raise RuntimeError(f"{name} changed unexpectedly; stopping before training.")
                if requirement.specifier and not requirement.specifier.contains(installed, prereleases=True):
                    raise RuntimeError(
                        f"Colab {name} {installed} does not satisfy {requirement_text}. "
                        "No PyTorch package will be replaced. Reconnect to a fresh runtime."
                    )
                continue

            if installed is None:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--no-deps", requirement_text
                ])
            elif requirement.specifier and not requirement.specifier.contains(installed, prereleases=True):
                raise RuntimeError(
                    f"Installed {name}=={installed} conflicts with {requirement_text}. "
                    "No existing package was upgraded; reconnect to a fresh Colab runtime."
                )
            pending.append(name)

    if (metadata.version("torch"), metadata.version("torchvision")) != torch_versions:
        raise RuntimeError("PyTorch or torchvision changed during dependency setup.")
    from ultralytics import YOLO
    return YOLO


def _validate_dataset(root: Path) -> tuple[dict[str, list[Path]], dict[str, Any]]:
    config_path = root / "dataset.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("names") != {0: "upper_arm"}:
        raise ValueError(f"Expected one class, 0: upper_arm; got {config.get('names')!r}.")
    for split in SPLITS:
        if config.get(split) != f"images/{split}":
            raise ValueError(f"dataset.yaml {split} path must be images/{split}.")
    configured_root = Path(str(config.get("path", "")))
    if not configured_root.is_absolute():
        configured_root = config_path.parent / configured_root
    if configured_root.resolve() != root.resolve():
        raise ValueError(
            f"dataset.yaml path must resolve to {root.resolve()}, got {config.get('path')!r}. "
            "Extract the ZIP into /content/poshanscan_yolo_dataset."
        )

    metadata_root = root / "split_metadata"
    split_report_path = metadata_root / "split_report.json"
    split_manifest_path = metadata_root / "split_manifest.csv"
    if not split_report_path.is_file() or not split_manifest_path.is_file():
        raise FileNotFoundError(
            "The dataset ZIP must include split_metadata/split_report.json and split_manifest.csv. "
            "Do not regenerate the split."
        )
    split_report = json.loads(split_report_path.read_text(encoding="utf-8"))
    if split_report.get("seed") != EXPECTED_SPLIT_SEED:
        raise ValueError(f"Expected split seed {EXPECTED_SPLIT_SEED}, got {split_report.get('seed')!r}.")
    dataset_profile = split_report.get("dataset_profile", "full_fixed_split")
    if dataset_profile not in {"full_fixed_split", "verified_quarantine_subset"}:
        raise ValueError(f"Unsupported dataset profile: {dataset_profile!r}")
    if dataset_profile == "verified_quarantine_subset":
        expected_counts = split_report.get("split_counts")
        if split_report.get("source_split_counts") != EXPECTED_COUNTS:
            raise ValueError("Quarantine subset does not record the original 84/24/12 source split.")
        if not isinstance(expected_counts, dict) or set(expected_counts) != set(SPLITS):
            raise ValueError("Quarantine subset has malformed included split counts.")
        expected_counts = {split: int(expected_counts[split]) for split in SPLITS}
        if any(value < 1 for value in expected_counts.values()):
            raise ValueError("The verified subset must retain train, validation, and test images.")
    else:
        expected_counts = EXPECTED_COUNTS
        if split_report.get("split_counts") != EXPECTED_COUNTS:
            raise ValueError(f"Full dataset counts differ from the frozen split: {split_report.get('split_counts')!r}.")

    label_root = root / "labels"
    split_images: dict[str, list[Path]] = {}
    hashes: dict[str, set[str]] = {}
    split_manifest: dict[str, list[str]] = {}
    file_sha256: dict[str, dict[str, dict[str, str]]] = {
        split: {"images": {}, "labels": {}} for split in SPLITS
    }
    active_count = 0
    for split, expected in expected_counts.items():
        image_dir = root / "images" / split
        labels_dir = label_root / split
        images = sorted(
            (path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda path: path.name.lower(),
        )
        labels = sorted(path for path in labels_dir.glob("*.txt") if path.is_file())
        if len(images) != expected or len(labels) != expected:
            raise ValueError(
                f"{split}: expected {expected} images and labels, got {len(images)} and {len(labels)}."
            )
        if {image.stem for image in images} != {label.stem for label in labels}:
            raise ValueError(f"{split} image basenames do not match label basenames.")
        split_images[split] = images
        split_manifest[split] = [image.name for image in images]
        active_count += len(images)
        for image in images:
            if cv2.imread(str(image), cv2.IMREAD_COLOR) is None:
                raise ValueError(f"Unreadable image: {image}")
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            hashes.setdefault(digest, set()).add(split)
            label_path = labels_dir / f"{image.stem}.txt"
            file_sha256[split]["images"][image.name] = digest
            file_sha256[split]["labels"][label_path.name] = hashlib.sha256(label_path.read_bytes()).hexdigest()
            _validate_label(label_path)
    expected_total = sum(expected_counts.values())
    if active_count != expected_total or len(hashes) != expected_total:
        raise ValueError(f"Expected {expected_total} exact-unique included images across the source partitions.")
    if any(len(splits) != 1 for splits in hashes.values()):
        raise ValueError("An exact duplicate image crosses train/validation/test splits.")
    if not split_report.get("no_duplicate_or_near_duplicate_group_leakage"):
        raise ValueError("Split report indicates near-duplicate/source-group leakage.")

    metadata_manifest: dict[str, str] = {}
    metadata_groups: dict[str, str] = {}
    group_splits: dict[str, set[str]] = {}
    with split_manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            filename = Path(str(row.get("image_filename", ""))).name
            split = str(row.get("split", ""))
            group = str(row.get("duplicate_group", ""))
            if not filename or split not in EXPECTED_COUNTS or not group:
                raise ValueError(f"Malformed row in {split_manifest_path}: {row!r}")
            if filename in metadata_manifest:
                raise ValueError(f"Repeated image in split manifest: {filename}")
            metadata_manifest[filename] = split
            metadata_groups[filename] = group
            group_splits.setdefault(group, set()).add(split)
    actual_splits = {
        image.name: split
        for split, images in split_images.items()
        for image in images
    }
    if metadata_manifest != actual_splits:
        raise ValueError("split_metadata/split_manifest.csv does not match the fixed image folders.")
    if any(len(group_values) != 1 for group_values in group_splits.values()):
        raise ValueError("A duplicate/source group is split across train/validation/test.")
    for split, expected in expected_counts.items():
        list_path = metadata_root / f"{split}.txt"
        if not list_path.is_file():
            raise FileNotFoundError(list_path)
        listed_names = {
            Path(line.strip().replace("\\", "/")).name
            for line in list_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        }
        actual_names = {image.name for image in split_images[split]}
        if len(listed_names) != expected or listed_names != actual_names:
            raise ValueError(f"Embedded {split}.txt does not match the existing {split} image folder.")
    if dataset_profile == "verified_quarantine_subset":
        source_path = metadata_root / "source_split_manifest.csv"
        selection_path = metadata_root / "dataset_selection_manifest.csv"
        if not source_path.is_file() or not selection_path.is_file():
            raise FileNotFoundError("Verified subset requires source_split_manifest.csv and dataset_selection_manifest.csv.")
        source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if source_digest != split_report.get("source_split_manifest_sha256"):
            raise ValueError("Embedded original split manifest SHA-256 does not match the audit report.")
        source_manifest: dict[str, tuple[str, str]] = {}
        source_groups: dict[str, set[str]] = {}
        with source_path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                filename = Path(str(row.get("image_filename", ""))).name
                split = str(row.get("split", ""))
                group = str(row.get("duplicate_group", ""))
                if not filename or split not in SPLITS or not group or filename in source_manifest:
                    raise ValueError(f"Malformed/duplicate original split row: {row!r}")
                source_manifest[filename] = (split, group)
                source_groups.setdefault(group, set()).add(split)
        source_counts = {split: sum(item[0] == split for item in source_manifest.values()) for split in SPLITS}
        if len(source_manifest) != 120 or source_counts != EXPECTED_COUNTS:
            raise ValueError(f"Original split manifest no longer records 84/24/12: {source_counts}")
        if any(len(groups) != 1 for groups in source_groups.values()):
            raise ValueError("Original near-duplicate/source group spans multiple splits.")
        if any(source_manifest.get(name) != (split, metadata_groups[name]) for name, split in metadata_manifest.items()):
            raise ValueError("An included image was moved from its original split/group assignment.")
        selection: dict[str, tuple[str, str, str]] = {}
        with selection_path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                filename = Path(str(row.get("image_filename", ""))).name
                split, group = str(row.get("split", "")), str(row.get("duplicate_group", ""))
                included = str(row.get("included", "")).lower()
                if not filename or split not in SPLITS or included not in {"true", "false"} or filename in selection:
                    raise ValueError(f"Malformed/duplicate dataset selection row: {row!r}")
                selection[filename] = (split, group, included)
        if len(selection) != 120:
            raise ValueError(f"Dataset selection manifest must account for all 120 source images; got {len(selection)}.")
        for filename, (split, group) in source_manifest.items():
            if selection.get(filename) != (split, group, "true" if filename in actual_splits else "false"):
                raise ValueError(f"Selection manifest disagrees with source/included set for {filename}.")
        if split_report.get("included_count") != expected_total or split_report.get("quarantined_count") != 120 - expected_total:
            raise ValueError("Subset report included/quarantined totals do not match its manifests.")
    return split_images, {
        "counts": expected_counts,
        "dataset_profile": dataset_profile,
        "class_mapping": {0: "upper_arm"},
        "split_seed": EXPECTED_SPLIT_SEED,
        "image_names_by_split": split_manifest,
        "exact_unique_images": len(hashes),
        "file_sha256": file_sha256,
        "duplicate_group_leakage": False,
        "split_files_regenerated": False,
    }


def _validate_label(label_path: Path) -> None:
    rows = [line.strip() for line in label_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty polygon label: {label_path}")
    for number, row in enumerate(rows, start=1):
        fields = row.split()
        try:
            class_id = int(fields[0])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Malformed/class-mismatched polygon: {label_path}:{number}") from exc
        if len(fields) < 7 or (len(fields) - 1) % 2 or class_id != 0:
            raise ValueError(f"Malformed/class-mismatched polygon: {label_path}:{number}")
        values = np.asarray([float(value) for value in fields[1:]], dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError(f"Invalid normalized coordinates: {label_path}:{number}")
        points = values.reshape(-1, 2)
        if len(np.unique(points, axis=0)) < 3:
            raise ValueError(f"Polygon has fewer than three unique points: {label_path}:{number}")
        area = 0.5 * abs(float(np.dot(points[:, 0], np.roll(points[:, 1], 1)) - np.dot(points[:, 1], np.roll(points[:, 0], 1))))
        if area <= 1e-8:
            raise ValueError(f"Zero-area polygon: {label_path}:{number}")
        if _polygon_has_proper_self_intersection(points):
            raise ValueError(f"Self-intersecting polygon: {label_path}:{number}; correct it manually before training.")


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


def _truth_mask(label_path: Path, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for row in label_path.read_text(encoding="utf-8-sig").splitlines():
        fields = row.split()
        if fields:
            points = np.asarray([float(value) for value in fields[1:]], dtype=np.float64).reshape(-1, 2)
            points = np.round(points * np.array([width - 1, height - 1])).astype(np.int32)
            cv2.fillPoly(mask, [points], 255)
    return mask


def _prediction_mask(result: Any, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    polygons = getattr(masks, "xy", None) if masks is not None else None
    if boxes is None or polygons is None:
        return mask
    classes = boxes.cls.detach().cpu().numpy().astype(int).reshape(-1)
    for index, polygon in enumerate(polygons):
        if index < len(classes) and classes[index] == 0:
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            if len(points) >= 3 and np.isfinite(points).all():
                cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 255)
    return mask


def _pixel_counts(predicted: np.ndarray, truth: np.ndarray) -> dict[str, int | float]:
    pred, gt = predicted > 0, truth > 0
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": int(pred.size - tp - fp - fn),
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "pixel_precision": tp / (tp + fp) if tp + fp else 0.0,
        "pixel_recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def _comparison_panel(image: np.ndarray, mask: np.ndarray, label: str, color: tuple[int, int, int]) -> np.ndarray:
    tint = np.zeros_like(image)
    tint[mask > 0] = color
    panel = cv2.addWeighted(image, 0.72, tint, 0.28, 0.0)
    panel = cv2.resize(panel, (360, 360))
    cv2.putText(panel, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return panel


def _save_pixel_confusion_matrix(matrix: list[list[int]], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(5, 4))
    axis.imshow(np.asarray(matrix, dtype=np.int64), cmap="Blues")
    axis.set_xticks([0, 1], labels=["background", "upper_arm"])
    axis.set_yticks([0, 1], labels=["background", "upper_arm"])
    axis.set_xlabel("Predicted pixel class")
    axis.set_ylabel("Ground-truth pixel class")
    axis.set_title("Held-out test pixel confusion matrix")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, f"{matrix[row][column]:,}", ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_training(
    data_root: Path,
    training_runs_root: Path,
    weights: str,
    epochs: int,
    imgsz: int,
    seed: int,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    torch, torch_versions = _load_colab_torch()
    split_images, dataset_summary = _validate_dataset(data_root)
    YOLO = _load_yolo_without_torch_updates(torch_versions)
    config_path = data_root / "dataset.yaml"
    training_runs_root = training_runs_root.resolve()
    training_runs_root.mkdir(parents=True, exist_ok=True)
    gpu = bool(torch.cuda.is_available())
    device: int | str = 0 if gpu else "cpu"
    # Keep the value passed into the trainer positive: training-time validation
    # shares this trainer configuration in some Ultralytics versions. The
    # separate held-out validation/test calls below use evaluation_batch=1.
    batch = 4 if gpu else 1
    evaluation_batch = 1
    workers = 2 if gpu else 0

    if resume_checkpoint is not None:
        resume_checkpoint = resume_checkpoint.resolve()
        if resume_checkpoint.name != "last.pt" or not resume_checkpoint.is_file():
            raise FileNotFoundError(f"Resume from a saved last.pt checkpoint, got {resume_checkpoint}.")
        run_dir = resume_checkpoint.parent.parent.resolve()
        try:
            run_dir.relative_to(training_runs_root)
        except ValueError as exc:
            raise ValueError(f"Resume checkpoint must be inside {training_runs_root}, got {resume_checkpoint}.") from exc
        run_name = run_dir.name
        model = YOLO(str(resume_checkpoint))
        training_config_path = run_dir / "training_config.json"
        if not training_config_path.is_file():
            raise FileNotFoundError(f"Cannot resume without the saved training config: {training_config_path}")
        preflight_path = run_dir / "dataset_preflight.json"
        if not preflight_path.is_file():
            raise FileNotFoundError(f"Cannot safely resume without the saved dataset audit: {preflight_path}")
        saved_dataset = json.loads(preflight_path.read_text(encoding="utf-8"))
        if saved_dataset.get("file_sha256") != dataset_summary.get("file_sha256"):
            raise ValueError(
                "Dataset image/label checksums differ from the interrupted run. "
                "Do not resume with changed annotations; start a separate new run after approval."
            )
        training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
        with (run_dir / "resume_history.jsonl").open("a", encoding="utf-8") as history:
            history.write(json.dumps({"resume_timestamp_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": str(resume_checkpoint)}) + "\n")
    else:
        run_name = "yolov8n-seg-upper-arm-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = (training_runs_root / run_name).resolve()
        if run_dir.exists():
            raise FileExistsError(f"Refusing to reuse an existing training directory: {run_dir}")
        run_dir.mkdir(parents=True)
        training_config = {
            "data": str(config_path),
            "model": weights,
            "pretrained": True,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "evaluation_batch": evaluation_batch,
            "device": str(device),
            "workers": workers,
            "seed": seed,
            "deterministic": True,
            "patience": 20,
            "fliplr": 0.5,
            "flipud": 0.0,
            "degrees": 5.0,
            "translate": 0.03,
            "scale": 0.10,
            "hsv_h": 0.01,
            "hsv_s": 0.30,
            "hsv_v": 0.20,
            "mosaic": 0.0,
            "copy_paste": 0.0,
            "plots": True,
            "save": True,
            "save_period": 10,
            "project": str(training_runs_root),
            "name": run_name,
            "run_directory": str(run_dir),
        }
        (run_dir / "training_config.json").write_text(
            json.dumps(_json_safe(training_config), indent=2) + "\n", encoding="utf-8"
        )
        model = YOLO(weights)

    if getattr(model, "task", None) != "segment":
        raise RuntimeError(f"Expected segmentation weights, got task={getattr(model, 'task', None)!r}.")
    model.add_callback("on_fit_epoch_end", lambda trainer: _refresh_training_curves(Path(trainer.save_dir)))
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "training_status.json"
    (run_dir / "dataset_preflight.json").write_text(
        json.dumps(_json_safe(dataset_summary), indent=2) + "\n", encoding="utf-8"
    )
    status = {
        "status": "running",
        "run_directory": str(run_dir),
        "training_log": str(run_dir / "training.log"),
        "best_checkpoint": str(run_dir / "weights" / "best.pt"),
        "last_checkpoint": str(run_dir / "weights" / "last.pt"),
        "started_or_resumed_utc": datetime.now(timezone.utc).isoformat(),
        "resumed": resume_checkpoint is not None,
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    print("Training run directory (Google Drive):", run_dir)
    print("CUDA available:", gpu)
    print("Selected device:", torch.cuda.get_device_name(0) if gpu else "CPU fallback")
    print("Training batch:", batch, "validation/test batch:", evaluation_batch)
    print("Image size:", imgsz, "seed:", seed)
    log_path = run_dir / "training.log"
    log_mode = "a" if resume_checkpoint is not None else "w"
    logger_handler = logging.FileHandler(log_path, mode=log_mode, encoding="utf-8")
    logger_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    ultralytics_logger = logging.getLogger("ultralytics")
    ultralytics_logger.addHandler(logger_handler)
    try:
        with log_path.open(log_mode, encoding="utf-8") as log_handle:
            with redirect_stdout(_Tee(sys.stdout, log_handle)), redirect_stderr(_Tee(sys.stderr, log_handle)):
                if resume_checkpoint is not None:
                    print("Resuming from Drive checkpoint:", resume_checkpoint)
                    model.train(resume=True, data=str(config_path))
                else:
                    print("Starting fresh pretrained transfer learning run:", run_dir)
                    model.train(
                        data=str(config_path),
                        epochs=epochs,
                        imgsz=imgsz,
                        batch=batch,
                        device=device,
                        workers=workers,
                        seed=seed,
                        deterministic=True,
                        pretrained=True,
                        patience=20,
                        fliplr=0.5,
                        flipud=0.0,
                        degrees=5.0,
                        translate=0.03,
                        scale=0.10,
                        hsv_h=0.01,
                        hsv_s=0.30,
                        hsv_v=0.20,
                        mosaic=0.0,
                        copy_paste=0.0,
                        plots=True,
                        save=True,
                        save_period=10,
                        project=str(training_runs_root),
                        name=run_name,
                        exist_ok=True,
                    )
    except Exception as exc:
        status.update({
            "status": "interrupted_or_failed",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "error": repr(exc),
        })
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        ultralytics_logger.removeHandler(logger_handler)
        logger_handler.close()

    trainer = getattr(model, "trainer", None)
    if trainer is None:
        raise RuntimeError("YOLO training returned without a trainer.")
    actual_run_dir = Path(trainer.save_dir).resolve()
    if actual_run_dir != run_dir:
        raise RuntimeError(f"Ultralytics wrote to {actual_run_dir}, expected the Drive run folder {run_dir}.")
    run_dir = actual_run_dir
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"
    if not best_path.is_file() or not last_path.is_file() or not results_csv.is_file():
        raise FileNotFoundError(f"Training outputs are incomplete under {run_dir}; inspect training.log and last.pt.")
    if not (run_dir / "training_curves_live.png").is_file():
        _refresh_training_curves(run_dir)

    ultralytics_version = metadata.version("ultralytics")
    training_report = {
        "status": "training_completed_test_evaluation_pending",
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_summary,
        "test_split_used_for_training": False,
        "pretrained_checkpoint": weights if resume_checkpoint is None else None,
        "training_config": training_config,
        "validation_metrics_last_epoch": _json_safe(getattr(trainer, "metrics", {}) or {}),
        "runtime_versions": {
            "python": sys.version.split()[0],
            "torch": torch_versions[0],
            "torchvision": torch_versions[1],
            "ultralytics": ultralytics_version,
        },
        "training_run_directory": str(run_dir),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "results_csv": str(results_csv),
        "training_log": str(log_path),
        "training_curves_live": str(run_dir / "training_curves_live.png"),
        "limitation": "Synthetic validation/test metrics do not establish real-world or clinical MUAC accuracy.",
    }
    (run_dir / "training_report.json").write_text(
        json.dumps(_json_safe(training_report), indent=2) + "\n", encoding="utf-8"
    )
    status.update({"status": "training_completed_test_evaluation_pending", "updated_utc": datetime.now(timezone.utc).isoformat()})
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    best_model = YOLO(str(best_path))
    names = best_model.names
    names = {int(key): value for key, value in names.items()} if isinstance(names, dict) else dict(enumerate(names))
    if best_model.task != "segment" or names != {0: "upper_arm"}:
        raise RuntimeError(f"Checkpoint does not match the FastAPI class/task contract: task={best_model.task}, names={names}.")
    compatibility = {
        "task": best_model.task,
        "names": names,
        "ultralytics_version": ultralytics_version,
        "torch_version": torch_versions[0],
        "torchvision_version": torch_versions[1],
        "checkpoint": str(best_path),
    }
    (run_dir / "checkpoint_compatibility.json").write_text(json.dumps(compatibility, indent=2) + "\n", encoding="utf-8")

    evaluation_name = "held_out_test_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    eval_dir = run_dir / "evaluation" / evaluation_name
    eval_dir.mkdir(parents=True)
    test_validation = None
    eval_logger_handler = logging.FileHandler(eval_dir / "evaluation.log", mode="a", encoding="utf-8")
    eval_logger_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    ultralytics_logger = logging.getLogger("ultralytics")
    ultralytics_logger.addHandler(eval_logger_handler)
    try:
        with (eval_dir / "evaluation.log").open("w", encoding="utf-8") as eval_log:
            with redirect_stdout(_Tee(sys.stdout, eval_log)), redirect_stderr(_Tee(sys.stderr, eval_log)):
                test_validation = best_model.val(
                    data=str(config_path), split="test", imgsz=imgsz, batch=evaluation_batch, device=device,
                    workers=2 if gpu else 0, plots=True, project=str(eval_dir.parent), name=evaluation_name,
                    exist_ok=True, verbose=True,
                )
                seg_metrics = getattr(test_validation, "seg", None)
                test_map = {
                    "mask_precision": _finite_float(getattr(seg_metrics, "mp", None)),
                    "mask_recall": _finite_float(getattr(seg_metrics, "mr", None)),
                    "map50": _finite_float(getattr(seg_metrics, "map50", None)),
                    "map50_95": _finite_float(getattr(seg_metrics, "map", None)),
                }
                confusion_dir = Path(getattr(test_validation, "save_dir", eval_dir))
                ultralytics_confusion = next(
                    (str(path) for name in ("confusion_matrix.png", "MaskConfusionMatrix.png", "BoxConfusionMatrix.png")
                     if (path := confusion_dir / name).is_file()),
                    None,
                )

                visuals_dir = eval_dir / "test_visual_comparisons"
                visuals_dir.mkdir(parents=True, exist_ok=True)
                label_root = data_root / "labels" / "test"
                per_image = []
                totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
                durations = []
                failure_cases = []
                for image_path in split_images["test"]:
                    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    height, width = image.shape[:2]
                    truth = _truth_mask(label_root / f"{image_path.stem}.txt", width, height)
                    if gpu:
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    predictions = best_model.predict(
                        source=image, imgsz=imgsz, conf=0.25, device=device, verbose=False
                    )
                    if gpu:
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    durations.append(elapsed)
                    predicted = _prediction_mask(predictions[0], width, height) if predictions else np.zeros_like(truth)
                    metrics = _pixel_counts(predicted, truth)
                    for key in totals:
                        totals[key] += metrics[key]
                    per_image.append({"image": image_path.name, "inference_seconds": elapsed, **metrics})
                    if metrics["iou"] < 0.5 or not np.any(predicted):
                        failure_cases.append({"image": image_path.name, "iou": metrics["iou"], "reason": "low_iou_or_empty_prediction"})
                    cv2.imwrite(str(visuals_dir / f"{image_path.stem}_comparison.jpg"), cv2.hconcat([
                        _comparison_panel(image, np.zeros_like(truth), "Original", (0, 0, 0)),
                        _comparison_panel(image, truth, "Ground truth", (0, 220, 0)),
                        _comparison_panel(image, predicted, "Prediction conf=0.25", (0, 0, 255)),
                    ]))

                confusion = [[totals["tn"], totals["fp"]], [totals["fn"], totals["tp"]]]
                confusion_png = eval_dir / "pixel_confusion_matrix.png"
                _save_pixel_confusion_matrix(confusion, confusion_png)
                with (eval_dir / "pixel_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["ground_truth\\predicted", "background", "upper_arm"])
                    writer.writerow(["background", *confusion[0]])
                    writer.writerow(["upper_arm", *confusion[1]])

                tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
                union = tp + fp + fn
                mean_seconds = float(np.mean(durations)) if durations else None
                evaluation_report = {
                    "evaluation_type": "held_out_test_split",
                    "test_image_count": len(per_image),
                    "weights": str(best_path),
                    "imgsz": imgsz,
                    "fixed_positive_batch": evaluation_batch,
                    "ultralytics_mask_metrics": test_map,
                    "pixel_metrics_aggregate": {
                        "iou": tp / union if union else 1.0,
                        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
                        "pixel_precision": tp / (tp + fp) if tp + fp else 0.0,
                        "pixel_recall": tp / (tp + fn) if tp + fn else 0.0,
                        "confusion_matrix_order": ["background", "upper_arm"],
                        "confusion_matrix_counts": confusion,
                    },
                    "mean_image_iou": float(np.mean([row["iou"] for row in per_image])) if per_image else None,
                    "mean_image_dice": float(np.mean([row["dice"] for row in per_image])) if per_image else None,
                    "inference_speed": {
                        "mean_seconds_per_image": mean_seconds,
                        "median_seconds_per_image": float(np.median(durations)) if durations else None,
                        "images_per_second": 1.0 / mean_seconds if mean_seconds and mean_seconds > 0 else None,
                        "timing_scope": "predict per image; excludes model loading and image decoding",
                    },
                    "ultralytics_confusion_matrix": ultralytics_confusion,
                    "pixel_confusion_matrix_png": str(confusion_png),
                    "per_image_metrics_csv": str(eval_dir / "per_image_metrics.csv"),
                    "visual_comparisons_directory": str(visuals_dir),
                    "failure_cases": failure_cases,
                    "limitation": "AI-generated test images do not establish performance on real images or clinical MUAC accuracy.",
                }
                with (eval_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                    fields = ["image", "inference_seconds", "iou", "dice", "pixel_precision", "pixel_recall", "tp", "fp", "fn", "tn"]
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(per_image)
                (eval_dir / "evaluation_report.json").write_text(
                    json.dumps(_json_safe(evaluation_report), indent=2) + "\n", encoding="utf-8"
                )
    except Exception as exc:
        status.update({
            "status": "training_completed_test_evaluation_failed",
            "evaluation_directory": str(eval_dir),
            "evaluation_error": repr(exc),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        })
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        ultralytics_logger.removeHandler(eval_logger_handler)
        eval_logger_handler.close()

    training_report["status"] = "training_and_test_evaluation_completed"
    training_report["test_evaluation_report"] = str(eval_dir / "evaluation_report.json")
    (run_dir / "training_report.json").write_text(
        json.dumps(_json_safe(training_report), indent=2) + "\n", encoding="utf-8"
    )
    status.update({
        "status": "training_and_test_evaluation_completed",
        "evaluation_directory": str(eval_dir),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    })
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print("Training run directory:", run_dir)
    print("Best checkpoint:", best_path)
    print("Last checkpoint:", last_path)
    print("Training log:", log_path)
    print("Test evaluation:", eval_dir / "evaluation_report.json")
    print(json.dumps(_json_safe(evaluation_report), indent=2))
    return training_report


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/content/poshanscan_yolo_dataset"))
    parser.add_argument(
        "--training-runs-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/PoshanScanColab/training_runs"),
    )
    parser.add_argument("--weights", default="yolov8n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", type=Path, help="Resume from an existing Drive run's weights/last.pt.")
    parser.add_argument("--validate-only", action="store_true", help="Validate dataset/split metadata without importing Ultralytics or training.")
    args = parser.parse_args()
    if args.epochs < 1 or args.imgsz < 64:
        parser.error("epochs must be positive and imgsz at least 64")
    if args.seed != EXPECTED_SPLIT_SEED:
        parser.error(f"The existing dataset split seed is fixed at {EXPECTED_SPLIT_SEED}; do not regenerate or change it.")
    if not args.data_root.is_dir():
        parser.error(f"Dataset root does not exist: {args.data_root}. Extract the Colab ZIP first.")
    if args.validate_only:
        _, report = _validate_dataset(args.data_root.resolve())
        print(json.dumps(_json_safe(report), indent=2))
        print("Validation only: no model was loaded and no training was started.")
        return 0
    expected_training_root = Path("/content/drive/MyDrive/PoshanScanColab/training_runs").resolve()
    if args.training_runs_dir.resolve() != expected_training_root:
        parser.error(f"Training output must persist directly under {expected_training_root}.")
    if not args.training_runs_dir.parent.is_dir():
        parser.error("Google Drive is not mounted at /content/drive/MyDrive; mount it before training.")
    run_training(
        args.data_root.resolve(),
        args.training_runs_dir.resolve(),
        args.weights,
        args.epochs,
        args.imgsz,
        args.seed,
        args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
