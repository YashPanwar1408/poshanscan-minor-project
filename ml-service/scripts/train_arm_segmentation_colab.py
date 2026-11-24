"""Train/evaluate PoshanScan YOLOv8-seg inside Google Colab from the dataset ZIP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from importlib import metadata
from importlib.metadata import PackageNotFoundError
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
import yaml

EXPECTED_SPLITS = {"train": 84, "val": 24, "test": 12}
SPLIT_NAMES = tuple(EXPECTED_SPLITS)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _require_colab_torch():
    """Use Colab's installed PyTorch pair; never pip-install/upgrade either package."""
    try:
        import torch
        import torchvision
    except Exception as exc:
        raise RuntimeError(
            "Could not import Colab's installed torch/torchvision pair. Restart with a fresh Colab GPU runtime; "
            "this workflow intentionally does not replace those packages."
        ) from exc
    before = (torch.__version__, torchvision.__version__)
    try:
        # This exercises torchvision's compiled NMS operator without using any data.
        torchvision.ops.nms(
            torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0]]),
            torch.tensor([0.9, 0.8]),
            0.5,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Colab torch/torchvision ({before[0]}, {before[1]}) cannot run torchvision NMS. "
            "Use Runtime > Disconnect and delete runtime, then reconnect; do not install another torch pair."
        ) from exc
    return torch, torchvision, before


def _install_ultralytics_without_torch_changes(torch_versions: tuple[str, str]):
    """Install Ultralytics and only absent non-Torch requirements, all with --no-deps."""
    try:
        metadata.version("ultralytics")
    except PackageNotFoundError:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--no-deps", "ultralytics>=8.2,<9"
        ])

    from packaging.requirements import Requirement

    queue = ["ultralytics"]
    processed: set[str] = set()
    while queue:
        package_name = queue.pop(0)
        normalized_package = package_name.lower().replace("_", "-")
        if normalized_package in processed:
            continue
        processed.add(normalized_package)
        try:
            requirements = metadata.requires(package_name) or []
        except PackageNotFoundError as exc:
            raise RuntimeError(f"Package metadata is missing for {package_name}.") from exc

        for requirement_text in requirements:
            requirement = Requirement(requirement_text)
            if requirement.marker and not requirement.marker.evaluate():
                continue
            dependency = requirement.name
            normalized_dependency = dependency.lower().replace("_", "-")
            try:
                installed_version = metadata.version(dependency)
            except PackageNotFoundError:
                installed_version = None

            if normalized_dependency in {"torch", "torchvision"}:
                protected_version = torch_versions[0 if normalized_dependency == "torch" else 1]
                if installed_version != protected_version:
                    raise RuntimeError(f"The runtime's {dependency} changed unexpectedly; stopping before training.")
                if requirement.specifier and not requirement.specifier.contains(
                    installed_version, prereleases=True
                ):
                    raise RuntimeError(
                        f"Colab's existing {dependency} {installed_version} does not satisfy {requirement_text}. "
                        "The script will not replace Colab's PyTorch packages. Use a fresh compatible runtime."
                    )
                continue

            if installed_version is None:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--no-deps", requirement_text
                ])
            elif requirement.specifier and not requirement.specifier.contains(
                installed_version, prereleases=True
            ):
                raise RuntimeError(
                    f"Installed package {dependency}=={installed_version} conflicts with {requirement_text}. "
                    "No packages were upgraded automatically; restart with a fresh Colab runtime."
                )
            queue.append(dependency)

    if (metadata.version("torch"), metadata.version("torchvision")) != torch_versions:
        raise RuntimeError("PyTorch or torchvision changed during optional dependency setup.")
    from ultralytics import YOLO
    return YOLO


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            member = PurePosixPath(item.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"Unsafe path in dataset archive: {item.filename}")
            resolved = (destination / Path(*member.parts)).resolve()
            if root not in resolved.parents and resolved != root:
                raise ValueError(f"Archive path escapes extraction directory: {item.filename}")
        archive.extractall(destination)


def _read_and_check_splits(dataset_root: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    split_dir = dataset_root / "splits"
    split_entries: dict[str, list[str]] = {}
    image_to_split: dict[str, str] = {}
    for split_name, expected_count in EXPECTED_SPLITS.items():
        path = split_dir / f"{split_name}.txt"
        if not path.is_file():
            raise FileNotFoundError(path)
        entries = [line.strip().replace("\\", "/") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(entries) != expected_count or len(set(entries)) != expected_count:
            raise ValueError(f"Expected {expected_count} unique {split_name} entries, found {len(entries)}.")
        split_entries[split_name] = entries
        for entry in entries:
            if not entry.startswith("images/"):
                raise ValueError(f"Split entries must start with images/: {entry}")
            if entry in image_to_split:
                raise ValueError(f"Image appears in multiple splits: {entry}")
            image_to_split[entry] = split_name
    if len(image_to_split) != 120:
        raise ValueError(f"Expected 120 split images, found {len(image_to_split)}.")

    annotation_report = json.loads(
        (dataset_root / "reports" / "annotation_validation.json").read_text(encoding="utf-8")
    )
    split_report = json.loads(
        (dataset_root / "reports" / "split_report.json").read_text(encoding="utf-8")
    )
    if not annotation_report.get("training_ready") or annotation_report.get("valid_annotated_images") != 120:
        raise ValueError("The annotation report does not confirm 120 valid active labels.")
    if split_report.get("split_counts") != EXPECTED_SPLITS:
        raise ValueError(f"The split report counts differ: {split_report.get('split_counts')}")
    if not split_report.get("no_duplicate_or_near_duplicate_group_leakage"):
        raise ValueError("The split report identifies duplicate/source group leakage.")

    manifest_path = split_dir / "split_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        manifest = list(csv.DictReader(handle))
    groups: dict[str, set[str]] = {}
    manifest_images: set[str] = set()
    for row in manifest:
        filename = row.get("image_filename", "")
        split = row.get("split", "")
        group = row.get("duplicate_group", "")
        key = f"images/{filename}"
        if image_to_split.get(key) != split:
            raise ValueError(f"Split manifest does not match split lists for {filename}.")
        manifest_images.add(key)
        groups.setdefault(group, set()).add(split)
    if manifest_images != set(image_to_split):
        raise ValueError("The group manifest does not cover exactly the 120 split images.")
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("A duplicate/source group crosses split boundaries.")
    return split_entries, {name: hashlib.sha256((split_dir / f"{name}.txt").read_bytes()).hexdigest() for name in SPLIT_NAMES}


def _validate_polygon_file(label_path: Path) -> None:
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty polygon file: {label_path}")
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) < 7 or (len(fields) - 1) % 2:
            raise ValueError(f"Malformed polygon in {label_path}, line {line_number}.")
        if int(fields[0]) != 0:
            raise ValueError(f"Class ID must be 0 in {label_path}, line {line_number}.")
        coordinates = np.asarray([float(value) for value in fields[1:]], dtype=np.float64)
        if not np.isfinite(coordinates).all() or np.any(coordinates < 0) or np.any(coordinates > 1):
            raise ValueError(f"Invalid normalized coordinates in {label_path}, line {line_number}.")
        if len(np.unique(coordinates.reshape(-1, 2), axis=0)) < 3:
            raise ValueError(f"Polygon has fewer than three unique points: {label_path}, line {line_number}.")


def _build_yolo_layout(dataset_root: Path, destination: Path, split_entries: dict[str, list[str]]) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    label_root = dataset_root / "annotations" / "labels"
    hash_splits: dict[str, set[str]] = {}
    for split_name, entries in split_entries.items():
        image_dir = destination / "images" / split_name
        labels_dir = destination / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for entry in entries:
            source_image = dataset_root / entry
            source_label = label_root / f"{source_image.stem}.txt"
            if not source_image.is_file() or cv2.imread(str(source_image)) is None:
                raise FileNotFoundError(f"Missing or unreadable split image: {source_image}")
            if not source_label.is_file():
                raise FileNotFoundError(f"Missing label for {source_image.name}: {source_label}")
            _validate_polygon_file(source_label)
            digest = hashlib.sha256(source_image.read_bytes()).hexdigest()
            hash_splits.setdefault(digest, set()).add(split_name)
            shutil.copy2(source_image, image_dir / source_image.name)
            shutil.copy2(source_label, labels_dir / source_label.name)
    if len(hash_splits) != 120 or any(len(splits) != 1 for splits in hash_splits.values()):
        raise ValueError("An exact duplicate image is repeated in or leaks across the fixed splits.")

    yaml_path = destination / "dataset.yaml"
    yaml_path.write_text(
        yaml.safe_dump({
            "path": str(destination.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {0: "upper_arm"},
        }, sort_keys=False),
        encoding="utf-8",
    )
    return yaml_path


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
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
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
        if index >= len(classes) or classes[index] != 0:
            continue
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) >= 3 and np.isfinite(points).all():
            cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 255)
    return mask


def _pixel_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
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
        "iou": tp / union if union else 1.0,
        "dice": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0,
        "pixel_precision": tp / (tp + fp) if tp + fp else 0.0,
        "pixel_recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def _overlay(image: np.ndarray, mask: np.ndarray, label: str, color: tuple[int, int, int]) -> np.ndarray:
    tint = np.zeros_like(image)
    tint[mask > 0] = color
    output = cv2.addWeighted(image, 0.72, tint, 0.28, 0)
    output = cv2.resize(output, (360, 360))
    cv2.putText(output, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return output


def _train_and_evaluate(
    dataset_root: Path,
    output_root: Path,
    weights: str,
    epochs: int,
    imgsz: int,
    seed: int,
) -> dict[str, Any]:
    split_entries, split_signatures = _read_and_check_splits(dataset_root)
    yaml_path = _build_yolo_layout(dataset_root, output_root / "yolo_dataset", split_entries)
    torch, _torchvision, protected_versions = _require_colab_torch()
    YOLO = _install_ultralytics_without_torch_changes(protected_versions)

    gpu = bool(torch.cuda.is_available())
    device: int | str = 0 if gpu else "cpu"
    batch: int = -1 if gpu else 1
    workers = 2 if gpu else 0
    run_name = "yolov8n-seg-upper-arm-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root.mkdir(parents=True, exist_ok=True)
    config = {
        "data": str(yaml_path),
        "weights": weights,
        "pretrained": True,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
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
        "project": str(output_root / "runs"),
        "name": run_name,
    }

    model = YOLO(weights)
    if getattr(model, "task", None) != "segment":
        raise RuntimeError(f"Expected pretrained segmentation weights, got task={getattr(model, 'task', None)!r}.")
    print(f"Device: {torch.cuda.get_device_name(0) if gpu else 'CPU fallback'}; batch={batch}; imgsz={imgsz}")
    model.train(
        data=str(yaml_path),
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
        project=str(output_root / "runs"),
        name=run_name,
        exist_ok=False,
        plots=True,
        save=True,
        save_period=10,
    )

    trainer = getattr(model, "trainer", None)
    if trainer is None:
        raise RuntimeError("Ultralytics training returned without a trainer object.")
    run_dir = Path(trainer.save_dir)
    best_source = Path(trainer.best)
    last_source = Path(trainer.last)
    if not best_source.is_file() or not last_source.is_file():
        raise FileNotFoundError(f"Expected best.pt and last.pt under {run_dir / 'weights'}.")
    best_path = output_root / "best.pt"
    last_path = output_root / "last.pt"
    shutil.copy2(best_source, best_path)
    shutil.copy2(last_source, last_path)

    training_report = {
        "status": "completed",
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "split_counts": EXPECTED_SPLITS,
        "split_signatures_sha256": split_signatures,
        "training_image_count": 84,
        "validation_image_count": 24,
        "test_image_count_reserved": 12,
        "test_split_used_for_training": False,
        "runtime_versions": {
            "python": sys.version.split()[0],
            "torch": protected_versions[0],
            "torchvision": protected_versions[1],
            "ultralytics": metadata.version("ultralytics"),
        },
        "training_config": config,
        "validation_metrics_last_epoch": _json_safe(getattr(trainer, "metrics", {}) or {}),
        "run_directory": str(run_dir),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "limitations": "Synthetic validation data do not establish performance on real images or clinical MUAC accuracy.",
    }
    (output_root / "training_config.json").write_text(json.dumps(_json_safe(config), indent=2) + "\n", encoding="utf-8")
    (output_root / "training_report.json").write_text(json.dumps(_json_safe(training_report), indent=2) + "\n", encoding="utf-8")

    best_model = YOLO(str(best_path))
    if best_model.task != "segment":
        raise RuntimeError(f"FastAPI expects task=segment, got {best_model.task!r}.")
    class_names = best_model.names
    class_names = {int(key): value for key, value in class_names.items()} if isinstance(class_names, dict) else dict(enumerate(class_names))
    if class_names != {0: "upper_arm"}:
        raise RuntimeError(f"Expected checkpoint class map {{0: 'upper_arm'}}, got {class_names}.")
    compatibility = {
        "task": best_model.task,
        "names": class_names,
        "checkpoint": str(best_path),
        "ultralytics_version": metadata.version("ultralytics"),
        "torch_version": protected_versions[0],
        "torchvision_version": protected_versions[1],
    }
    (output_root / "checkpoint_compatibility.json").write_text(json.dumps(compatibility, indent=2) + "\n", encoding="utf-8")

    test_results = best_model.val(
        data=str(yaml_path),
        split="test",
        imgsz=imgsz,
        batch=batch,
        device=device,
        plots=True,
        project=str(output_root / "evaluation_runs"),
        name="held_out_test",
        exist_ok=True,
        verbose=False,
    )
    seg_metrics = getattr(test_results, "seg", None)
    test_map = {
        "mask_precision": _finite_float(getattr(seg_metrics, "mp", None)),
        "mask_recall": _finite_float(getattr(seg_metrics, "mr", None)),
        "map50": _finite_float(getattr(seg_metrics, "map50", None)),
        "map50_95": _finite_float(getattr(seg_metrics, "map", None)),
    }
    val_dir = Path(getattr(test_results, "save_dir", output_root / "evaluation_runs" / "held_out_test"))
    confusion_matrix = next(
        (str(path) for name in ("confusion_matrix.png", "MaskConfusionMatrix.png", "BoxConfusionMatrix.png")
         if (path := val_dir / name).is_file()),
        None,
    )

    visuals_dir = output_root / "test_visual_comparisons"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    label_root = dataset_root / "annotations" / "labels"
    rows: list[dict[str, Any]] = []
    totals = {"tp": 0, "fp": 0, "fn": 0}
    durations: list[float] = []
    failures = []
    for entry in split_entries["test"]:
        image_path = dataset_root / entry
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        height, width = image.shape[:2]
        truth = _truth_mask(label_root / f"{image_path.stem}.txt", width, height)
        if gpu:
            torch.cuda.synchronize()
        started = time.perf_counter()
        predictions = best_model.predict(source=image, imgsz=imgsz, conf=0.25, device=device, verbose=False)
        if gpu:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        durations.append(elapsed)
        predicted = _prediction_mask(predictions[0], width, height) if predictions else np.zeros_like(truth)
        metrics = _pixel_metrics(predicted, truth)
        for key in totals:
            totals[key] += metrics[key]
        rows.append({"image": image_path.name, "inference_seconds": elapsed, **metrics})
        if metrics["iou"] < 0.5 or not np.any(predicted):
            failures.append({"image": image_path.name, "iou": metrics["iou"], "reason": "low_iou_or_empty_prediction"})
        comparison = cv2.hconcat([
            _overlay(image, np.zeros_like(truth), "Original", (0, 0, 0)),
            _overlay(image, truth, "Ground truth", (0, 220, 0)),
            _overlay(image, predicted, "Prediction", (0, 0, 255)),
        ])
        cv2.imwrite(str(visuals_dir / f"{image_path.stem}_comparison.jpg"), comparison)

    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    union = tp + fp + fn
    mean_duration = float(np.mean(durations)) if durations else None
    report = {
        "evaluation_type": "held_out_test_split",
        "test_image_count": len(rows),
        "weights": str(best_path),
        "imgsz": imgsz,
        "ultralytics_mask_metrics": test_map,
        "pixel_metrics_aggregate": {
            "iou": tp / union if union else 1.0,
            "dice": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0,
            "pixel_precision": tp / (tp + fp) if tp + fp else 0.0,
            "pixel_recall": tp / (tp + fn) if tp + fn else 0.0,
        },
        "mean_image_iou": float(np.mean([row["iou"] for row in rows])) if rows else None,
        "mean_image_dice": float(np.mean([row["dice"] for row in rows])) if rows else None,
        "inference_speed": {
            "mean_seconds_per_image": mean_duration,
            "median_seconds_per_image": float(np.median(durations)) if durations else None,
            "images_per_second": 1.0 / mean_duration if mean_duration and mean_duration > 0 else None,
            "timing_scope": "predict per image; excludes model loading and image decoding",
        },
        "confusion_matrix": confusion_matrix,
        "per_image_metrics_csv": str(output_root / "per_image_metrics.csv"),
        "visual_comparisons_directory": str(visuals_dir),
        "failure_cases": failures,
        "limitations": "AI-generated test images do not establish real-world or clinical MUAC accuracy.",
    }
    with (output_root / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["image", "inference_seconds", "iou", "dice", "pixel_precision", "pixel_recall", "tp", "fp", "fn"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "evaluation_report.json").write_text(json.dumps(_json_safe(report), indent=2) + "\n", encoding="utf-8")

    actual_signatures = {
        name: hashlib.sha256((dataset_root / "splits" / f"{name}.txt").read_bytes()).hexdigest()
        for name in SPLIT_NAMES
    }
    if actual_signatures != split_signatures:
        raise RuntimeError("A source split file changed during the run; refusing to package results.")

    archive_path = output_root.parent / "poshanscan_colab_artifacts.zip"
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_root.parent).as_posix())
    report["artifact_archive"] = str(archive_path)
    print(json.dumps(_json_safe(report), indent=2))
    print("Artifact bundle:", archive_path)
    print("Download the checkpoint in Colab with: from google.colab import files; files.download('/content/poshanscan_colab_output/best.pt')")
    print("Download all checkpoints, logs, reports and visuals with: from google.colab import files; files.download('/content/poshanscan_colab_artifacts.zip')")
    return report


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-zip", type=Path, required=True, help="Prepared poshanscan_colab_dataset.zip")
    parser.add_argument("--output-dir", type=Path, default=Path("/content/poshanscan_colab_output"))
    parser.add_argument("--weights", default="yolov8n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.epochs < 1 or args.imgsz < 64:
        parser.error("--epochs must be positive and --imgsz must be at least 64")
    if not args.dataset_zip.is_file():
        parser.error(f"Dataset ZIP not found: {args.dataset_zip}")
    output_root = args.output_dir.resolve()
    input_root = output_root.parent / "poshanscan_colab_input"
    if input_root.exists():
        shutil.rmtree(input_root)
    _safe_extract(args.dataset_zip, input_root)
    dataset_root = input_root / "dataset"
    if not dataset_root.is_dir():
        parser.error("ZIP must contain dataset/images, dataset/annotations/labels, dataset/splits and dataset/reports")
    _train_and_evaluate(dataset_root, output_root, args.weights, args.epochs, args.imgsz, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
