"""Train the optional YOLOv8-seg upper-arm model after labels are reviewed."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_annotations import LABEL_DIR, validate_annotations  # noqa: E402
from scripts.yolo_dataset import build_yolo_layout  # noqa: E402

DEFAULT_WEIGHTS = "yolov8n-seg.pt"
DEFAULT_OUTPUT = ROOT / "results" / "arm_segmentation"
DEFAULT_BEST = ROOT / "app" / "models" / "arm_segmentation" / "best.pt"


def _split_counts() -> tuple[dict[str, int], Optional[str]]:
    split_dir = ROOT / "dataset" / "splits"
    entries: dict[str, list[str]] = {}
    for name in ("train", "val", "test"):
        path = split_dir / f"{name}.txt"
        if not path.is_file():
            return {}, f"Missing split file {path}; run scripts/split_dataset.py first."
        entries[name] = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not entries[name]:
            return {}, f"Split {name} is empty; regenerate the splits after dataset inventory."
    seen: set[str] = set()
    for name, rows in entries.items():
        duplicate = seen.intersection(rows)
        if duplicate:
            return {}, f"Images appear in multiple splits ({name}): {sorted(duplicate)[:3]}"
        seen.update(rows)
    manifest = split_dir / "split_manifest.csv"
    if manifest.is_file():
        import csv

        group_split: dict[str, str] = {}
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                group = row.get("duplicate_group", "")
                split = row.get("split", "")
                if group in group_split and group_split[group] != split:
                    return {}, f"Duplicate group {group} leaks across splits."
                group_split[group] = split
    return {name: len(rows) for name, rows in entries.items()}, None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Pretrained segmentation checkpoint, default yolov8n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--best-model", type=Path, default=DEFAULT_BEST)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.imgsz < 64:
        print("epochs must be positive and imgsz must be at least 64.", file=sys.stderr)
        return 2

    validation, _ = validate_annotations(split_dir=ROOT / "dataset" / "splits")
    if not validation["training_ready"]:
        print(
            "Training stopped: reviewed ground-truth upper-arm polygon labels are missing or invalid. "
            f"Images={validation['image_count']}, valid_labels={validation['valid_annotated_images']}, "
            f"missing={validation['missing_labels']}, invalid={validation['invalid_annotations_or_orphan_labels']}. "
            "Annotate the images with scripts/validate_annotations.py guidance, then validate again. "
            "No masks or labels were generated automatically.",
            file=sys.stderr,
        )
        return 2

    split_counts, split_error = _split_counts()
    if split_error:
        print(f"Training stopped: {split_error}", file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "Ultralytics is not installed. Install optional segmentation dependencies with "
            "'pip install -r requirements-segmentation.txt'.",
            file=sys.stderr,
        )
        return 2

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    best_model = args.best_model if args.best_model.is_absolute() else ROOT / args.best_model
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="poshanscan-yolo-data-") as temp_name:
        data_yaml = build_yolo_layout(Path(temp_name))
        model = YOLO(args.weights)
        import torch

        cuda_available = bool(torch.cuda.is_available())
        device = 0 if cuda_available else "cpu"
        batch_size: int | str = -1 if cuda_available else 1
        training_config = {
            "data": str(data_yaml),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": batch_size,
            "device": device,
            "patience": 20,
            "seed": args.seed,
            "deterministic": True,
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
            "workers": 0,
            "project": str(output_dir / "runs"),
            "name": "yolov8n-seg-upper-arm",
            "exist_ok": False,
            "plots": True,
            "save": True,
        }
        model.train(**training_config)

    trainer: Any = getattr(model, "trainer", None)
    run_dir = Path(getattr(trainer, "save_dir", output_dir / "runs" / "yolov8n-seg-upper-arm"))
    best_source = Path(getattr(trainer, "best", run_dir / "weights" / "best.pt"))
    last_source = Path(getattr(trainer, "last", run_dir / "weights" / "last.pt"))
    if not best_source.is_file():
        print(f"Training finished but best checkpoint was not found at {best_source}.", file=sys.stderr)
        return 1
    shutil.copy2(best_source, best_model)
    last_model = best_model.with_name("last.pt")
    if last_source.is_file():
        shutil.copy2(last_source, last_model)

    raw_metrics = getattr(trainer, "metrics", {}) or {}
    metrics = {
        str(key): float(value)
        for key, value in raw_metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    report = {
        "dataset_provenance": "AI-generated synthetic images; manually reviewed polygons required",
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "training_image_count": split_counts["train"],
        "validation_image_count": split_counts["val"],
        "test_image_count_reserved": split_counts["test"],
        "test_split_used_for_training": False,
        "seed": args.seed,
        "pretrained_checkpoint": args.weights,
        "best_checkpoint": str(best_model),
        "last_checkpoint": str(last_model) if last_model.is_file() else None,
        "imgsz": args.imgsz,
        "epochs_requested": args.epochs,
        "device": "cuda:0" if cuda_available else "cpu",
        "batch_setting": batch_size,
        "early_stopping_patience": 20,
        "training_configuration": {
            key: value for key, value in training_config.items() if key not in {"data", "project"}
        },
        "validation_metrics_last_epoch": metrics,
        "training_run_directory": str(run_dir),
        "limitations": "Synthetic segmentation metrics are not clinical validation and do not validate MUAC estimation.",
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(report["training_configuration"], indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
