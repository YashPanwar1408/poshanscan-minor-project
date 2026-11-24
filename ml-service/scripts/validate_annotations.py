"""Validate YOLO polygon labels and their image-file correspondence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "dataset" / "images"
LABEL_DIR = ROOT / "dataset" / "annotations" / "labels"
SPLIT_DIR = ROOT / "dataset" / "splits"
DEFAULT_JSON = ROOT / "dataset" / "reports" / "annotation_validation.json"
SUPPORTED = {".jpg", ".jpeg", ".png"}


def validate_polygon_line(line: str, line_number: int) -> tuple[bool, str]:
    fields = line.split()
    if len(fields) < 7:
        return False, f"line {line_number}: polygon needs class plus at least 3 coordinate pairs"
    if (len(fields) - 1) % 2:
        return False, f"line {line_number}: coordinates must be x/y pairs"
    try:
        class_id = int(fields[0])
    except ValueError:
        return False, f"line {line_number}: class ID must be an integer"
    if class_id != 0:
        return False, f"line {line_number}: expected class ID 0 (upper_arm), got {class_id}"
    try:
        coordinates = [float(value) for value in fields[1:]]
    except ValueError:
        return False, f"line {line_number}: coordinates must be numeric"
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in coordinates):
        return False, f"line {line_number}: every normalized coordinate must be finite and in [0, 1]"
    points = list(zip(coordinates[::2], coordinates[1::2]))
    if len(set(points)) < 3:
        return False, f"line {line_number}: polygon needs at least 3 distinct points"
    return True, ""


def validate_annotations(
    image_dir: Path = IMAGE_DIR,
    label_dir: Path = LABEL_DIR,
    split_dir: Optional[Path] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    images = sorted(
        (path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED),
        key=lambda path: path.name.lower(),
    ) if image_dir.is_dir() else []
    labels = {path.stem: path for path in label_dir.glob("*.txt")} if label_dir.is_dir() else {}
    all_image_names = {path.name for path in images}
    split_error = ""
    if split_dir is None:
        active_names = set(all_image_names)
    else:
        active_names: set[str] = set()
        split_entries = []
        for split_name in ("train", "val", "test"):
            split_path = split_dir / f"{split_name}.txt"
            if not split_path.is_file():
                split_error = f"Missing {split_name} split file: {split_path}"
                break
            entries = [line.strip().replace("\\", "/") for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not entries:
                split_error = f"The {split_name} split is empty: {split_path}"
                break
            split_entries.extend(entries)
            active_names.update(Path(entry).name for entry in entries)
        if not split_error:
            if len(split_entries) != len(set(split_entries)):
                split_error = "An image appears more than once across the train/val/test lists."
            unknown = sorted(active_names - all_image_names)
            if unknown:
                split_error = f"Split refers to images not found in the image directory: {unknown[:5]}"

    image_stems: dict[str, Path] = {}
    rows: list[dict[str, Any]] = []
    valid_count = 0
    missing_count = 0
    invalid_count = 0

    for image in images:
        stem = image.stem
        if stem in image_stems:
            rows.append({
                "image_filename": image.name,
                "label_filename": "",
                "status": "duplicate_image_stem",
                "polygon_count": 0,
                "error": f"Images {image_stems[stem].name} and {image.name} share a label basename.",
            })
            invalid_count += 1
            continue
        image_stems[stem] = image
        required_for_training = image.name in active_names
        label = labels.get(stem)
        if label is None:
            rows.append({
                "image_filename": image.name,
                "label_filename": "",
                "status": "missing_label" if required_for_training else "excluded_from_current_split",
                "polygon_count": 0,
                "error": "No matching YOLO polygon label file." if required_for_training else "Not in the current train/val/test split.",
            })
            if required_for_training:
                missing_count += 1
            continue
        try:
            lines = [line.strip() for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        except OSError as exc:
            lines = []
            errors = [f"Could not read label file: {exc}"]
        else:
            errors = []
        if not lines and not errors:
            errors.append("Label file is empty; annotate the visible upper arm or explicitly exclude this image.")
        for line_number, line in enumerate(lines, start=1):
            valid, message = validate_polygon_line(line, line_number)
            if not valid:
                errors.append(message)
        status = (
            ("valid" if required_for_training else "valid_but_not_in_current_split")
            if not errors
            else ("invalid_label" if required_for_training else "invalid_label_not_in_current_split")
        )
        rows.append({
            "image_filename": image.name,
            "label_filename": label.name,
            "status": status,
            "polygon_count": len(lines) if not errors else len(lines),
            "error": "; ".join(errors),
        })
        if errors:
            invalid_count += 1
        elif required_for_training:
            valid_count += 1

    extra_labels = sorted(set(labels) - set(image_stems))
    for stem in extra_labels:
        rows.append({
            "image_filename": "",
            "label_filename": labels[stem].name,
            "status": "orphan_label",
            "polygon_count": 0,
            "error": "Label has no image with the matching basename.",
        })
        invalid_count += 1

    active_count = len(active_names & all_image_names)
    excluded_unlabeled = sum(
        row["status"] == "excluded_from_current_split" for row in rows
    )
    ready = (
        bool(active_count)
        and not split_error
        and valid_count == active_count
        and missing_count == 0
        and invalid_count == 0
    )
    summary = {
        "image_count": len(images),
        "split_image_count": active_count,
        "images_outside_current_split_without_labels": excluded_unlabeled,
        "valid_annotated_images": valid_count,
        "missing_labels": missing_count,
        "invalid_annotations_or_orphan_labels": invalid_count,
        "split_error": split_error,
        "training_ready": ready,
        "class_id": 0,
        "class_name": "upper_arm",
        "label_format": "Ultralytics YOLO segmentation polygons, normalized coordinates",
        "note": (
            "Only images listed in the current splits require labels for training. Every provided polygon is checked, "
            "including labels for excluded images. No labels are synthesized or inferred from OpenCV output."
        ),
    }
    return summary, rows


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=IMAGE_DIR)
    parser.add_argument("--labels", type=Path, default=LABEL_DIR)
    parser.add_argument("--splits", type=Path, default=SPLIT_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary, rows = validate_annotations(_resolve(args.images), _resolve(args.labels), _resolve(args.splits))
    output_json = _resolve(args.output_json)
    output_csv = output_json.with_suffix(".csv")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_filename", "label_filename", "status", "polygon_count", "error"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    print(f"Wrote annotation report: {output_json} and {output_csv}")
    return 0 if summary["training_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
