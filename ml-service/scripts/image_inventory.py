"""Inventory and validate uploaded images without treating them as calibration data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES = ROOT / "dataset" / "images"
DEFAULT_JSON = ROOT / "dataset" / "reports" / "image_inventory.json"
SUPPORTED = {".jpg", ".jpeg", ".png"}
PLACEHOLDERS = {"README.md", ".gitkeep"}

if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))
from app.pipeline.reference_detect import detect_reference_marker  # noqa: E402


def _natural_key(path: Path, images_dir: Path) -> tuple[int, int | str, str]:
    relative = str(path.relative_to(images_dir)).lower()
    if path.stem.isdigit():
        return (0, int(path.stem), path.suffix.lower())
    return (1, relative, "")


def difference_hash(image: np.ndarray) -> int:
    """Small 64-bit image fingerprint for finding visually similar files."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(first: int, second: int) -> int:
    return (int(first) ^ int(second)).bit_count()


def inspect_images(images_dir: Path, marker_size_mm: float = 50.0) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    files = sorted(
        (path for path in images_dir.rglob("*") if path.is_file() and path.name not in PLACEHOLDERS),
        key=lambda path: _natural_key(path, images_dir),
    ) if images_dir.is_dir() else []
    seen_hashes: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    format_counts: Counter[str] = Counter()

    for path in files:
        relative = str(path.relative_to(images_dir))
        row: dict[str, Any] = {
            "filename": relative,
            "dataset_label": "synthetic_unverified",
            "file_extension": path.suffix.lower(),
            "file_format": "",
            "width_px": "",
            "height_px": "",
            "dimensions": "",
            "mean_brightness_0_255": "",
            "laplacian_variance_blur_proxy": "",
            "sha256": "",
            "duplicate_of": "",
            "exact_duplicate_group": "",
            "is_unique_image": "",
            "unique_canonical_filename": "",
            "difference_hash": "",
            "near_duplicate_group": "",
            "marker_status": "not_checked",
            "potentially_insufficient_arm_visibility": "",
            "visibility_review_reason": "",
            "validation_status": "invalid",
            "error": "",
        }
        if path.suffix.lower() not in SUPPORTED:
            row["error"] = f"Unsupported image extension: {path.suffix or '(none)'}; expected JPG, JPEG, or PNG."
            rows.append(row)
            continue

        try:
            payload = path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            row["sha256"] = digest
            row["duplicate_of"] = seen_hashes.get(digest, "")
            seen_hashes.setdefault(digest, relative)

            with Image.open(BytesIO(payload)) as source:
                actual_format = str(source.format or "").upper()
                source.verify()
            expected_format = "PNG" if path.suffix.lower() == ".png" else "JPEG"
            if actual_format != expected_format:
                raise ValueError(
                    f"Extension indicates {expected_format}, but file content is {actual_format or 'unknown'}."
                )
            decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None or decoded.size == 0:
                raise ValueError("OpenCV could not decode the image.")

            height, width = decoded.shape[:2]
            gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
            try:
                marker = detect_reference_marker(decoded, marker_size_mm)
                marker_status = "detected" if marker["detected"] else "not_detected"
            except Exception as exc:  # Marker diagnostics must not stop inventorying other files.
                marker_status = "error"
                row["error"] = f"Marker check failed: {exc}"
            visibility_review = marker_status != "detected" or min(width, height) < 640
            row.update(
                {
                    "file_format": actual_format,
                    "width_px": int(width),
                    "height_px": int(height),
                    "dimensions": f"{width}x{height}",
                    "mean_brightness_0_255": round(float(np.mean(gray)), 3),
                    "laplacian_variance_blur_proxy": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 3),
                    "difference_hash": f"{difference_hash(gray):016x}",
                    "marker_status": marker_status,
                    "potentially_insufficient_arm_visibility": visibility_review,
                    "visibility_review_reason": (
                        "No ArUco marker detected; manually review framing and visibility."
                        if marker_status != "detected"
                        else "Shorter image side is below 640 px; manually review arm detail."
                        if min(width, height) < 640
                        else "No automatic visibility issue detected; manual mask annotation review is still required."
                    ),
                    "validation_status": "valid",
                }
            )
            format_counts[actual_format] += 1
        except Exception as exc:
            row["error"] = str(exc) or "Image could not be read."
        rows.append(row)

    valid = [row for row in rows if row["validation_status"] == "valid"]
    exact_groups_by_hash: dict[str, list[dict[str, Any]]] = {}
    for row in valid:
        exact_groups_by_hash.setdefault(str(row["sha256"]), []).append(row)
    exact_duplicate_groups: list[dict[str, Any]] = []
    for group_index, group in enumerate(exact_groups_by_hash.values(), start=1):
        canonical = group[0]["filename"]
        group_id = f"EXACT-{group_index:04d}" if len(group) > 1 else ""
        aliases = [row["filename"] for row in group[1:]]
        for index, row in enumerate(group):
            row["exact_duplicate_group"] = group_id
            row["is_unique_image"] = index == 0
            row["unique_canonical_filename"] = canonical
        if aliases:
            exact_duplicate_groups.append(
                {
                    "group_id": group_id,
                    "canonical_filename": canonical,
                    "duplicate_filenames": aliases,
                }
            )

    near_pairs: list[dict[str, Any]] = []
    parent = list(range(len(valid)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    hashes = [int(row["difference_hash"], 16) for row in valid]
    for first in range(len(valid)):
        for second in range(first + 1, len(valid)):
            distance = hamming_distance(hashes[first], hashes[second])
            if distance <= 4:
                union(first, second)
                near_pairs.append(
                    {
                        "first": valid[first]["filename"],
                        "second": valid[second]["filename"],
                        "dhash_hamming_distance": distance,
                        "exact_duplicate": valid[first]["sha256"] == valid[second]["sha256"],
                    }
                )
    root_counts = Counter(find(index) for index in range(len(valid)))
    group_names: dict[int, str] = {}
    for index, row in enumerate(valid):
        root_index = find(index)
        if root_counts[root_index] > 1:
            group_names.setdefault(root_index, f"NEAR-{len(group_names) + 1:04d}")
            row["near_duplicate_group"] = group_names[root_index]

    unique_inventory: list[dict[str, Any]] = []
    for group in exact_groups_by_hash.values():
        canonical = group[0]
        unique_inventory.append(
            {
                "canonical_filename": canonical["filename"],
                "sha256": canonical["sha256"],
                "difference_hash": canonical["difference_hash"],
                "exact_duplicate_group": canonical["exact_duplicate_group"],
                "duplicate_count_including_canonical": len(group),
                "duplicate_filenames": ";".join(row["filename"] for row in group[1:]),
                "near_duplicate_group": canonical["near_duplicate_group"],
                "file_format": canonical["file_format"],
                "width_px": canonical["width_px"],
                "height_px": canonical["height_px"],
                "marker_status": canonical["marker_status"],
                "potentially_insufficient_arm_visibility": canonical[
                    "potentially_insufficient_arm_visibility"
                ],
            }
        )

    dhash_by_filename = {row["filename"]: int(row["difference_hash"], 16) for row in valid}
    exact_pairs = [
        {
            "first": first["filename"],
            "second": second["filename"],
            "dhash_hamming_distance": hamming_distance(
                dhash_by_filename[first["filename"]], dhash_by_filename[second["filename"]]
            ),
        }
        for group in exact_groups_by_hash.values()
        for index, first in enumerate(group)
        for second in group[index + 1:]
    ]

    summary = {
        "dataset_label": "synthetic_unverified",
        "images_directory": str(images_dir),
        "total_images": len(rows),
        "valid_images": len(valid),
        "invalid_images": len(rows) - len(valid),
        "duplicate_images": sum(bool(row["duplicate_of"]) for row in valid),
        "unique_image_count_after_exact_deduplication": len(unique_inventory),
        "exact_duplicate_group_count": len(exact_duplicate_groups),
        "exact_duplicate_pair_count": sum(
            len(group) * (len(group) - 1) // 2 for group in exact_groups_by_hash.values()
        ),
        "exact_duplicate_groups": exact_duplicate_groups,
        "exact_duplicate_pairs": exact_pairs,
        "near_duplicate_pair_count": len(near_pairs),
        "non_exact_near_duplicate_pair_count": sum(not pair["exact_duplicate"] for pair in near_pairs),
        "near_duplicate_group_count": len(group_names),
        "near_duplicate_pairs": near_pairs,
        "images_with_detectable_aruco_markers": sum(row["marker_status"] == "detected" for row in valid),
        "images_without_detectable_aruco_markers": sum(row["marker_status"] == "not_detected" for row in valid),
        "marker_check_errors": sum(row["marker_status"] == "error" for row in valid),
        "potentially_insufficient_arm_visibility_count": sum(
            bool(row["potentially_insufficient_arm_visibility"]) for row in valid
        ),
        "file_formats": dict(sorted(format_counts.items())),
        "dimensions_by_image": {row["filename"]: row["dimensions"] for row in valid},
        "unique_image_inventory": unique_inventory,
        "quality_metrics_note": (
            "Brightness, blur, duplicate, and visibility flags are screening heuristics only. "
            "They do not establish a correct arm mask, physical scale, or clinical image quality."
        ),
        "calibration_note": (
            "Synthetic/unverified images are inventoried only. Marker detection does not prove physical scale, "
            "and these images are not tape-labelled calibration samples."
        ),
    }
    return summary, rows


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--marker-size-mm", type=float, default=50.0)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    images_dir = _resolve(args.images)
    summary, rows = inspect_images(images_dir, args.marker_size_mm)
    json_path = _resolve(args.output_json)
    csv_path = json_path.with_suffix(".csv")
    unique_json_path = json_path.with_name("unique_image_inventory.json")
    unique_csv_path = json_path.with_name("unique_image_inventory.csv")
    duplicate_audit_path = json_path.with_name("duplicate_pair_audit.csv")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    unique_rows = summary.pop("unique_image_inventory", [])
    summary["unique_image_inventory_json"] = str(unique_json_path)
    summary["unique_image_inventory_csv"] = str(unique_csv_path)
    summary["duplicate_pair_audit_csv"] = str(duplicate_audit_path)
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fields = list(rows[0].keys()) if rows else [
        "filename", "dataset_label", "file_extension", "file_format", "width_px", "height_px",
        "dimensions", "mean_brightness_0_255", "laplacian_variance_blur_proxy", "sha256",
        "duplicate_of", "exact_duplicate_group", "is_unique_image", "unique_canonical_filename",
        "difference_hash", "near_duplicate_group", "marker_status",
        "potentially_insufficient_arm_visibility", "visibility_review_reason", "validation_status", "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    unique_fields = list(unique_rows[0].keys()) if unique_rows else [
        "canonical_filename", "sha256", "difference_hash", "exact_duplicate_group",
        "duplicate_count_including_canonical", "duplicate_filenames", "near_duplicate_group",
        "file_format", "width_px", "height_px", "marker_status",
        "potentially_insufficient_arm_visibility",
    ]
    unique_json_path.write_text(json.dumps(unique_rows, indent=2) + "\n", encoding="utf-8")
    with unique_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=unique_fields)
        writer.writeheader()
        writer.writerows(unique_rows)

    pair_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for pair in summary["exact_duplicate_pairs"]:
        key = tuple(sorted((pair["first"], pair["second"])))
        pair_rows[key] = {
            "first": key[0],
            "second": key[1],
            "exact_duplicate": True,
            "near_duplicate_by_dhash": pair["dhash_hamming_distance"] <= 4,
            "dhash_hamming_distance": pair["dhash_hamming_distance"],
        }
    for pair in summary["near_duplicate_pairs"]:
        key = tuple(sorted((pair["first"], pair["second"])))
        pair_rows[key] = {
            "first": key[0],
            "second": key[1],
            "exact_duplicate": pair["exact_duplicate"],
            "near_duplicate_by_dhash": True,
            "dhash_hamming_distance": pair["dhash_hamming_distance"],
        }
    with duplicate_audit_path.open("w", newline="", encoding="utf-8") as handle:
        pair_fields = ["first", "second", "exact_duplicate", "near_duplicate_by_dhash", "dhash_hamming_distance"]
        writer = csv.DictWriter(handle, fieldnames=pair_fields)
        writer.writeheader()
        writer.writerows(pair_rows.values())
    print(json.dumps(summary, indent=2))
    print(
        f"Wrote image inventory: {json_path}, {csv_path}, {unique_json_path}, "
        f"{unique_csv_path}, and {duplicate_audit_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
