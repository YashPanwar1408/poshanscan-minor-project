"""Create reproducible 70/20/10 splits without separating duplicate groups."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "dataset" / "images"
SPLIT_DIR = ROOT / "dataset" / "splits"
REPORT_PATH = ROOT / "dataset" / "reports" / "split_report.json"
EXCLUSIONS_PATH = ROOT / "dataset" / "annotations" / "excluded_images.csv"
SUPPORTED = {".jpg", ".jpeg", ".png"}

if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))


def hamming_distance(first: int, second: int) -> int:
    return (int(first) ^ int(second)).bit_count()


def _cached_hashes(images: list[Path]) -> dict[str, tuple[str, int]]:
    """Reuse the image inventory's verified hashes, decoding only changed files."""
    inventory_path = ROOT / "dataset" / "reports" / "image_inventory.csv"
    inventory_rows: dict[str, dict[str, str]] = {}
    if inventory_path.is_file():
        with inventory_path.open(newline="", encoding="utf-8-sig") as handle:
            inventory_rows = {row.get("filename", ""): row for row in csv.DictReader(handle)}

    hashes: dict[str, tuple[str, int]] = {}
    needs_decode = []
    for path in images:
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        row = inventory_rows.get(path.name)
        if row and row.get("sha256") == digest and row.get("difference_hash"):
            hashes[path.name] = (digest, int(row["difference_hash"], 16))
        else:
            needs_decode.append((path, payload, digest))

    if needs_decode:
        import cv2
        import numpy as np

        from scripts.image_inventory import difference_hash

        for path, payload, digest in needs_decode:
            decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if decoded is None:
                raise ValueError(f"Could not decode image: {path}")
            hashes[path.name] = (digest, difference_hash(decoded))
    return hashes


def _natural_key(path: Path) -> tuple[int, str]:
    return (int(path.stem), path.suffix.lower()) if path.stem.isdigit() else (10**12, path.name.lower())


def _normalized_source_stem(original_name: str) -> str:
    stem = Path(original_name).stem.lower()
    return re.sub(r"\s*\(\d+\)$", "", stem).strip()


def discover_duplicate_groups(
    images: list[Path],
    *,
    near_threshold: int = 4,
    rename_map_path: Optional[Path] = None,
) -> tuple[list[list[Path]], list[dict[str, Any]]]:
    if near_threshold < 0 or near_threshold > 16:
        raise ValueError("near_threshold must be between 0 and 16.")
    parent = list(range(len(images)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    image_hashes = _cached_hashes(images)
    hashes = [image_hashes[image_path.name][0] for image_path in images]
    dhashes = [image_hashes[image_path.name][1] for image_path in images]

    old_names: dict[str, str] = {}
    if rename_map_path and rename_map_path.is_file():
        with rename_map_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                old_names[row.get("new_filename", "")] = row.get("original_filename", "")

    reasons: list[dict[str, Any]] = []
    for first in range(len(images)):
        for second in range(first + 1, len(images)):
            exact = hashes[first] == hashes[second]
            distance = hamming_distance(dhashes[first], dhashes[second])
            first_source = old_names.get(images[first].name, "")
            second_source = old_names.get(images[second].name, "")
            source_match = bool(first_source and second_source) and (
                _normalized_source_stem(first_source) == _normalized_source_stem(second_source)
            )
            if exact or distance <= near_threshold or source_match:
                union(first, second)
                reasons.append({
                    "first": images[first].name,
                    "second": images[second].name,
                    "exact_duplicate": exact,
                    "dhash_hamming_distance": distance,
                    "same_source_name_group": source_match,
                })

    groups_by_root: dict[int, list[Path]] = {}
    for index, image_path in enumerate(images):
        groups_by_root.setdefault(find(index), []).append(image_path)
    return list(groups_by_root.values()), reasons


def assign_splits(groups: list[list[Path]], seed: int = 2026) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda group: -len(group))
    total = sum(map(len, shuffled))
    targets = {"train": total * 0.70, "val": total * 0.20, "test": total * 0.10}
    splits: dict[str, list[Path]] = {name: [] for name in targets}
    for group in shuffled:
        deficits = {name: targets[name] - len(splits[name]) for name in targets}
        best_deficit = max(deficits.values())
        candidates = [name for name, deficit in deficits.items() if deficit == best_deficit]
        selected = rng.choice(candidates)
        splits[selected].extend(group)
    for name in splits:
        splits[name].sort(key=_natural_key)
    return splits


def create_splits(
    image_dir: Path = IMAGE_DIR,
    split_dir: Path = SPLIT_DIR,
    *,
    seed: int = 2026,
    near_threshold: int = 4,
    rename_map_path: Optional[Path] = None,
) -> tuple[dict[str, Any], dict[str, list[Path]]]:
    images = sorted(
        (path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED),
        key=_natural_key,
    ) if image_dir.is_dir() else []
    if not images:
        raise ValueError(f"No supported images found in {image_dir}")
    exact_groups: dict[str, list[Path]] = {}
    for image_path in images:
        exact_groups.setdefault(hashlib.sha256(image_path.read_bytes()).hexdigest(), []).append(image_path)
    images_by_name = {path.name.casefold(): path for path in images}
    exclusions: dict[str, str] = {}
    if EXCLUSIONS_PATH.is_file():
        with EXCLUSIONS_PATH.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                filename = (row.get("filename") or "").strip()
                reason = (row.get("reason") or "").strip()
                if not filename and not reason:
                    continue
                if not filename or not reason:
                    raise ValueError("Every exclusion row needs both filename and reason.")
                if filename.casefold() not in images_by_name:
                    raise ValueError(f"Excluded image is not in the dataset: {filename}")
                exclusions[images_by_name[filename.casefold()].name] = reason

    excluded_hashes = {
        digest
        for digest, group in exact_groups.items()
        if any(path.name in exclusions for path in group)
    }
    unique_images = [group[0] for digest, group in exact_groups.items() if digest not in excluded_hashes]
    annotation_exclusions = []
    for digest, group in exact_groups.items():
        if digest not in excluded_hashes:
            continue
        trigger = next(path for path in group if path.name in exclusions)
        trigger_reason = exclusions[trigger.name]
        annotation_exclusions.extend(
            {
                "filename": path.name,
                "reason": exclusions.get(
                    path.name,
                    f"Exact duplicate of {trigger.name}; excluded because: {trigger_reason}",
                ),
            }
            for path in group
        )
    excluded_exact_duplicates = [
        {"canonical_filename": group[0].name, "duplicate_filename": alias.name}
        for group in exact_groups.values()
        for alias in group[1:]
    ]

    groups, duplicate_pairs = discover_duplicate_groups(
        unique_images,
        near_threshold=near_threshold,
        rename_map_path=rename_map_path,
    )
    splits = assign_splits(groups, seed)
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_name, paths in splits.items():
        (split_dir / f"{split_name}.txt").write_text(
            "".join(f"images/{path.name}\n" for path in paths), encoding="utf-8"
        )

    group_id_by_name = {
        path.name: f"GROUP-{group_index:04d}"
        for group_index, group in enumerate(groups, start=1)
        for path in group
    }
    split_by_name = {path.name: split_name for split_name, paths in splits.items() for path in paths}
    group_split_leaks = []
    for pair in duplicate_pairs:
        first_split = split_by_name[pair["first"]]
        second_split = split_by_name[pair["second"]]
        if first_split != second_split:
            group_split_leaks.append({**pair, "first_split": first_split, "second_split": second_split})
    groups_by_split = {
        name: sum(1 for group in groups if group and split_by_name[group[0].name] == name)
        for name in splits
    }
    manifest_path = split_dir / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_filename", "split", "duplicate_group"])
        writer.writeheader()
        for image_path in unique_images:
            writer.writerow({
                "image_filename": image_path.name,
                "split": split_by_name[image_path.name],
                "duplicate_group": group_id_by_name[image_path.name],
            })
    excluded_path = split_dir / "excluded_exact_duplicates.csv"
    with excluded_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["canonical_filename", "duplicate_filename"])
        writer.writeheader()
        writer.writerows(excluded_exact_duplicates)
    exclusions_report_path = split_dir / "excluded_annotation_images.csv"
    with exclusions_report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "reason"])
        writer.writeheader()
        writer.writerows(annotation_exclusions)

    annotation_dir = ROOT / "dataset" / "annotations" / "labels"
    annotated_count = sum((annotation_dir / f"{image.stem}.txt").is_file() for image in unique_images)
    report = {
        "source_image_count": len(images),
        "unique_image_count_after_exact_deduplication": len(exact_groups),
        "unique_image_count_in_active_splits": len(unique_images),
        "excluded_annotation_image_count": len(annotation_exclusions),
        "excluded_annotation_unique_group_count": len(excluded_hashes),
        "excluded_annotation_images": annotation_exclusions,
        "excluded_exact_duplicate_count": len(excluded_exact_duplicates),
        "excluded_exact_duplicates": excluded_exact_duplicates,
        "image_count": len(unique_images),
        "seed": seed,
        "near_duplicate_dhash_threshold": near_threshold,
        "duplicate_or_source_group_count": len(groups),
        "near_duplicate_or_source_pairs": duplicate_pairs,
        "groups_per_split": groups_by_split,
        "split_counts": {name: len(paths) for name, paths in splits.items()},
        "no_duplicate_or_near_duplicate_group_leakage": not group_split_leaks,
        "group_leakage_pairs": group_split_leaks,
        "annotated_image_count": annotated_count,
        "all_split_images_have_label_files": annotated_count == len(unique_images),
        "note": (
            "Exact duplicate aliases are preserved in dataset/images but omitted from the split to keep one copy. "
            "Manually excluded ambiguous images are preserved but omitted when their filename and review reason are "
            "recorded in dataset/annotations/excluded_images.csv. "
            "Near-duplicate and identifiable source groups remain together. "
            + (
                "All active split images have label files; scripts/validate_annotations.py checks polygon validity."
                if annotated_count == len(unique_images)
                else "Splits are provisional until each included image has a reviewed polygon."
            )
        ),
    }
    return report, splits


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--near-threshold", type=int, default=4)
    parser.add_argument("--images", type=Path, default=IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--report-json", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)
    image_dir = args.images if args.images.is_absolute() else ROOT / args.images
    split_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    report_path = args.report_json if args.report_json.is_absolute() else ROOT / args.report_json
    report, _ = create_splits(
        image_dir,
        split_dir,
        seed=args.seed,
        near_threshold=args.near_threshold,
        rename_map_path=ROOT / "dataset" / "annotations" / "image_rename_map.csv",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote split files to {split_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
