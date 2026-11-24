"""Package the unchanged PoshanScan splits and labels for Colab upload."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_annotations import validate_annotations  # noqa: E402

DATASET = ROOT / "dataset"
OUTPUT = ROOT / "results" / "colab" / "poshanscan_colab_dataset.zip"
SPLITS = ("train", "val", "test")
EXPECTED_COUNTS = {"train": 84, "val": 24, "test": 12}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MANIFEST_FILES = (
    "dataset/splits/train.txt",
    "dataset/splits/val.txt",
    "dataset/splits/test.txt",
    "dataset/splits/split_manifest.csv",
    "dataset/splits/excluded_exact_duplicates.csv",
    "dataset/splits/excluded_annotation_images.csv",
    "dataset/reports/split_report.json",
    "dataset/reports/annotation_validation.json",
)


def _active_image_entries() -> dict[str, list[str]]:
    entries: dict[str, list[str]] = {}
    seen: set[str] = set()
    for split in SPLITS:
        path = DATASET / "splits" / f"{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing split file: {path}")
        names = [line.strip().replace("\\", "/") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(names) != EXPECTED_COUNTS[split]:
            raise ValueError(f"Expected {EXPECTED_COUNTS[split]} {split} entries, found {len(names)}.")
        overlap = seen.intersection(names)
        if overlap:
            raise ValueError(f"Images leak between splits: {sorted(overlap)[:3]}")
        seen.update(names)
        entries[split] = names
    return entries


def build_archive(output: Path = OUTPUT) -> dict[str, object]:
    validation, _ = validate_annotations(split_dir=DATASET / "splits")
    if not validation.get("training_ready"):
        raise ValueError("Colab export refused because split annotations are incomplete or invalid.")

    report_path = DATASET / "reports" / "split_report.json"
    split_report = json.loads(report_path.read_text(encoding="utf-8"))
    if not split_report.get("no_duplicate_or_near_duplicate_group_leakage"):
        raise ValueError("Colab export refused because the split report shows duplicate/source-group leakage.")

    entries = _active_image_entries()
    exact_hash_splits: dict[str, set[str]] = {}
    for split, split_entries in entries.items():
        for entry in split_entries:
            image_path = DATASET / entry
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing split image: {image_path}")
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            exact_hash_splits.setdefault(digest, set()).add(split)
    if any(len(splits) != 1 for splits in exact_hash_splits.values()):
        raise ValueError("An exact duplicate appears in more than one split.")

    filename_to_split = {
        Path(entry).name: split
        for split, split_entries in entries.items()
        for entry in split_entries
    }
    manifest_path = DATASET / "splits" / "split_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    group_splits: dict[str, set[str]] = {}
    manifest_names: set[str] = set()
    for row in rows:
        filename = row.get("image_filename", "")
        split = row.get("split", "")
        group = row.get("duplicate_group", "")
        if filename not in filename_to_split or filename_to_split[filename] != split:
            raise ValueError(f"Split manifest disagrees with split list for {filename!r}.")
        manifest_names.add(filename)
        group_splits.setdefault(group, set()).add(split)
    if manifest_names != set(filename_to_split):
        raise ValueError("Split manifest does not cover exactly the active split images.")
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("A duplicate/source group appears in multiple splits.")

    image_files = sorted({DATASET / entry for values in entries.values() for entry in values})
    source_image_count = sum(
        path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        for path in (DATASET / "images").iterdir()
    )
    active_labels = sorted(
        DATASET / "annotations" / "labels" / f"{Path(entry).stem}.txt"
        for split_entries in entries.values()
        for entry in split_entries
    )
    if source_image_count != 131 or len(image_files) != 120 or len(active_labels) != 120:
        raise ValueError(
            f"Expected 131 local originals, 120 active split images, and 120 active labels; "
            f"found {source_image_count}, {len(image_files)}, and {len(active_labels)}."
        )
    if len(exact_hash_splits) != len(image_files):
        raise ValueError("An exact duplicate is still present in the active split images.")
    missing = [path for path in active_labels if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing active label file: {missing[0]}")

    readme = (
        "PoshanScan Colab segmentation dataset bundle\n"
        "\n"
        "Contents: the 120 active unique split images, their 120 YOLO polygon labels, the unchanged 84/24/12 split lists,\n"
        "the duplicate-group manifest, and the validation/split reports. The 11 exact duplicate aliases are preserved\n"
        "in the local project but omitted from this training bundle because no split uses them. The notebook constructs\n"
        "a Colab-local YOLO dataset.yaml; do not regenerate or edit the split files in Colab. Class: 0 = upper_arm.\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w") as archive:
        for image_path in image_files:
            archive.write(image_path, image_path.relative_to(ROOT).as_posix(), compress_type=ZIP_STORED)
        for label_path in active_labels:
            archive.write(label_path, label_path.relative_to(ROOT).as_posix(), compress_type=ZIP_DEFLATED)
        for relative in MANIFEST_FILES:
            path = ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(f"Missing required metadata file: {path}")
            archive.write(path, path.relative_to(ROOT).as_posix(), compress_type=ZIP_DEFLATED)
        archive.writestr("dataset/COLAB_README.txt", readme, compress_type=ZIP_DEFLATED)

    return {
        "archive": str(output.resolve()),
        "source_image_count": source_image_count,
        "uploaded_active_image_count": len(image_files),
        "active_label_count": len(active_labels),
        "excluded_exact_duplicate_aliases_not_uploaded": source_image_count - len(image_files),
        "split_counts": {name: len(values) for name, values in entries.items()},
        "split_lists_modified": False,
        "bytes": output.stat().st_size,
    }


def main() -> int:
    print(json.dumps(build_archive(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
