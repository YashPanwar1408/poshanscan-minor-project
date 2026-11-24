"""Export only the existing YOLO split images, labels, and a Colab-ready YAML."""

from __future__ import annotations

import hashlib
import csv
import json
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET = ROOT / "dataset"
YOLO_ROOT = DATASET / "yolo_working"
OUTPUT = ROOT / "results" / "colab" / "poshanscan_yolo_dataset.zip"
EXPECTED = {"train": 84, "val": 24, "test": 12}
EXPECTED_SEED = 2026
COLAB_ROOT = "/content/poshanscan_yolo_dataset"


def build_archive(output: Path = OUTPUT) -> dict[str, object]:
    staged_config = YOLO_ROOT / "dataset.yaml"
    if not staged_config.is_file():
        raise FileNotFoundError(staged_config)

    split_report_path = DATASET / "reports" / "split_report.json"
    split_report = json.loads(split_report_path.read_text(encoding="utf-8"))
    if split_report.get("seed") != EXPECTED_SEED:
        raise ValueError(f"Expected split seed {EXPECTED_SEED}, got {split_report.get('seed')!r}.")
    if split_report.get("split_counts") != EXPECTED:
        raise ValueError(f"Expected split counts {EXPECTED}, got {split_report.get('split_counts')!r}.")
    if not split_report.get("no_duplicate_or_near_duplicate_group_leakage"):
        raise ValueError("The current split report indicates duplicate/source leakage.")
    filename_split: dict[str, str] = {}
    for split, expected_count in EXPECTED.items():
        split_file = DATASET / "splits" / f"{split}.txt"
        split_entries = [line.strip().replace("\\", "/") for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(split_entries) != expected_count:
            raise ValueError(f"The {split} list does not have the expected {expected_count} rows.")
        filename_split.update({Path(entry).name: split for entry in split_entries})
    if len(filename_split) != sum(EXPECTED.values()):
        raise ValueError("An image appears in more than one fixed split list.")
    with (DATASET / "splits" / "split_manifest.csv").open(newline="", encoding="utf-8-sig") as handle:
        manifest_rows = list(csv.DictReader(handle))
    group_splits: dict[str, set[str]] = {}
    for row in manifest_rows:
        filename = row.get("image_filename", "")
        split = row.get("split", "")
        group = row.get("duplicate_group", "")
        if filename_split.get(filename) != split:
            raise ValueError(f"Split manifest differs from fixed split lists for {filename}.")
        group_splits.setdefault(group, set()).add(split)
    if set(filename_split) != {row.get("image_filename", "") for row in manifest_rows}:
        raise ValueError("Split manifest does not cover exactly the current split lists.")
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("A duplicate/source group crosses split boundaries.")

    metadata_sources = [
        ("split_metadata/split_manifest.csv", DATASET / "splits" / "split_manifest.csv"),
        ("split_metadata/split_report.json", split_report_path),
        *[
            (f"split_metadata/{split}.txt", DATASET / "splits" / f"{split}.txt")
            for split in EXPECTED
        ],
    ]
    for _, source in metadata_sources:
        if not source.is_file():
            raise FileNotFoundError(source)

    files: list[tuple[str, Path, str]] = []
    for split, expected_count in EXPECTED.items():
        split_file = DATASET / "splits" / f"{split}.txt"
        entries = [line.strip().replace("\\", "/") for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(entries) != expected_count or len(set(entries)) != expected_count:
            raise ValueError(f"Existing {split} split must contain {expected_count} unique entries.")

        staged_images = sorted(path for path in (YOLO_ROOT / "images" / split).iterdir() if path.is_file())
        staged_labels = sorted(path for path in (YOLO_ROOT / "labels" / split).glob("*.txt") if path.is_file())
        if len(staged_images) != expected_count or len(staged_labels) != expected_count:
            raise ValueError(
                f"Staged {split} files differ from the existing split: "
                f"images={len(staged_images)}, labels={len(staged_labels)}."
            )

        expected_names = {Path(entry).name for entry in entries}
        if {path.name for path in staged_images} != expected_names:
            raise ValueError(f"Staged {split} image names do not match its split list.")
        if {path.stem for path in staged_labels} != {Path(entry).stem for entry in entries}:
            raise ValueError(f"Staged {split} label names do not match its split list.")

        for entry in entries:
            original_image = DATASET / entry
            staged_image = YOLO_ROOT / "images" / split / Path(entry).name
            original_label = DATASET / "annotations" / "labels" / f"{Path(entry).stem}.txt"
            staged_label = YOLO_ROOT / "labels" / split / original_label.name
            if hashlib.sha256(original_image.read_bytes()).digest() != hashlib.sha256(staged_image.read_bytes()).digest():
                raise ValueError(f"Staged image was changed: {staged_image}")
            if original_label.read_bytes() != staged_label.read_bytes():
                raise ValueError(f"Staged label differs from its source: {staged_label}")
            image_arcname = f"images/{split}/{staged_image.name}"
            label_arcname = f"labels/{split}/{staged_label.name}"
            files.append((image_arcname, staged_image, ZIP_STORED))
            files.append((label_arcname, staged_label, ZIP_DEFLATED))

    dataset_yaml = (
        f"path: {COLAB_ROOT}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: upper_arm\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w") as archive:
        for archive_name, source, compression in files:
            archive.write(source, archive_name, compress_type=compression)
        archive.writestr("dataset.yaml", dataset_yaml, compress_type=ZIP_DEFLATED)
        for archive_name, source in metadata_sources:
            archive.write(source, archive_name, compress_type=ZIP_DEFLATED)

    expected_file_count = sum(EXPECTED.values()) * 2 + 1 + len(metadata_sources)
    with ZipFile(output) as archive:
        names = archive.namelist()
        if len(names) != expected_file_count or len(set(names)) != expected_file_count:
            raise ValueError(f"Expected {expected_file_count} archive entries, got {len(names)}.")
        allowed_prefixes = ("images/", "labels/", "split_metadata/")
        unexpected = [name for name in names if not (name == "dataset.yaml" or name.startswith(allowed_prefixes))]
        if unexpected:
            raise ValueError(f"Unexpected file in Colab dataset ZIP: {unexpected[0]}")
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"Archive integrity check failed at {bad_member}")

    return {
        "archive": str(output.resolve()),
        "split_counts": EXPECTED,
        "class_mapping": {0: "upper_arm"},
        "archive_entries": expected_file_count,
        "split_seed": EXPECTED_SEED,
        "split_metadata_files": len(metadata_sources),
        "contains_only_dataset_and_split_audit_files": True,
        "split_lists_modified": False,
        "size_bytes": output.stat().st_size,
    }


def main() -> int:
    print(json.dumps(build_archive(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
