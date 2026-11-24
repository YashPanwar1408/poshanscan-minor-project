"""Build the conventional Ultralytics images/{train,val,test} layout temporarily."""

from __future__ import annotations

import os
import shutil
import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
SPLIT_NAMES = ("train", "val", "test")


def _resolve_image(entry: str) -> Path:
    normalized = entry.replace("\\", "/").strip()
    if not normalized or Path(normalized).is_absolute():
        raise ValueError(f"Split entry must be a relative image path, got {entry!r}")
    path = (DATASET_DIR / normalized).resolve()
    image_root = (DATASET_DIR / "images").resolve()
    if image_root not in path.parents:
        raise ValueError(f"Split entry escapes the dataset image directory: {entry!r}")
    if not path.is_file():
        raise ValueError(f"Split image does not exist: {path}")
    return path


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            if os.path.samefile(source, destination):
                return
        except OSError:
            pass
        # This is a generated staging directory; refresh only the staged file,
        # never the source image in dataset/images.
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def build_yolo_layout(destination: Path) -> Path:
    """Materialize hard-linked images and copied polygon labels for Ultralytics."""
    destination = destination.resolve()
    for split_name in SPLIT_NAMES:
        split_file = DATASET_DIR / "splits" / f"{split_name}.txt"
        if not split_file.is_file():
            raise ValueError(f"Missing {split_name} split file: {split_file}")
        entries = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not entries:
            raise ValueError(f"The {split_name} split is empty.")
        for entry in entries:
            image_path = _resolve_image(entry)
            label_source = DATASET_DIR / "annotations" / "labels" / f"{image_path.stem}.txt"
            if not label_source.is_file():
                raise ValueError(f"Missing polygon label for {image_path.name}: {label_source}")
            image_destination = destination / "images" / split_name / image_path.name
            label_destination = destination / "labels" / split_name / f"{image_path.stem}.txt"
            _link_or_copy(image_path, image_destination)
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label_source, label_destination)

    yaml_path = destination / "dataset.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    root_as_posix = destination.as_posix().replace("'", "''")
    yaml_path.write_text(
        f"path: '{root_as_posix}'\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: upper_arm\n",
        encoding="utf-8",
    )
    return yaml_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATASET_DIR / "yolo_working",
        help="Generated YOLO images/labels staging directory.",
    )
    args = parser.parse_args(argv)
    destination = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    yaml_path = build_yolo_layout(destination)
    print(json.dumps({"dataset_yaml": str(yaml_path), "staging_directory": str(destination.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
