"""Package exact-unique source images for a focused CVAT annotation task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence
from zipfile import ZIP_STORED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES = ROOT / "dataset" / "images"
DEFAULT_INVENTORY = ROOT / "dataset" / "reports" / "unique_image_inventory.json"
DEFAULT_OUTPUT = ROOT / "results" / "annotation" / "cvat_unique_images.zip"
SPLIT_DIR = ROOT / "dataset" / "splits"
SUPPORTED = {".jpg", ".jpeg", ".png"}


def export_unique_images(
    images_dir: Path,
    inventory_json: Path,
    output_zip: Path,
    *,
    split_dir: Optional[Path] = None,
) -> int:
    if not inventory_json.is_file():
        raise FileNotFoundError(f"Unique inventory not found: {inventory_json}; run scripts/inventory_images.py first.")
    unique_rows = json.loads(inventory_json.read_text(encoding="utf-8"))
    if not isinstance(unique_rows, list) or not unique_rows:
        raise ValueError("Unique image inventory is empty or invalid.")

    active_names: Optional[set[str]] = None
    split_paths = [split_dir / f"{name}.txt" for name in ("train", "val", "test")] if split_dir else []
    if split_dir and not all(path.is_file() for path in split_paths):
        raise FileNotFoundError(f"All train/val/test split files are required in {split_dir}.")
    if split_dir:
        active_names = {
            Path(line.strip().replace("\\", "/")).name
            for split_path in split_paths
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    sources: list[Path] = []
    stems: set[str] = set()
    for row in unique_rows:
        filename = Path(str(row.get("canonical_filename", ""))).name
        if active_names is not None and filename not in active_names:
            continue
        source = images_dir / filename
        if not filename or source.suffix.lower() not in SUPPORTED or not source.is_file():
            raise ValueError(f"Invalid or missing canonical image in unique inventory: {filename!r}")
        if source.stem.casefold() in stems:
            raise ValueError(f"Duplicate image basename stem cannot be represented by YOLO labels: {source.stem}")
        stems.add(source.stem.casefold())
        sources.append(source)

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_zip, "w", compression=ZIP_STORED) as archive:
        for source in sources:
            archive.write(source, arcname=source.name)
    return len(sources)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    images_dir = args.images if args.images.is_absolute() else ROOT / args.images
    inventory = args.inventory if args.inventory.is_absolute() else ROOT / args.inventory
    output = args.output if args.output.is_absolute() else ROOT / args.output
    try:
        count = export_unique_images(images_dir, inventory, output, split_dir=SPLIT_DIR)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Packaged {count} exact-unique source images: {output}")
    print("Original images were read only; no masks or labels were generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
