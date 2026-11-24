"""Import reviewed CVAT Ultralytics YOLO segmentation labels into PoshanScan."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Optional, Sequence
from zipfile import BadZipFile, ZipFile

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "dataset" / "images"
LABEL_DIR = ROOT / "dataset" / "annotations" / "labels"
SPLIT_DIR = ROOT / "dataset" / "splits"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_annotations import validate_polygon_line  # noqa: E402


def _cross(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _has_proper_self_intersection(points: list[tuple[float, float]]) -> bool:
    count = len(points)
    epsilon = 1e-12
    for first in range(count):
        a, b = points[first], points[(first + 1) % count]
        for second in range(first + 2, count):
            if first == 0 and second == count - 1:
                continue
            c, d = points[second], points[(second + 1) % count]
            ab_c, ab_d = _cross(a, b, c), _cross(a, b, d)
            cd_a, cd_b = _cross(c, d, a), _cross(c, d, b)
            opposite_ab = (ab_c > epsilon and ab_d < -epsilon) or (ab_c < -epsilon and ab_d > epsilon)
            opposite_cd = (cd_a > epsilon and cd_b < -epsilon) or (cd_a < -epsilon and cd_b > epsilon)
            if opposite_ab and opposite_cd:
                return True
    return False


def _polygon_geometry_error(line: str, line_number: int) -> str:
    fields = line.split()
    try:
        values = [float(value) for value in fields[1:]]
    except ValueError:
        return ""
    points = list(zip(values[::2], values[1::2]))
    area_twice = sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )
    if not math.isfinite(area_twice) or abs(area_twice) <= 2e-10:
        return f"line {line_number}: polygon has zero or near-zero area"
    if _has_proper_self_intersection(points):
        return f"line {line_number}: polygon has a proper self-intersection"
    return ""


def _scalar(value: str) -> str:
    return value.strip().strip(",").strip().strip("\"'")


def parse_class_names(data_yaml: str) -> list[str]:
    """Read the simple names list/map emitted by a CVAT YOLO export."""
    lines = data_yaml.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^\s*names\s*:\s*(.*?)\s*$", line)
        if not match:
            continue
        inline = match.group(1)
        if inline:
            body = inline.strip()
            if body.startswith("[") and body.endswith("]"):
                return [_scalar(item) for item in body[1:-1].split(",") if _scalar(item)]
            if body.startswith("{") and body.endswith("}"):
                values: dict[int, str] = {}
                for item in body[1:-1].split(","):
                    entry = re.match(r"\s*(\d+)\s*:\s*(.+?)\s*", item)
                    if entry:
                        values[int(entry.group(1))] = _scalar(entry.group(2))
                return [values[key] for key in sorted(values)]
            return [_scalar(body)]

        values_by_id: dict[int, str] = {}
        list_values: list[str] = []
        for child in lines[index + 1:]:
            if child.strip() and not child[0].isspace():
                break
            value = child.strip()
            if not value or value.startswith("#"):
                continue
            mapping = re.match(r"^(\d+)\s*:\s*(.+?)\s*$", value)
            sequence = re.match(r"^-\s*(.+?)\s*$", value)
            if mapping:
                values_by_id[int(mapping.group(1))] = _scalar(mapping.group(2))
            elif sequence:
                list_values.append(_scalar(sequence.group(1)))
        if values_by_id:
            if sorted(values_by_id) != list(range(len(values_by_id))):
                raise ValueError("Class IDs in data.yaml names must start at 0 and be consecutive.")
            return [values_by_id[class_id] for class_id in range(len(values_by_id))]
        return list_values
    raise ValueError("CVAT export is missing a data.yaml names section.")


def import_cvat_archive(
    archive_path: Path,
    image_dir: Path = IMAGE_DIR,
    label_dir: Path = LABEL_DIR,
    *,
    overwrite: bool = False,
    expected_stems: Optional[set[str]] = None,
) -> dict[str, object]:
    """Validate a CVAT export fully, then copy only its polygon TXT files."""
    if not archive_path.is_file():
        raise FileNotFoundError(f"CVAT export ZIP not found: {archive_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    images_by_stem: dict[str, Path] = {}
    for path in image_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        stem_key = path.stem.casefold()
        if stem_key in images_by_stem:
            raise ValueError(
                f"Images {images_by_stem[stem_key].name} and {path.name} share a YOLO label basename."
            )
        images_by_stem[stem_key] = path

    try:
        archive = ZipFile(archive_path)
    except BadZipFile as exc:
        raise ValueError(f"Not a valid ZIP archive: {archive_path}") from exc

    with archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        yaml_members = [item for item in members if PurePosixPath(item.filename).name.casefold() == "data.yaml"]
        if len(yaml_members) != 1:
            raise ValueError("Expected exactly one data.yaml in the CVAT YOLO export ZIP.")
        class_names = parse_class_names(archive.read(yaml_members[0]).decode("utf-8-sig"))
        if class_names != ["upper_arm"]:
            raise ValueError(f"Expected exactly one class named upper_arm, found: {class_names!r}.")

        label_members = [
            item for item in members
            if PurePosixPath(item.filename).suffix.casefold() == ".txt"
            and any(part.casefold() == "labels" for part in PurePosixPath(item.filename).parts[:-1])
        ]
        if not label_members:
            raise ValueError("No labels/**/*.txt polygon files were found in the CVAT export ZIP.")

        validated: dict[str, bytes] = {}
        errors: list[str] = []
        for item in label_members:
            label_name = PurePosixPath(item.filename).name
            stem = PurePosixPath(label_name).stem
            image = images_by_stem.get(stem.casefold())
            if image is None:
                errors.append(f"{label_name}: no matching image basename in {image_dir}")
                continue
            key = image.stem.casefold()
            if key in validated:
                errors.append(f"{label_name}: duplicate label basename in CVAT ZIP")
                continue
            payload = archive.read(item)
            try:
                text = payload.decode("utf-8-sig")
            except UnicodeDecodeError:
                errors.append(f"{label_name}: label file is not UTF-8 text")
                continue
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if not lines:
                errors.append(f"{label_name}: empty label files are not accepted")
                continue
            for line_number, line in enumerate(lines, start=1):
                valid, message = validate_polygon_line(line, line_number)
                if not valid:
                    errors.append(f"{label_name}: {message}")
                    continue
                geometry_error = _polygon_geometry_error(line, line_number)
                if geometry_error:
                    errors.append(f"{label_name}: {geometry_error}")
            validated[key] = ("\n".join(lines) + "\n").encode("utf-8")

    if errors:
        raise ValueError("CVAT import rejected; no labels were copied:\n- " + "\n- ".join(errors))

    if expected_stems is not None:
        expected = {stem.casefold() for stem in expected_stems}
        actual = set(validated)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            raise ValueError(
                "CVAT import rejected before copying because labels do not cover exactly the fixed split. "
                f"Missing stems: {missing[:20]}; extra stems: {extra[:20]}"
            )

    conflicts = [label_dir / f"{images_by_stem[key].stem}.txt" for key in validated if (label_dir / f"{images_by_stem[key].stem}.txt").exists()]
    if conflicts and not overwrite:
        raise FileExistsError(
            "Label files already exist; nothing was copied. Re-run with --overwrite only if the CVAT export is reviewed. "
            f"Conflicts: {[path.name for path in conflicts[:5]]}"
        )

    label_dir.mkdir(parents=True, exist_ok=True)
    for key, payload in validated.items():
        target = label_dir / f"{images_by_stem[key].stem}.txt"
        target.write_bytes(payload)

    return {
        "class_names": class_names,
        "imported_label_files": len(validated),
        "fixed_split_coverage_verified": expected_stems is not None,
        "destination": str(label_dir),
        "source_images_unchanged": True,
        "labels_are_manual_cvat_exports": True,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="CVAT Ultralytics YOLO Segmentation export ZIP")
    parser.add_argument("--images", type=Path, default=IMAGE_DIR)
    parser.add_argument("--labels", type=Path, default=LABEL_DIR)
    parser.add_argument(
        "--splits",
        type=Path,
        default=SPLIT_DIR,
        help="Existing fixed split directory; all listed image labels are required by default.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing label files after review")
    args = parser.parse_args(argv)
    archive_path = args.archive if args.archive.is_absolute() else Path.cwd() / args.archive
    image_dir = args.images if args.images.is_absolute() else ROOT / args.images
    label_dir = args.labels if args.labels.is_absolute() else ROOT / args.labels
    split_dir = args.splits if args.splits.is_absolute() else ROOT / args.splits
    expected_stems: set[str] = set()
    try:
        expected_counts = {"train": 84, "val": 24, "test": 12}
        for split_name, expected_count in expected_counts.items():
            split_path = split_dir / f"{split_name}.txt"
            entries = [line.strip().replace("\\", "/") for line in split_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            stems = [PurePosixPath(entry).stem.casefold() for entry in entries]
            if len(entries) != expected_count or len(stems) != len(set(stems)):
                raise ValueError(
                    f"The existing {split_name} split must contain {expected_count} unique images; found {len(entries)} entries."
                )
            expected_stems.update(PurePosixPath(entry).stem.casefold() for entry in entries)
        if len(expected_stems) != 120:
            raise ValueError(f"The fixed split lists must cover 120 unique images; found {len(expected_stems)}.")
    except (OSError, ValueError) as exc:
        print(f"Fixed split coverage check failed for {split_dir}: {exc}", file=sys.stderr)
        return 2
    try:
        result = import_cvat_archive(
            archive_path,
            image_dir,
            label_dir,
            overwrite=args.overwrite,
            expected_stems=expected_stems,
        )
    except (OSError, ValueError) as exc:
        print(f"CVAT import stopped: {exc}", file=sys.stderr)
        return 2
    import json

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
