"""Build a small, review-backed YOLO dataset while quarantining ambiguous labels.

This script never edits source images, source labels, or the fixed split. It
stages the latest CVAT labels for previously cleared images, the three approved
visible-crop-only images, and the exact-match val/54 scope cleanup proposal.
All other reviewed-queue items are excluded and recorded as quarantined.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

SPLITS = ("train", "val", "test")
SOURCE_COUNTS = {"train": 84, "val": 24, "test": 12}
EXPECTED_SEED = 2026
APPROVED_CROP_ONLY = {("train", "14.png"), ("train", "26.png"), ("train", "44.png")}
SCOPE_CORRECTION = ("val", "54.png")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cross(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _proper_intersection(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    eps = 1e-10
    o1, o2, o3, o4 = _cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)
    return ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and (
        (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)
    )


def parse_label(path: Path, *, require_single: bool = True) -> list[np.ndarray]:
    rows: list[np.ndarray] = []
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if require_single and len(lines) != 1:
        raise ValueError(f"Expected exactly one upper_arm polygon: {path} has {len(lines)} rows")
    for row_number, line in enumerate(lines, 1):
        fields = line.split()
        if len(fields) < 7 or (len(fields) - 1) % 2 or int(fields[0]) != 0:
            raise ValueError(f"Malformed or non-class-0 YOLO segmentation row: {path}:{row_number}")
        xy = np.asarray([float(x) for x in fields[1:]], dtype=np.float64).reshape(-1, 2)
        if not np.isfinite(xy).all() or np.any(xy < 0) or np.any(xy > 1) or len(np.unique(xy, axis=0)) < 3:
            raise ValueError(f"Invalid/non-finite/out-of-range polygon coordinates: {path}:{row_number}")
        x, y = xy[:, 0], xy[:, 1]
        area = 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
        if area <= 1e-8:
            raise ValueError(f"Zero-area polygon: {path}:{row_number}")
        n = len(xy)
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                if _proper_intersection(xy[i], xy[(i + 1) % n], xy[j], xy[(j + 1) % n]):
                    raise ValueError(f"Self-crossing polygon: {path}:{row_number}")
        rows.append(xy)
    return rows


def rasterize(label: Path, width: int, height: int, *, require_single: bool = True) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for xy in parse_label(label, require_single=require_single):
        px = np.rint(xy * np.array([width - 1, height - 1])).astype(np.int32)
        cv2.fillPoly(mask, [px], 1)
    return mask


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = image.copy()
    tint = np.zeros_like(image)
    tint[:] = color
    on = mask.astype(bool)
    out[on] = cv2.addWeighted(image[on], 0.48, tint[on], 0.52, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, max(1, round(min(image.shape[:2]) / 500)))
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_contact_sheet(rows: list[dict[str, Any]], image_root: Path, label_root: Path, out: Path) -> int:
    from PIL import Image, ImageDraw, ImageFont

    tile_w, tile_h, columns = 300, 250, 4
    font = ImageFont.load_default()
    page_size = 20
    page_count = 0
    for page_index, start in enumerate(range(0, len(rows), page_size), 1):
        subset = rows[start : start + page_size]
        rcount = (len(subset) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * tile_w, rcount * tile_h), (35, 35, 35))
        draw = ImageDraw.Draw(sheet)
        for idx, row in enumerate(subset):
            split, filename = row["split"], row["filename"]
            image_path = image_root / split / filename
            label_path = label_root / split / f"{Path(filename).stem}.txt"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            tile_x, tile_y = (idx % columns) * tile_w, (idx // columns) * tile_h
            if image is not None:
                if label_path.is_file():
                    try:
                        vis = overlay(image, rasterize(label_path, image.shape[1], image.shape[0]), (0, 220, 50))
                    except Exception:
                        vis = image
                else:
                    vis = image
                rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                pil.thumbnail((tile_w - 8, tile_h - 34))
                sheet.paste(pil, (tile_x + (tile_w - pil.width) // 2, tile_y + 26))
            title = f"{split}/{filename}"
            draw.text((tile_x + 4, tile_y + 4), title, font=font, fill="white")
            issue = str(row.get("issue", ""))
            draw.text((tile_x + 4, tile_y + tile_h - 18), issue[:44], font=font, fill=(255, 200, 130))
        target = out.with_name(f"{out.stem}_{page_index:02d}.jpg")
        target.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(target, quality=88)
        page_count += 1
    return page_count


def build(project: Path, output: Path) -> dict[str, Any]:
    ml = project / "ml-service"
    package = ml / "results" / "cvat_import_audit_execution_check" / "human_review_package"
    image_root = ml / "dataset" / "yolo_working" / "images"
    stage_labels = ml / "results" / "cvat_import_staging_execution_check" / "labels"
    split_manifest_source = ml / "dataset" / "splits" / "split_manifest.csv"
    queue_path = package / "automation_followup" / "unresolved_cases.csv"
    decision_path = package / "automation_followup" / "crop_policy_decisions.csv"
    scope_label = package / "proposed_scope_cleanup" / "labels" / "val" / "54.txt"
    old_scope_label = ml / "dataset" / "yolo_working" / "labels" / "val" / "54.txt"
    if not all(p.is_file() for p in (split_manifest_source, queue_path, decision_path, scope_label, old_scope_label)):
        raise FileNotFoundError("Required source split/review/proposal artifact is missing.")

    source_rows = _read_csv(split_manifest_source)
    if len(source_rows) != 120:
        raise ValueError(f"Expected the frozen 120-image source split manifest; found {len(source_rows)} rows")
    source_by_key: dict[tuple[str, str], dict[str, str]] = {}
    source_group_splits: dict[str, set[str]] = defaultdict(set)
    for row in source_rows:
        key = (row["split"], row["image_filename"])
        if key in source_by_key or row["split"] not in SPLITS or not row.get("duplicate_group"):
            raise ValueError(f"Invalid/duplicate source split row: {row}")
        source_by_key[key] = row
        source_group_splits[row["duplicate_group"]].add(row["split"])
    source_counts = dict(Counter(row["split"] for row in source_rows))
    if source_counts != SOURCE_COUNTS or any(len(splits) != 1 for splits in source_group_splits.values()):
        raise ValueError("Source split counts or duplicate-group isolation do not match the fixed split.")

    queue = _read_csv(queue_path)
    queue_by_key = {(r["split"], r["filename"]): r for r in queue}
    if len(queue_by_key) != len(queue):
        raise ValueError("Review queue contains duplicate image rows.")
    crop_rows = _read_csv(decision_path)
    crop_by_key = {(r["split"], r["filename"]): r for r in crop_rows}
    for key in APPROVED_CROP_ONLY:
        row = queue_by_key.get(key)
        crop = crop_by_key.get(key)
        if not row or row.get("issue") != "polygon touches image boundary" or not crop:
            raise ValueError(f"{key} is not a crop-only review item with an audited crop proposal.")
        if crop.get("visible_region_candidate_pixel_iou") != "1.000000":
            raise ValueError(f"Visible-region proposal does not match source pixels exactly for {key}.")

    selected_keys = (set(source_by_key) - set(queue_by_key)) | APPROVED_CROP_ONLY
    if SCOPE_CORRECTION not in selected_keys or SCOPE_CORRECTION in queue_by_key:
        raise ValueError("val/54 must be cleared from the refreshed queue and retained via its scope proposal.")
    if len(selected_keys) + (len(queue_by_key) - len(APPROVED_CROP_ONLY)) != 120:
        raise ValueError("Selected/quarantined inventory does not account for all 120 source images.")

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing review/dataset output: {output}")
    data_root = output / "dataset"
    for split in SPLITS:
        (data_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (data_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (data_root / "split_metadata").mkdir(parents=True, exist_ok=True)

    auto_rows: list[dict[str, Any]] = []
    retained_rows: list[dict[str, Any]] = []
    quarantine_rows: list[dict[str, Any]] = []
    included_manifest: list[dict[str, Any]] = []
    included_counts = Counter()
    image_hashes: dict[str, tuple[str, str]] = {}
    source_label_hashes: dict[str, str] = {}
    image_label_rows = 0

    for key, source in sorted(source_by_key.items()):
        split, filename = key
        src_image = image_root / split / filename
        src_label = stage_labels / split / f"{Path(filename).stem}.txt"
        if not src_image.is_file():
            raise FileNotFoundError(f"Source split image missing: {src_image}")
        if key not in selected_keys:
            queued = queue_by_key.get(key, {})
            reason = queued.get("issue", "Not in the approved quarantine-safe selection")
            quarantine_rows.append({
                "split": split, "filename": filename, "status": "QUARANTINED_UNRESOLVED",
                "issue": reason, "reason": queued.get("reason", ""),
                "minimum_action": queued.get("minimum_action", "Review/correct in CVAT"),
                "sam2_candidate_used_as_label": "false",
            })
            continue

        proposed = None
        status = "CONFIDENTLY_RETAINED"
        reason = "Not present in the refreshed unresolved queue; carried forward from previous visual review."
        if key == SCOPE_CORRECTION:
            proposed = scope_label
            status = "AUTO_CORRECTED_SCOPE"
            reason = "Retained polygon exactly matches the previous reviewed target mask; removed row is an unrelated edge region; marker-gap check is clear."
        elif key in APPROVED_CROP_ONLY:
            crop = crop_by_key[key]
            proposed = ml / crop["candidate_label"]
            status = "CONFIDENTLY_RETAINED_VISIBLE_CROP"
            reason = "Global visible-region crop policy approved; this is a crop-only queue item; audited candidate raster IoU is 1.0 and no polygon pixel changes."
        source_for_label = proposed or src_label
        if not source_for_label.is_file():
            raise FileNotFoundError(f"Approved candidate label missing for {split}/{filename}: {source_for_label}")
        parse_label(source_for_label)
        image = cv2.imread(str(src_image), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable image: {src_image}")
        if key in APPROVED_CROP_ONLY:
            cvat_mask = rasterize(src_label, image.shape[1], image.shape[0])
            crop_mask = rasterize(source_for_label, image.shape[1], image.shape[0])
            if not np.array_equal(cvat_mask, crop_mask):
                raise ValueError(f"Crop policy label changes raster pixels for {split}/{filename}.")
        if key == SCOPE_CORRECTION:
            proposed_mask = rasterize(source_for_label, image.shape[1], image.shape[0])
            previous_rows = parse_label(old_scope_label, require_single=False)
            row_ious = []
            for old_polygon in previous_rows:
                old_mask = np.zeros_like(proposed_mask)
                points = np.rint(old_polygon * np.array([image.shape[1] - 1, image.shape[0] - 1])).astype(np.int32)
                cv2.fillPoly(old_mask, [points], 1)
                union = np.logical_or(proposed_mask, old_mask).sum()
                row_ious.append(float(np.logical_and(proposed_mask, old_mask).sum() / union) if union else 1.0)
            iou = max(row_ious, default=0.0)
            if iou != 1.0:
                raise ValueError(f"val/54 proposed kept polygon no longer matches the previous reviewed mask: IoU={iou:.6f}")
            # Save a direct before/after comparison without touching either source.
            before = overlay(image, rasterize(src_label, image.shape[1], image.shape[0], require_single=False), (0, 140, 255))
            after = overlay(image, proposed_mask, (40, 220, 50))
            compare = np.hstack([before, after])
            cv2.imwrite(str(output / "before_after_val_54_scope.jpg"), compare)

        destination_image = data_root / "images" / split / filename
        destination_label = data_root / "labels" / split / f"{Path(filename).stem}.txt"
        shutil.copy2(src_image, destination_image)
        shutil.copy2(source_for_label, destination_label)
        included_counts[split] += 1
        image_label_rows += len(parse_label(destination_label))
        digest = sha256(destination_image)
        if digest in image_hashes:
            raise ValueError(f"Duplicate image bytes in candidate subset: {image_hashes[digest]} and {key}")
        image_hashes[digest] = key
        source_label_hashes[f"{split}/{filename}"] = sha256(source_for_label)
        group = source["duplicate_group"]
        included_manifest.append({
            "image_filename": filename, "split": split, "duplicate_group": group,
            "source_image_sha256": digest, "candidate_label_sha256": sha256(destination_label),
            "label_source": str(source_for_label.relative_to(project)), "selection_status": status,
        })
        decision = {
            "split": split, "filename": filename, "status": status,
            "correction": "None; source raster retained" if status != "AUTO_CORRECTED_SCOPE" else "Drop unrelated left-edge polygon; retain the exact previous target mask",
            "confidence": "high", "reason": reason,
            "source_image_sha256": digest, "candidate_label_sha256": sha256(destination_label),
        }
        if status == "AUTO_CORRECTED_SCOPE" or status == "CONFIDENTLY_RETAINED_VISIBLE_CROP":
            auto_rows.append(decision)
        else:
            retained_rows.append(decision)

    if not included_counts["train"] or not included_counts["val"] or not included_counts["test"]:
        raise ValueError(f"Candidate subset must retain all three source split partitions: {dict(included_counts)}")
    if len(selected_keys) != sum(included_counts.values()):
        raise ValueError(f"Selection/labels mismatch: selected {len(selected_keys)} vs copied {sum(included_counts.values())}")

    # Global selected-group isolation inherits the frozen 120-item source mapping.
    selected_group_splits: dict[str, set[str]] = defaultdict(set)
    for row in included_manifest:
        selected_group_splits[row["duplicate_group"]].add(row["split"])
    if any(len(splits) != 1 for splits in selected_group_splits.values()):
        raise ValueError("A near-duplicate/source group leaked across included split partitions.")

    config = {
        "path": ".", "train": "images/train", "val": "images/val", "test": "images/test",
        "names": {0: "upper_arm"},
    }
    (data_root / "dataset.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    source_split_sha = sha256(split_manifest_source)
    included_manifest_path = data_root / "split_metadata" / "split_manifest.csv"
    _write_csv(included_manifest_path, included_manifest, [
        "image_filename", "split", "duplicate_group", "source_image_sha256", "candidate_label_sha256", "label_source", "selection_status"
    ])
    _write_csv(data_root / "split_metadata" / "source_split_manifest.csv", source_rows,
               ["image_filename", "split", "duplicate_group"])
    selection_rows: list[dict[str, Any]] = []
    retained_map = {(r["split"], r["filename"]): r for r in auto_rows + retained_rows}
    quarantined_map = {(r["split"], r["filename"]): r for r in quarantine_rows}
    for key in sorted(source_by_key):
        src = source_by_key[key]
        decision = retained_map.get(key) or quarantined_map[key]
        selection_rows.append({
            "image_filename": key[1], "split": key[0], "duplicate_group": src["duplicate_group"],
            "included": "true" if key in retained_map else "false",
            "status": decision["status"], "reason": decision.get("reason", decision.get("issue", "")),
        })
    _write_csv(data_root / "split_metadata" / "dataset_selection_manifest.csv", selection_rows,
               ["image_filename", "split", "duplicate_group", "included", "status", "reason"])
    for split in SPLITS:
        (data_root / "split_metadata" / f"{split}.txt").write_text(
            "".join(f"{row['image_filename']}\n" for row in included_manifest if row["split"] == split), encoding="utf-8"
        )

    split_report = {
        "dataset_profile": "verified_quarantine_subset",
        "seed": EXPECTED_SEED,
        "source_split_counts": SOURCE_COUNTS,
        "split_counts": {split: int(included_counts[split]) for split in SPLITS},
        "source_split_manifest_sha256": source_split_sha,
        "no_duplicate_or_near_duplicate_group_leakage": True,
        "source_split_regenerated": False,
        "selection_policy": "Keep prior visually reviewed non-queue images; add three crop-only items covered by approved visible-region policy; apply exact-match val/54 scope cleanup; quarantine all remaining queue rows.",
        "included_count": len(included_manifest),
        "quarantined_count": len(quarantine_rows),
        "class_mapping": {"0": "upper_arm"},
    }
    (data_root / "split_metadata" / "split_report.json").write_text(json.dumps(split_report, indent=2), encoding="utf-8")

    source_audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "approved_crop_policy": "visible upper-arm portion only; clipped at image boundary; do not reconstruct hidden anatomy",
        "source_split_manifest_sha256": source_split_sha,
        "source_cvat_review_queue_sha256": sha256(queue_path),
        "sam2_proposals_used_as_ground_truth": False,
        "source_images_or_active_labels_modified": False,
        "selected_image_count": len(included_manifest),
        "selected_label_rows": image_label_rows,
        "selected_split_counts": dict(split_report["split_counts"]),
        "selected_image_label_pairs": len(included_manifest),
        "selected_exact_unique_images": len(image_hashes),
        "selected_polygon_validation": "all selected labels passed class-0, normalized-coordinate, positive-area, one-polygon, and proper-self-intersection checks",
        "near_duplicate_group_leakage": False,
        "automatic_decisions": len(auto_rows),
        "confidently_retained": len(retained_rows),
        "quarantined_unresolved": len(quarantine_rows),
        "quarantine_issue_counts": dict(Counter(r["issue"] for r in quarantine_rows)),
        "training_started": False,
    }
    (output / "verified_dataset_audit.json").write_text(json.dumps(source_audit, indent=2), encoding="utf-8")

    _write_csv(output / "automatic_decisions.csv", auto_rows,
               ["split", "filename", "status", "correction", "confidence", "reason", "source_image_sha256", "candidate_label_sha256"])
    _write_csv(output / "confidently_retained.csv", retained_rows,
               ["split", "filename", "status", "correction", "confidence", "reason", "source_image_sha256", "candidate_label_sha256"])
    _write_csv(output / "quarantined_unresolved.csv", quarantine_rows,
               ["split", "filename", "status", "issue", "reason", "minimum_action", "sam2_candidate_used_as_label"])
    _write_csv(output / "remaining_unresolved.csv", quarantine_rows,
               ["split", "filename", "status", "issue", "reason", "minimum_action", "sam2_candidate_used_as_label"])

    # Compact queue sheet: only unresolved/quarantined images.
    _write_contact_sheet(quarantine_rows, image_root, stage_labels, output / "unresolved_contact_sheet.jpg")

    # Preview the three crop-policy items. The accepted mask is pixel-identical;
    # the side-by-side output makes that unchanged result explicit.
    for split, filename in sorted(APPROVED_CROP_ONLY):
        image = cv2.imread(str(image_root / split / filename), cv2.IMREAD_COLOR)
        original = stage_labels / split / f"{Path(filename).stem}.txt"
        crop = ml / crop_by_key[(split, filename)]["candidate_label"]
        before_mask = rasterize(original, image.shape[1], image.shape[0])
        after_mask = rasterize(crop, image.shape[1], image.shape[0])
        comparison = np.hstack([
            overlay(image, before_mask, (0, 140, 255)), overlay(image, after_mask, (40, 220, 50))
        ])
        cv2.imwrite(str(output / f"before_after_{split}_{Path(filename).stem}_visible_crop.jpg"), comparison)

    # Zip contains only dataset payload and required split/selection manifests.
    archive_path = output / "poshanscan_verified_quarantine_dataset.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(data_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(data_root).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Dataset ZIP CRC failure: {bad}")
        members = archive.namelist()
        if len(members) != len(set(members)) or any(Path(n).is_absolute() or ".." in Path(n).parts for n in members):
            raise ValueError("Dataset ZIP has duplicate or unsafe paths.")
    source_audit["dataset_zip"] = str(archive_path)
    source_audit["dataset_zip_sha256"] = sha256(archive_path)
    source_audit["dataset_zip_member_count"] = len(members)
    (output / "verified_dataset_audit.json").write_text(json.dumps(source_audit, indent=2), encoding="utf-8")

    report = [
        "# PoshanScan verified quarantine dataset audit",
        "",
        "**Status: annotation preflight passed for a 40-image quarantine-safe subset.** This is a small pipeline-training subset; it is not a representative or clinical validation dataset. No source image, active label, backup, or split assignment was changed.",
        "",
        "## Decisions",
        "",
        "- Automatically corrected: **1 image**. `val/54.png` drops the unrelated left-edge polygon; the retained mask exactly matches the previously reviewed mask at source resolution.",
        f"- Confidently retained under the approved visible-region crop policy: **{len(APPROVED_CROP_ONLY)} images** (`train/14.png`, `train/26.png`, `train/44.png`). Each current mask is raster-identical to its policy candidate; no hidden anatomy was added.",
        f"- Confidently retained from prior visual review: **{len(retained_rows)} images**. These were absent from the unresolved queue.",
        f"- Quarantined and still unresolved: **{len(quarantine_rows)} images**. SAM2 outputs were not used as labels.",
        "- The missing `train/110.png` CVAT annotation remains quarantined. The old mask and SAM2 proposal were not substituted.",
        "",
        "## Validation",
        "",
        f"- Source split remains seed 2026 with 84/24/12 membership; split manifest SHA-256: `{source_split_sha}`.",
        f"- Included subset counts by original partition: `{json.dumps(split_report['split_counts'])}`. These counts are smaller because unresolved items are quarantined; no image was moved or split regenerated.",
        f"- Included pairs: {len(included_manifest)} image-label pairs; exact-unique images: {len(image_hashes)}; polygon rows: {image_label_rows}.",
        "- Every included label has one class-0 polygon, finite normalized coordinates, positive area, and no proper self-intersection. Image-label basenames match.",
        "- All selected source duplicate groups retain their original partition; exact duplicate images do not appear in the selected dataset.",
        f"- ZIP: `{archive_path.name}`; members: {len(members)}; SHA-256: `{source_audit['dataset_zip_sha256']}`; CRC verification passed.",
        "",
        "## Artifacts",
        "",
        "- `automatic_decisions.csv`, `confidently_retained.csv`, `quarantined_unresolved.csv`, and `remaining_unresolved.csv` list the per-image outcomes.",
        "- `unresolved_contact_sheet_*.jpg` contains only the quarantined items.",
        "- `before_after_val_54_scope.jpg` and three crop-policy previews show each staged change/unchanged mask.",
        "- `dataset/` is the extracted tree represented by the ZIP; source/selection manifests record original assignments and decision reasons.",
        "",
        "## Training decision",
        "",
        "Annotation preflight for this isolated subset passed. Training has not been run in this workspace: the available machine has CPU-only PyTorch, and no Colab runtime is connected. The updated Colab notebook is configured to run pretrained YOLOv8n-seg training only after its own preflight passes and to persist outputs directly in Google Drive. Held-out metrics from the small synthetic test subset will be exploratory and must not be described as real-world or MUAC accuracy.",
    ]
    (output / "verified_dataset_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return source_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "verified_quarantine_dataset")
    args = parser.parse_args()
    result = build(args.project.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
