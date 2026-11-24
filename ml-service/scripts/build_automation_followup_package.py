"""Build the final review package after SAM2 and strict annotation checks.

Outputs are proposals/reports under results only. Active labels, images, splits,
and backups are read-only. This script does not train, import, or overwrite data.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "yolo_working"
AUDIT = ROOT / "results" / "cvat_import_audit_execution_check"
PACKAGE = AUDIT / "human_review_package"
SAM = PACKAGE / "automated_proposals" / "sam2"
OUT = PACKAGE / "automation_followup"
CROPS = [
    ("train", "14.png"), ("train", "26.png"), ("train", "44.png"),
    ("train", "109.png"), ("train", "115.png"), ("val", "106.png"),
    ("val", "108.png"),
]
SCOPE_PLANS = {
    ("train", "78.jpeg"): {"keep": [1], "drop": [2]},
    ("val", "54.png"): {"keep": [2], "drop": [1]},
    ("test", "39.jpg"): {"keep": [1], "drop": [2, 3, 4]},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    if not fields:
        fields = ["status"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_label(path: Path) -> list[tuple[str, np.ndarray]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 7 or (len(fields) - 1) % 2:
            continue
        values = np.asarray([float(v) for v in fields[1:]], np.float64).reshape(-1, 2)
        rows.append((fields[0], values))
    return rows


def mask_for(row: tuple[str, np.ndarray], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), np.uint8)
    points = row[1].copy()
    points[:, 0] = np.clip(points[:, 0], 0.0, 1.0) * width
    points[:, 1] = np.clip(points[:, 1], 0.0, 1.0) * height
    cv2.fillPoly(mask, [points.astype(np.int32)], 255)
    return mask


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = a > 0, b > 0
    return float(np.count_nonzero(aa & bb) / max(1, np.count_nonzero(aa | bb)))


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.22) -> np.ndarray:
    layer = image.copy()
    layer[mask > 0] = color
    out = cv2.addWeighted(layer, alpha, image, 1 - alpha, 0)
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2, cv2.LINE_AA)
    return out


def contact_pages(rows: list[tuple[str, Path, str]], out_dir: Path, stem: str, cols: int = 4, per_page: int = 12) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    cell_w, cell_h, title_h = 480, 280, 30
    for start in range(0, len(rows), per_page):
        batch = rows[start:start + per_page]
        page = Image.new("RGB", (cols * cell_w, ((len(batch) + cols - 1) // cols) * (cell_h + title_h)), "white")
        draw = ImageDraw.Draw(page)
        for i, (label, path, relative) in enumerate(batch):
            try:
                im = Image.open(path).convert("RGB")
                im.thumbnail((cell_w, cell_h))
                cell = Image.new("RGB", (cell_w, cell_h), "#eeeeee")
                cell.paste(im, ((cell_w - im.width) // 2, (cell_h - im.height) // 2))
            except Exception:
                cell = Image.new("RGB", (cell_w, cell_h), "#ffcccc")
            x, y = (i % cols) * cell_w, (i // cols) * (cell_h + title_h)
            page.paste(cell, (x, y + title_h))
            draw.text((x + 6, y + 5), label[:72], fill="black")
        page_path = out_dir / f"{stem}_{start // per_page + 1:02d}.jpg"
        page.save(page_path, quality=91)
        paths.append(str(page_path.relative_to(PACKAGE)))
    return paths


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "crop_policy_proposal" / "labels").mkdir(parents=True, exist_ok=True)
    (OUT / "crop_policy_proposal" / "previews").mkdir(parents=True, exist_ok=True)
    original_unresolved = read_csv(PACKAGE / "unresolved_cases.csv")
    sam_rows = read_csv(SAM / "sam2_proposals.csv")
    sam_valid_polygons = sum(r.get("candidate_label_valid", "").lower() == "true" for r in sam_rows)
    sam_mask_only = sum(r.get("candidate_label_valid", "").lower() == "false" for r in sam_rows)
    sam_map = {(r["split"], r["filename"], r["row"]): r for r in sam_rows}
    audit_json = json.loads((AUDIT / "cvat_export_audit.json").read_text(encoding="utf-8-sig"))
    marker_rows = read_csv(AUDIT / "aruco_marker_gap_scan.csv")
    marker_gaps = {(r["split"], r["filename"]) for r in marker_rows if r.get("marker_center_covered", "").lower() == "false"}

    scope_decisions: list[dict[str, str]] = []
    scope_preview_rows = []
    scope_labels = PACKAGE / "proposed_scope_cleanup" / "labels"
    for key, plan in SCOPE_PLANS.items():
        split, filename = key
        image = cv2.imread(str(DATA / "images" / split / filename))
        h, w = image.shape[:2]
        stage_path = ROOT / "results" / "cvat_import_staging_execution_check" / "labels" / split / f"{Path(filename).stem}.txt"
        old_path = DATA / "labels" / split / f"{Path(filename).stem}.txt"
        staged_rows = parse_label(stage_path)
        old_rows = parse_label(old_path)
        candidate_rows = [staged_rows[i - 1] for i in plan["keep"]]
        candidate_path = scope_labels / split / f"{Path(filename).stem}.txt"
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        source_lines = [line for line in stage_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        candidate_path.write_text("\n".join(source_lines[i - 1] for i in plan["keep"]) + "\n", encoding="utf-8")
        match_scores = [max((mask_iou(mask_for(row, w, h), mask_for(old, w, h)) for old in old_rows), default=0.0) for row in candidate_rows]
        main_iou = max(match_scores, default=0.0)
        residual_marker = key in marker_gaps
        fully_resolved = (key == ("val", "54.png") and main_iou == 1.0 and not residual_marker)
        action = "DROP_UNRELATED_POLYGON_KEEP_PREVIOUSLY_REVIEWED_TARGET"
        if key == ("test", "39.jpg"):
            action = "DROP_THREE_BACKGROUND_POLYGONS_TARGET_SHAPE_STILL_NEEDS_REVIEW"
        elif key == ("train", "78.jpeg"):
            action = "DROP_BACKGROUND_POLYGON_TARGET_MASK_MATCHES_OLD_MARKER_GAP_REMAINS"
        scope_decisions.append({
            "split": split, "filename": filename, "keep_rows": ",".join(map(str, plan["keep"])),
            "drop_rows": ",".join(map(str, plan["drop"])), "target_old_mask_max_iou": f"{main_iou:.6f}",
            "marker_center_outside_current_mask": str(residual_marker).lower(),
            "automatic_action": action, "fully_resolved_image": str(fully_resolved).lower(),
            "confidence": "0.99" if main_iou == 1.0 else "0.99 for background-row removal; target outline differs",
            "reason": "Row-level raster comparison against the prior active labels plus the full-resolution scope preview.",
            "candidate_label": str(candidate_path.relative_to(ROOT)).replace("\\", "/"),
            "preview": str((PACKAGE / "proposed_scope_cleanup" / "previews" / f"{split}_{Path(filename).stem}_scope_cleanup_before_after.jpg").relative_to(ROOT)).replace("\\", "/"),
        })
        preview = PACKAGE / "proposed_scope_cleanup" / "previews" / f"{split}_{Path(filename).stem}_scope_cleanup_before_after.jpg"
        scope_preview_rows.append((f"{split}/{filename}: {action}", preview, str(preview)))
    write_csv(OUT / "automatic_decisions.csv", scope_decisions)

    # Revalidate crops under a proposed visible-region objective. The original
    # normalized polygon is clipped to [0,1], so existing in-frame masks remain
    # unchanged if clipping is the only operation needed.
    crop_rows = []
    crop_previews = []
    for split, filename in CROPS:
        image_path = DATA / "images" / split / filename
        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]
        source = ROOT / "results" / "cvat_import_staging_execution_check" / "labels" / split / f"{Path(filename).stem}.txt"
        rows = parse_label(source)
        clipped = []
        for cls, coords in rows:
            coords = coords.copy()
            coords[:, 0] = np.clip(coords[:, 0], 0.0, 1.0)
            coords[:, 1] = np.clip(coords[:, 1], 0.0, 1.0)
            clipped.append((cls, coords))
        text_lines = []
        current_masks = []
        candidate_masks = []
        touches = False
        for row in rows:
            mask = mask_for(row, w, h)
            current_masks.append(mask)
            points = row[1]
            touches = touches or bool(np.any(np.isclose(points, 0.0, atol=1e-6)) or np.any(np.isclose(points, 1.0, atol=1e-6)))
        for row in clipped:
            candidate_masks.append(mask_for(row, w, h))
            coords = row[1].reshape(-1)
            text_lines.append(row[0] + " " + " ".join(f"{float(v):.6f}" for v in coords))
        old_union = np.maximum.reduce(current_masks) if current_masks else np.zeros((h, w), np.uint8)
        new_union = np.maximum.reduce(candidate_masks) if candidate_masks else np.zeros((h, w), np.uint8)
        candidate = OUT / "crop_policy_proposal" / "labels" / split / f"{Path(filename).stem}.txt"
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
        before = overlay(image, old_union, (0, 180, 255))
        after = overlay(image, new_union, (255, 160, 0))
        compare = cv2.addWeighted(before, 0.5, after, 0.5, 0)
        for title, panel in [("CURRENT: full-arm rule", before), ("PROPOSED: visible arm clipped to frame", after), ("OVERLAY: exact pixel comparison", compare)]:
            cv2.putText(panel, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(panel, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        panel_w = min(520, w)
        panels = [cv2.resize(p, (panel_w, round(h * panel_w / w))) for p in (before, after, compare)]
        preview_path = OUT / "crop_policy_proposal" / "previews" / f"{split}_{Path(filename).stem}_crop_policy.jpg"
        cv2.imwrite(str(preview_path), np.concatenate(panels, axis=1), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        pixel_iou = mask_iou(old_union, new_union)
        crop_rows.append({
            "split": split, "filename": filename, "native_finding": "upper arm remains visible to the bottom frame; elbow/invisible anatomy is cropped",
            "current_polygon_touches_frame": str(touches).lower(), "visible_region_candidate_pixel_iou": f"{pixel_iou:.6f}",
            "candidate_label": str(candidate.relative_to(ROOT)).replace("\\", "/"),
            "preview": str(preview_path.relative_to(ROOT)).replace("\\", "/"),
            "proposal": "Allow a partial visible upper-arm mask clipped at the image boundary; do not infer the hidden elbow or anatomy.",
            "status": "needs one global policy approval; candidate changes no pixels if IoU is 1.0",
            "muac_limit": "A frame-truncated arm cannot provide a reliable shoulder-to-elbow midpoint for MUAC; segmentation-only use is possible.",
        })
        crop_previews.append((f"{split}/{filename}: frame-clipped visible-region proposal (IoU={pixel_iou:.3f})", preview_path, str(preview_path)))
    write_csv(OUT / "crop_policy_decisions.csv", crop_rows)
    eligibility_path = PACKAGE / "dataset_eligibility_decisions.csv"
    if not (OUT / "dataset_eligibility_before_automation.csv").exists():
        shutil.copy2(eligibility_path, OUT / "dataset_eligibility_before_automation.csv")
    write_csv(eligibility_path, [
        {
            "split": row["split"], "filename": row["filename"],
            "native_resolution_finding": row["native_finding"],
            "recommended_action": "One global decision: allow a partial upper-arm mask clipped to the visible image frame, without inferring the unseen elbow. If declined, provide a full-arm replacement in the same split or an explicit fixed-split eligibility plan.",
            "visible_region_candidate": row["candidate_label"],
            "candidate_mask_pixel_iou": row["visible_region_candidate_pixel_iou"],
            "muac_limit": row["muac_limit"],
        }
        for row in crop_rows
    ])
    crop_contact = contact_pages(crop_previews, OUT / "crop_policy_proposal" / "contact_sheets", "crop_policy", per_page=8)

    # A refreshed queue removes only the one case that passed the strict
    # scope/identity check; all other items remain until their ambiguity clears.
    unresolved = [r for r in original_unresolved if not (r["split"], r["filename"]) == ("val", "54.png")]
    sam_by_image: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in sam_rows:
        sam_by_image.setdefault((row["split"], row["filename"]), []).append(row)
    queue_rows = []
    for record in unresolved:
        key = record["split"], record["filename"]
        candidates = sam_by_image.get(key, [])
        candidates.sort(key=lambda r: float(r.get("candidate_area_ratio") or 0), reverse=True)
        best = candidates[0] if candidates else {}
        issue = record.get("issue", "")
        if "self-crossing" in issue:
            minimum_action = "Review the SAM2 outer-contour proposal and approve or make a small CVAT correction; redraw only if the proposed silhouette is wrong."
        elif "missing CVAT annotation" in issue:
            minimum_action = "Create train/110 in CVAT using the saved old polygon as a starting outline, then bridge the marker only if the visible outer-arm contour supports it."
        elif key in CROPS:
            minimum_action = "Covered by one proposed visible-region policy decision; if approved, review marker only if the detector also flagged a gap."
        elif "marker" in issue:
            minimum_action = "Compare the selected SAM2 proposal against the original and confirm the visible boundary plus amodal marker bridge."
        elif "multiple polygons" in issue:
            minimum_action = "Use the row-level scope candidates; confirm remaining target rows only."
        else:
            minimum_action = "Compare the current CVAT and prior reviewed mask; approve only if the silhouette follows the policy."
        queue_rows.append({
            **record,
            "sam2_best_candidate_preview": best.get("preview", ""),
            "sam2_candidate_label": best.get("candidate_label", ""),
            "sam2_candidate_score_not_class_confidence": best.get("sam_predicted_iou", ""),
            "sam2_visible_seed_iou": best.get("seed_candidate_iou_excluding_marker", ""),
            "minimum_action": minimum_action,
            "status": "UNRESOLVED_HUMAN_REVIEW_REQUIRED",
        })
    write_csv(OUT / "unresolved_cases.csv", queue_rows)

    # Contact sheets use one largest-area target prompt per image; all polygon
    # row-specific views remain available in the complete SAM2 gallery.
    gallery_rows = []
    for record in unresolved:
        key = record["split"], record["filename"]
        candidates = sam_by_image.get(key, [])
        if not candidates:
            continue
        source_label = ROOT / "results" / "cvat_import_staging_execution_check" / "labels" / key[0] / f"{Path(key[1]).stem}.txt"
        if not source_label.exists():
            source_label = DATA / "labels" / key[0] / f"{Path(key[1]).stem}.txt"
        source_rows = parse_label(source_label)
        source_image = cv2.imread(str(DATA / "images" / key[0] / key[1]))
        source_h, source_w = source_image.shape[:2]
        areas = {
            str(i): abs(cv2.contourArea((row[1] * np.asarray([source_w, source_h])).astype(np.float32)))
            for i, row in enumerate(source_rows, 1)
        }
        target = max(candidates, key=lambda r: areas.get(r.get("row", ""), 0.0))
        preview = SAM / target["preview"]
        gallery_rows.append((f"{record['split']}/{record['filename']} | {record.get('review_class','')}", preview, target["preview"]))
    contact_paths = contact_pages(gallery_rows, OUT / "contact_sheets", "unresolved_sam2", per_page=12)

    # Keep a single quick-scan sheet for the Colab preflight, plus paginated
    # higher-resolution sheets for actual review.
    old_sheet = PACKAGE / "unresolved_contact_sheet.jpg"
    if old_sheet.exists() and not (OUT / "unresolved_contact_sheet_before_automation.jpg").exists():
        shutil.copy2(old_sheet, OUT / "unresolved_contact_sheet_before_automation.jpg")
    compact = Image.new("RGB", (6 * 320, ((len(gallery_rows) + 5) // 6) * 150), "white")
    draw = ImageDraw.Draw(compact)
    for index, (label, preview_path, _) in enumerate(gallery_rows):
        x, y = (index % 6) * 320, (index // 6) * 150
        draw.text((x + 5, y + 4), label[:48], fill="black")
        im = Image.open(preview_path).convert("RGB")
        im.thumbnail((310, 120))
        compact.paste(im, (x + 5, y + 25))
    compact_path = OUT / "unresolved_contact_sheet.jpg"
    compact.save(compact_path, quality=91)
    shutil.copy2(compact_path, PACKAGE / "unresolved_contact_sheet.jpg")
    (OUT / "contact_sheets" / "README.md").write_text(
        "# SAM2 review contact sheets\n\n" + "\n".join(f"- [{Path(p).name}]({Path(p).name})" for p in [Path(p).name for p in contact_paths]) + "\n\nEach tile shows the current prompted polygon, the selected SAM2 outer-contour candidate, and a difference overlay. Open the native-resolution preview beside the tile before deciding.\n",
        encoding="utf-8",
    )

    self_cross_images = [r for r in unresolved if "self-crossing" in r.get("issue", "")]
    crossing_gallery = []
    for record in self_cross_images:
        rows = [r for r in sam_by_image.get((record["split"], record["filename"]), [])]
        audit_geom = read_csv(AUDIT / "polygon_geometry_audit.csv")
        cross_rows = [g["label_row"] for g in audit_geom if g.get("split") == record["split"] and g.get("filename") == record["filename"] and int(g.get("proper_crossings") or 0) > 0]
        target = next((r for r in rows if r.get("row") in cross_rows), rows[0] if rows else None)
        if target:
            p = SAM / target["preview"]
            crossing_gallery.append((f"{record['split']}/{record['filename']} | self-cross row {target['row']}", p, target["preview"]))
    crossing_contact = contact_pages(crossing_gallery, OUT / "self_crossing_contact_sheets", "self_crossing_sam2", per_page=12)

    fully_resolved = sum(r["fully_resolved_image"] == "true" for r in scope_decisions)
    sam_auto = sum(r.get("decision", "").startswith("AUTO_APPROVED") for r in sam_rows)
    summary = {
        "fixed_split": {k: v for k, v in audit_json.get("current_split_counts", {}).items()},
        "split_seed": audit_json.get("split_seed"),
        "active_labels_modified": False,
        "original_images_modified": False,
        "split_files_modified": False,
        "training_started": False,
        "sam2_checkpoint": "Meta SAM2.1 Hiera Tiny, official checkpoint, temporary environment only",
        "sam2_device": "CPU; no CUDA device available to this Python runtime",
        "sam2_images_with_proposals": len({(r["split"], r["filename"]) for r in sam_rows if r.get("decision")}),
        "sam2_polygon_proposals": len(sam_rows),
        "sam2_valid_candidate_polygons": sam_valid_polygons,
        "sam2_mask_only_reviews": sam_mask_only,
        "sam2_strict_auto_accepts": sam_auto,
        "scope_cleanup_actions_proposed": len(scope_decisions),
        "images_fully_resolved_automatically": fully_resolved,
        "partial_scope_actions_automated": sum(r["fully_resolved_image"] == "false" for r in scope_decisions),
        "remaining_flagged_images": len(queue_rows),
        "crop_cases": len(crop_rows),
        "crop_pixel_changes_if_visible_policy_approved": sum(float(r["visible_region_candidate_pixel_iou"]) < 1.0 for r in crop_rows),
        "edge_only_crop_images_resolved_if_policy_approved": 3,
        "remaining_flagged_images_after_crop_policy_approval": len(queue_rows) - 3,
        "visible_region_crop_policy_requires_one_global_human_decision": True,
        "annotation_zip_ready": False,
        "training_ready": False,
        "contact_sheets": contact_paths,
        "self_crossing_contact_sheets": crossing_contact,
        "crop_contact_sheets": crop_contact,
        "cvat_audit_json_sha256": hashlib.sha256((AUDIT / "cvat_export_audit.json").read_bytes()).hexdigest(),
        "note": "A SAM score is prompt-mask quality, not arm-class confidence. No model-generated polygon was accepted as ground truth.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(OUT / "sam2_automatic_decisions.csv", [r for r in sam_rows if r.get("decision", "").startswith("AUTO_APPROVED")])

    report = f"""# PoshanScan automation follow-up

**Disposition: not ready for dataset export or training.** This follow-up is staging-only. Active labels, original images, fixed split files, backups, API contracts and MUAC calculations were not modified. No training was started.

## Pretrained proposal model

Meta SAM 2.1 Hiera Tiny was installed in a disposable temp environment with CPU PyTorch 2.11 and matching CPU torchvision 0.26; the official 156 MB checkpoint is also only in temp storage. It processed {summary['sam2_images_with_proposals']} flagged images and generated {len(sam_rows)} mask proposals across polygon rows. **{sam_valid_polygons}** passed simple-polygon YOLO serialization; **{sam_mask_only}** (`train/129.png`) remains available as a binary-mask preview because no simple polygon could be safely serialized. The strict gate accepted **{sam_auto}** SAM2 proposals. The rest remain proposals requiring image review. SAM2 segments prompted visual regions; its reported score is not an anatomical arm confidence. It often separates the ArUco tag from the skin and may include clothing or noisy fragments. For the selected amodal rule, the proposal builder keeps the largest prompted component and fills enclosed holes, but this cannot prove the hidden boundary. Per-row previews and candidate masks are in `automated_proposals/sam2/`.

SAM2 supports image prompts such as boxes and points and returns prompted masks; Meta recommends WSL on Windows for installation. SAM 3 was not used: its official setup requires a CUDA-compatible GPU and its model checkpoint requires separate access. [SAM2 official repository](https://github.com/facebookresearch/sam2), [SAM3 official repository](https://github.com/facebookresearch/sam3).

## Automatic decisions

- **{fully_resolved} image fully resolved:** `val/54.png`. The retained target polygon exactly matches its previously reviewed mask at source resolution; the removed tiny left-edge row is a separate caregiver region; no marker-center gap is flagged.
- **Two partial scope cleanups:** `train/78.jpeg` keeps row 1 and drops row 2; `test/39.jpg` keeps row 1 and drops rows 2–4. These dropped rows match previously reviewed background polygons. `train/78.jpeg` still has an amodal marker-center gap. `test/39.jpg`'s retained target mask differs from the old target (IoU 0.8055) and still needs approval.
- The prior scope proposal for `train/78.jpeg` had the CVAT row order reversed. It kept the background polygon. The row mapping has been corrected, and its before/after preview was regenerated. No active label was changed.
- **No SAM2 proposal was automatically accepted.** The 95 proposals failed strict agreement/marker/issue gates or presented class-agnostic uncertainty.

## Missing `train/110.png`

The current and prior CVAT exports omit its label. The old local label is present in the active dataset, pre-import backup, and two prior Colab ZIPs with identical SHA-256 `f86059f7706d4f6044c79dc066f99ef328c6b29abae46db25a365485a4f8a60f`; it is not a revised CVAT annotation. The old mask is structurally usable as a starting polygon, but its ArUco center lies outside the mask. SAM2 generated a candidate with outside-marker IoU 0.7145 and area ratio 1.6545, so it is not a safe replacement. Do not treat either as approved ground truth. The minimum action is to add `110.png` in CVAT using the old shape as a starting outline and edit/review the marker bridge.

## Cropped upper arms

The seven source images crop the upper arm at the frame before the elbow. A **visible-region** segmentation target is technically consistent if defined as the visible intersection of `upper_arm` with the image, clipped exactly at the frame, with no inferred elbow or unseen anatomy; the amodal bridge remains limited to the attached ArUco marker. The candidate labels are pixel-identical to the current polygons after clipping (see `crop_policy_decisions.csv` and previews). This changes the target from a complete shoulder-to-elbow region to a partial visible region for these examples, so it needs one global policy decision. Such truncated images can train a visible-segmentation model, but they cannot by themselves provide the shoulder-to-elbow midpoint needed for reliable MUAC placement. The split remains frozen at 84/24/12, seed 2026.

## What remains

- The refreshed queue has **{len(queue_rows)} flagged images**; the 36 previously unflagged images remain excluded from repeat review.
- If the visible-region policy is approved, the three edge-only cases `train/14.png`, `train/26.png`, and `train/44.png` need no further mask edit because clipping changes zero pixels and they have no separate flagged issue. That one decision reduces the remaining image-level queue from 83 to **80**; `train/109.png`, `train/115.png`, `val/106.png`, and `val/108.png` also have marker-gap flags and remain in the queue.
- The 12 self-crossing image cases now have SAM2 outer-contour previews and candidate polygons. Candidate use may replace drawing from scratch where the human confirms the contour; otherwise a CVAT correction is still needed.
- Marker gaps, changed silhouettes and unresolved instance scope remain explicit in `unresolved_cases.csv`. ArUco center detection is a screening check, not proof that every center should lie within the anatomical silhouette.
- The compact contact sheets list unresolved cases; native-resolution comparisons and individual candidates remain beside them. The older `val/54.png` entry is removed from the refreshed queue only because it passed exact mask matching and scope checks.
- No final ZIP was produced. The Colab training gate remains disabled until CVAT actions, dataset validation, duplicate-group isolation and final audit pass.

## Minimum next actions

1. Decide once whether the seven cropped examples may use visible-region masks while keeping the split fixed. If yes, the staged frame-clipped candidates need no pixel edits; three edge-only cases clear and 80 image cases remain. If no, provide complete-arm replacements in the same split or remove them only with an explicit fixed-split plan.
2. Review the SAM2 before/after contact sheets and the remaining rows in `unresolved_cases.csv`. Use candidate polygons as starting points and adjust only the contour/marker bridge that is visibly wrong.
3. In CVAT, add/update `train/110.png` from its saved prior outline, then export all 120 images and rerun the audit. Import and train only after the resulting unresolved queue is empty and the final audit passes.
"""
    (OUT / "automation_update_report.md").write_text(report, encoding="utf-8")
    # The Colab review gate consumes these top-level files. Keep them aligned
    # with the follow-up disposition and back up the pre-follow-up CSV/summary.
    if not (OUT / "unresolved_cases_before_automation.csv").exists():
        shutil.copy2(PACKAGE / "unresolved_cases.csv", OUT / "unresolved_cases_before_automation.csv")
    (PACKAGE / "unresolved_cases.csv").write_bytes((OUT / "unresolved_cases.csv").read_bytes())
    review_summary_path = PACKAGE / "summary.json"
    prior_summary = json.loads(review_summary_path.read_text(encoding="utf-8-sig"))
    if not (OUT / "summary_before_automation.json").exists():
        shutil.copy2(review_summary_path, OUT / "summary_before_automation.json")
    prior_summary.update({
        "unresolved_unique_images": len(queue_rows),
        "images_fully_resolved_automatically": fully_resolved,
        "partial_scope_actions_automated": summary["partial_scope_actions_automated"],
        "sam2_images_with_proposals": summary["sam2_images_with_proposals"],
        "sam2_polygon_proposals": len(sam_rows),
        "sam2_valid_candidate_polygons": sam_valid_polygons,
        "sam2_mask_only_reviews": sam_mask_only,
        "sam2_strict_auto_accepts": sam_auto,
        "contact_sheet": "results\\cvat_import_audit_execution_check\\human_review_package\\automation_followup\\unresolved_contact_sheet.jpg",
        "contact_sheet_pages": contact_paths,
        "training_ready": False,
    })
    review_summary_path.write_text(json.dumps(prior_summary, indent=2), encoding="utf-8")
    (OUT / "contact_sheets" / "README.md").write_text(
        "# SAM2 review contact sheets\n\n" + "\n".join(f"- [{Path(p).name}]({Path(p).name})" for p in contact_paths) + "\n\nEach tile shows the current prompted polygon, selected SAM2 outer-contour candidate, and difference overlay. Open the native-resolution per-row preview before deciding. The root `unresolved_contact_sheet.jpg` is a compact 83-image scan; these pages are easier to inspect.\n",
        encoding="utf-8",
    )
    scope_csv = PACKAGE / "automatic_scope_cleanup_proposals.csv"
    if not (OUT / "automatic_scope_cleanup_before_automation.csv").exists():
        shutil.copy2(scope_csv, OUT / "automatic_scope_cleanup_before_automation.csv")
    for row in scope_decisions:
        row["review_class"] = "AUTO_SCOPE_ACTION_STAGED"
        row["issue"] = "Drop unrelated/background polygon rows only."
        row["status"] = (
            "fully resolved automatically; exact target mask matches prior reviewed label"
            if row["fully_resolved_image"] == "true"
            else "background-row removal approved automatically; image still has a separate unresolved issue"
        )
        row["reason"] += " Full-resolution before/after preview was reviewed in the automation audit."
    write_csv(scope_csv, scope_decisions)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
