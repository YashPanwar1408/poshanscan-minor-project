"""Build a compact, read-only human review package for the current CVAT export.

This utility never edits active labels, images, split files, or the CVAT export.
It consumes the staged CVAT labels and writes only beneath the audit report
directory.  It is intentionally not a training/import command.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "yolo_working"
DEFAULT_STAGE = ROOT / "results" / "cvat_import_staging_execution_check"
DEFAULT_AUDIT = ROOT / "results" / "cvat_import_audit_execution_check"

EDGE_CASES = {
    ("train", "14.png"), ("train", "26.png"), ("train", "44.png"),
    ("train", "109.png"), ("train", "115.png"), ("val", "54.png"),
    ("val", "106.png"), ("val", "108.png"),
}
MULTI_CASES = {
    ("train", "69.png"), ("train", "70.png"), ("train", "78.jpeg"),
    ("val", "34.jpg"), ("val", "35.jpg"), ("val", "54.png"),
    ("val", "68.png"), ("val", "74.png"), ("test", "39.jpg"),
}
SCOPE_CLEANUP = {
    # 1-based CVAT row numbers to keep and drop; suggestions are review-only.
    # CVAT row order differs from the previous active annotation in train/78.jpeg.
    # Pixel comparison confirmed current row 1 is the target arm (IoU 1.0 with
    # the old target) and current row 2 is the unrelated background arm.
    ("train", "78.jpeg"): {"keep": [1], "drop": [2], "confidence": "0.99", "reason": "Current CVAT row 1 is pixel-identical to the previously reviewed target arm; current row 2 is the small background child's arm (the CVAT export reordered the rows). The marker-gap issue remains a separate review item."},
    ("val", "54.png"): {"keep": [2], "drop": [1], "confidence": "0.93", "reason": "The retained polygon is the marker-bearing infant's arm; the tiny separate polygon at the far-left edge is on the caregiver/foreground, outside the selected subject."},
    ("test", "39.jpg"): {"keep": [1], "drop": [2, 3, 4], "confidence": "0.99", "reason": "The retained polygon is the marker-bearing foreground child's arm; the other three polygons fall on separate background people."},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_polygons(path: Path, width: int, height: int) -> list[np.ndarray]:
    polygons: list[np.ndarray] = []
    if not path.exists():
        return polygons
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 7 or (len(values) - 1) % 2:
            continue
        xy = np.asarray([float(v) for v in values[1:]], dtype=np.float64).reshape(-1, 2)
        xy[:, 0] *= width
        xy[:, 1] *= height
        # Use the same truncating pixel conversion as the existing polygon audit.
        polygons.append(xy.astype(np.int32))
    return polygons


def raster_polygon(path: Path, row_number: int, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    polygons = read_polygons(path, width, height)
    if 0 <= row_number - 1 < len(polygons):
        cv2.fillPoly(mask, [polygons[row_number - 1]], 255)
    return mask


def make_overlay(image: np.ndarray, polygons: list[np.ndarray], marker_boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    result = image.copy()
    if polygons:
        layer = result.copy()
        cv2.fillPoly(layer, polygons, (30, 90, 230))
        result = cv2.addWeighted(layer, 0.2, result, 0.8, 0)
        cv2.polylines(result, polygons, True, (40, 40, 255), 3, cv2.LINE_AA)
    for x, y, w, h in marker_boxes:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 230, 255), 3)
        cv2.circle(result, (x + w // 2, y + h // 2), 5, (0, 255, 255), -1)
    return result


def detect_markers(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    try:
        aruco = cv2.aruco
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        params = aruco.DetectorParameters()
        corners, _, _ = aruco.ArucoDetector(dictionary, params).detectMarkers(image)
        return [cv2.boundingRect(c.astype(np.float32)) for c in corners]
    except (AttributeError, cv2.error):
        return []


def crop_to_polygons(image: np.ndarray, polygons: list[np.ndarray]) -> np.ndarray:
    h, w = image.shape[:2]
    points = np.concatenate(polygons, axis=0) if polygons else np.asarray([[0, 0], [w - 1, h - 1]])
    x0, y0 = points.min(axis=0).astype(int)
    x1, y1 = points.max(axis=0).astype(int)
    margin_x = max(60, int((x1 - x0 + 1) * 0.16))
    margin_y = max(60, int((y1 - y0 + 1) * 0.12))
    x0, y0 = max(0, x0 - margin_x), max(0, y0 - margin_y)
    x1, y1 = min(w, x1 + margin_x + 1), min(h, y1 + margin_y + 1)
    return image[y0:y1, x0:x1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    output = args.audit / "human_review_package"
    output.mkdir(parents=True, exist_ok=True)
    (output / "polygon_repairs").mkdir(exist_ok=True)

    geometry_rows = read_csv(args.audit / "polygon_geometry_audit.csv")
    pair_rows = read_csv(args.audit / "image_label_pair_audit.csv")
    marker_rows = read_csv(args.audit / "aruco_marker_gap_scan.csv")
    proposal_rows = read_csv(args.audit / "geometric_repair_candidates.csv")

    crossings: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in geometry_rows:
        if int(row.get("proper_crossings") or 0) > 0:
            crossings[(row["split"], row["filename"])].append(row)
    markers: dict[tuple[str, str], dict[str, str]] = {}
    for row in marker_rows:
        key = (row["split"], row["filename"])
        if row.get("marker_center_covered", "").lower() == "false":
            markers[key] = row
    metadata: dict[tuple[str, str], dict[str, str]] = {
        (row["split"], row["filename"]): row for row in pair_rows
    }
    missing_labels = {
        key for key, row in metadata.items()
        if row.get("label_present", "").lower() == "false"
    }
    changed_labels = {
        key for key, row in metadata.items()
        if row.get("label_vs_previous", "").lower() == "changed"
    }
    # Ignore label-file changes that are only row-order/serialization changes.
    semantic_changes: set[tuple[str, str]] = set()
    for key in changed_labels:
        split, filename = key
        image = cv2.imread(str(DATA / "images" / split / filename))
        if image is None:
            semantic_changes.add(key)
            continue
        height, width = image.shape[:2]
        old = read_polygons(DATA / "labels" / split / f"{Path(filename).stem}.txt", width, height)
        current = read_polygons(args.stage / "labels" / split / f"{Path(filename).stem}.txt", width, height)
        old_digests, current_digests = [], []
        for polygons, digest_list in ((old, old_digests), (current, current_digests)):
            for polygon in polygons:
                mask = np.zeros((height, width), np.uint8)
                cv2.fillPoly(mask, [polygon], 255)
                digest_list.append(hashlib.sha256(mask.tobytes()).hexdigest())
        if sorted(old_digests) != sorted(current_digests):
            semantic_changes.add(key)
    proposals = {
        (r["split"], r["filename"]): r for r in proposal_rows
        if r.get("proposed_vertices")
        and (args.audit / "proposed_geometric_repairs" / "labels" / r["split"] / f"{Path(r['filename']).stem}.txt").exists()
    }
    verified_proposals: dict[tuple[str, str], dict[str, str]] = {}
    proposal_checks: list[dict[str, str]] = []
    for key, record in proposals.items():
        split, filename = key
        image = cv2.imread(str(DATA / "images" / split / filename))
        if image is None:
            continue
        height, width = image.shape[:2]
        row_number = int(record["label_row"])
        staged_path = args.stage / "labels" / split / f"{Path(filename).stem}.txt"
        proposed_path = args.audit / "proposed_geometric_repairs" / "labels" / split / f"{Path(filename).stem}.txt"
        before = raster_polygon(staged_path, row_number, width, height)
        after = raster_polygon(proposed_path, row_number, width, height)
        mismatch_pixels = int(np.count_nonzero(before != after))
        exact = mismatch_pixels == 0
        if exact:
            verified_proposals[key] = record
        proposal_checks.append({
            "split": split, "filename": filename, "label_row": str(row_number),
            "pixel_mismatch_count": str(mismatch_pixels),
            "pixel_iou": f"{float(np.count_nonzero((before > 0) & (after > 0)) / max(1, np.count_nonzero((before > 0) | (after > 0)))):.12f}",
            "exact_pixel_mask_match": str(exact).lower(),
            "classification": "A_exact_mask_repair_proposal" if exact else "C_manual_redraw",
            "reason": "Identical fill at source resolution." if exact else "The proposal changes rasterized pixels; it is not safe for automatic application."
        })
    with (args.audit / "geometric_repair_candidates.csv").open("w", newline="", encoding="utf-8-sig") as f:
        columns = list(dict.fromkeys(list(proposal_rows[0].keys()) + ["pixel_mismatch_count", "exact_pixel_mask_match"]))
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for original in proposal_rows:
            row = dict(original)
            check = next((c for c in proposal_checks if c["split"] == row["split"] and c["filename"] == row["filename"]), None)
            if check:
                row["classification"] = check["classification"]
                row["reason"] = check["reason"]
                row["mask_iou"] = check["pixel_iou"]
                row["pixel_mismatch_count"] = check["pixel_mismatch_count"]
                row["exact_pixel_mask_match"] = check["exact_pixel_mask_match"]
            else:
                row["pixel_mismatch_count"] = ""
                row["exact_pixel_mask_match"] = ""
            writer.writerow(row)
    with (output / "geometric_proposal_pixel_check.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(proposal_checks[0]) if proposal_checks else [
            "split", "filename", "label_row", "pixel_mismatch_count", "pixel_iou", "exact_pixel_mask_match", "classification", "reason"
        ])
        writer.writeheader()
        writer.writerows(proposal_checks)

    # Measure 8-connected components in each individual YOLO polygon fill.
    # Multiple polygon rows are audited separately as distinct instances.
    component_rows: list[dict[str, str]] = []
    disconnected_images: set[tuple[str, str]] = set()
    for key, pair in sorted(metadata.items()):
        split, filename = key
        image = cv2.imread(str(DATA / "images" / split / filename))
        if image is None:
            continue
        height, width = image.shape[:2]
        label_path = args.stage / "labels" / split / f"{Path(filename).stem}.txt"
        polygons = read_polygons(label_path, width, height)
        if not polygons:
            component_rows.append({
                "split": split, "filename": filename, "polygon_row": "", "component_count": "0",
                "component_pixel_areas": "", "interpretation": "No staged polygon; missing label is unresolved."
            })
            continue
        for row_number, polygon in enumerate(polygons, start=1):
            mask = np.zeros((height, width), np.uint8)
            cv2.fillPoly(mask, [polygon], 255)
            count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            areas = [int(value) for value in stats[1:count, cv2.CC_STAT_AREA]]
            component_count = count - 1
            interpretation = "Single connected raster component."
            if component_count > 1:
                disconnected_images.add(key)
                interpretation = "One polygon row rasterizes to multiple disconnected 8-connected components; redraw or adjudicate before import."
            component_rows.append({
                "split": split, "filename": filename, "polygon_row": str(row_number),
                "component_count": str(component_count),
                "component_pixel_areas": ";".join(map(str, areas)),
                "interpretation": interpretation,
            })
    with (output / "polygon_component_audit.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(component_rows[0]) if component_rows else [
            "split", "filename", "polygon_row", "component_count", "component_pixel_areas", "interpretation"
        ])
        writer.writeheader()
        writer.writerows(component_rows)

    issue_map: dict[tuple[str, str], set[str]] = defaultdict(set)
    for key in crossings:
        issue_map[key].add("self-crossing polygon")
    for key in EDGE_CASES:
        issue_map[key].add("polygon touches image boundary")
    for key in MULTI_CASES:
        issue_map[key].add("multiple polygons")
    for key in markers:
        issue_map[key].add("detected ArUco center outside mask")
    for key in missing_labels:
        issue_map[key].add("missing CVAT annotation")
    for key in semantic_changes:
        issue_map[key].add("changed from previously reviewed label")
    for key in disconnected_images:
        issue_map[key].add("polygon fill has disconnected raster components")

    rows: list[dict[str, str]] = []
    sheet_items: list[tuple[str, str, np.ndarray]] = []
    for key, issue_set in sorted(issue_map.items()):
        split, filename = key
        repair = verified_proposals.get(key)
        severe_crossing = key in crossings and repair is None
        has_edge = key in EDGE_CASES
        edge_requires_dataset_decision = has_edge and key != ("val", "54.png")
        has_multi = key in MULTI_CASES
        has_gap = key in markers
        if key in missing_labels:
            classification = "C_draw_missing_polygon"
            confidence = "high that a new annotation is required; no valid updated label was found"
            correction = "Annotate the complete upper arm in CVAT from the source image; do not copy the older local label as a fallback."
            reason = "The current and previous CVAT exports omit this label. The local copy is an older active annotation, not a recovered CVAT correction."
        elif severe_crossing or key in disconnected_images:
            classification = "C_polygon_redraw"
            confidence = "high that redraw is required; intended silhouette cannot be resolved geometrically"
            correction = "Redraw a single simple upper-arm silhouette in CVAT; preserve the source image and fixed split."
            reason = "The polygon is self-crossing or rasterizes into disconnected components; alternative geometric repair could change the intended arm boundary."
        elif edge_requires_dataset_decision:
            classification = "B_dataset_eligibility_decision"
            confidence = "high; source image visibly cuts off before the elbow"
            correction = "Do not trace to the image border. Supply a full-arm replacement for this split, or explicitly authorize excluding/reassigning it while preserving duplicate isolation and counts."
            reason = "At native resolution the bottom frame cuts off the arm before the elbow. A complete shoulder-to-elbow polygon cannot be drawn from the available pixels."
        elif has_multi:
            classification = "B_yes_no_decision"
            confidence = "medium; intended subject/instance scope needs confirmation"
            correction = "For each polygon, confirm keep/drop as an in-scope upper arm; merge same-arm pieces only if the underlying arm is continuous."
            reason = "The class and image alone do not identify whether separate rows are two arms, a marker-split arm, or an unrelated person/object."
        elif key in semantic_changes:
            classification = "B_yes_no_decision"
            confidence = "medium; compare against the previously reviewed label"
            correction = "Confirm the CVAT change follows the amodal shoulder-to-elbow policy and is intentional."
            reason = "This label differs from the active label that received the previous review; the updated outline needs a focused before/after check."
        elif has_gap:
            classification = "B_yes_no_decision"
            confidence = "medium; marker is detected, attachment/arm footprint needs visual confirmation"
            correction = "Confirm the marker is attached to the labeled arm and visible arm contour supports a continuous amodal bridge beneath it."
            reason = "The detected marker center falls outside every current polygon. Center coverage is a screening heuristic; it cannot prove the marker's arm footprint or outer contour."
        else:
            classification = "A_exact_mask_repair_proposal"
            confidence = "high for raster equivalence; no active label changed"
            correction = "Use the separate simple-contour proposal; it reproduces the CVAT raster mask exactly."
            reason = "A single connected raster mask has a simple contour whose filled pixels are exactly identical; proposal is isolated from active annotations."
        pair = metadata.get(key, {})
        detail_parts = []
        if key in crossings:
            detail_parts.append(f"{len(crossings[key])} crossing polygon row(s)")
        if has_edge:
            detail_parts.append("edge-touching polygon")
        if has_multi:
            detail_parts.append("multiple polygon rows")
        if has_gap:
            detail_parts.append("marker center outside mask")
        if key in semantic_changes:
            detail_parts.append("label changed since prior review")
        rows.append({
            "split": split,
            "filename": filename,
            "issue": "; ".join(sorted(issue_set)),
            "review_class": classification,
            "automatic_scope_cleanup_proposal": "available" if key in SCOPE_CLEANUP else "",
            "automatic_scope_cleanup_confidence": SCOPE_CLEANUP.get(key, {}).get("confidence", ""),
            "proposed_correction": correction,
            "confidence": confidence,
            "reason": reason,
            "polygon_count": pair.get("polygon_count", ""),
            "label_status": pair.get("label_vs_previous", ""),
        })

        # Exact pixel-preserving repairs need no visual yes/no decision, unless
        # that same image has a separate boundary, scope, or marker issue.
        if classification == "A_exact_mask_repair_proposal" and not issue_set.intersection({
            "polygon touches image boundary", "multiple polygons", "detected ArUco center outside mask"
        }):
            continue
        image_path = DATA / "images" / split / filename
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        label_path = args.stage / "labels" / split / f"{Path(filename).stem}.txt"
        polygons = read_polygons(label_path, image.shape[1], image.shape[0])
        marked = make_overlay(image, polygons, detect_markers(image))
        crop = crop_to_polygons(marked, polygons)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        thumb = Image.fromarray(rgb)
        thumb.thumbnail((224, 146), Image.Resampling.LANCZOS)
        sheet_items.append((f"{split}/{filename}", ", ".join(sorted(issue_set)), np.asarray(thumb)))

    csv_path = output / "unresolved_cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [
            "split", "filename", "issue", "review_class", "automatic_scope_cleanup_proposal", "automatic_scope_cleanup_confidence", "proposed_correction", "confidence", "reason", "polygon_count", "label_status"
        ])
        writer.writeheader()
        writer.writerows(rows)

    # Compact single sheet, containing only unresolved B/C cases.
    columns, tile_w, tile_h, label_h = 8, 244, 186, 40
    count = len(sheet_items)
    sheet = Image.new("RGB", (columns * tile_w, ((count + columns - 1) // columns) * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (name, issues, arr) in enumerate(sheet_items):
        x = (i % columns) * tile_w
        y = (i // columns) * (tile_h + label_h)
        tile = Image.fromarray(arr)
        sheet.paste(tile, (x + (tile_w - tile.width) // 2, y + (tile_h - tile.height) // 2))
        draw.text((x + 4, y + tile_h), name, fill="black")
        draw.text((x + 4, y + tile_h + 16), issues[:42], fill=(110, 25, 25))
    sheet_path = output / "unresolved_contact_sheet.jpg"
    sheet.save(sheet_path, quality=92)

    # Compare source/previous/CVAT/contour-candidate for every attempted
    # geometric transformation, including rejected candidates.
    comparison_rows = []
    for key, record in sorted(proposals.items()):
        split, filename = key
        source = cv2.imread(str(DATA / "images" / split / filename))
        if source is None:
            continue
        h, w = source.shape[:2]
        base = DATA / "labels" / split / f"{Path(filename).stem}.txt"
        staged = args.stage / "labels" / split / f"{Path(filename).stem}.txt"
        proposed = args.audit / "proposed_geometric_repairs" / "labels" / split / f"{Path(filename).stem}.txt"
        base_polys, staged_polys, proposed_polys = (
            read_polygons(p, w, h) for p in (base, staged, proposed)
        )
        old_mask = np.zeros((h, w), np.uint8)
        staged_mask = np.zeros((h, w), np.uint8)
        proposed_mask = np.zeros((h, w), np.uint8)
        if staged_polys:
            cv2.fillPoly(staged_mask, [staged_polys[int(record["label_row"]) - 1]], 255)
        if proposed_polys:
            # Repair proposals preserve row order and update the listed row.
            row_no = int(record["label_row"]) - 1
            cv2.fillPoly(proposed_mask, [proposed_polys[row_no]], 255)
        exact = bool(np.array_equal(staged_mask, proposed_mask))
        if base_polys:
            cv2.fillPoly(old_mask, [base_polys[min(int(record["label_row"]) - 1, len(base_polys) - 1)]], 255)
        panel_images = []
        for title, mask in (("Previous active label", old_mask), ("CVAT export", staged_mask), ("Contour candidate: " + ("EXACT" if exact else "REJECTED"), proposed_mask)):
            panel = source.copy()
            panel[mask > 0] = (panel[mask > 0] * 0.72 + np.array([20, 100, 240]) * 0.28).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(panel, contours, -1, (30, 30, 255), 3)
            cv2.putText(panel, title, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(panel, title, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
            panel_images.append(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        max_width = 700
        resized = []
        for panel in panel_images:
            ph, pw = panel.shape[:2]
            resized.append(cv2.resize(panel, (max_width, round(ph * max_width / pw))))
        comparison = np.concatenate(resized, axis=1)
        status = "exact" if exact else "rejected_mask_changed"
        out = output / "polygon_repairs" / f"{split}_{Path(filename).stem}_{status}_before_after.jpg"
        Image.fromarray(comparison).save(out, quality=94)
        comparison_rows.append({
            "split": split, "filename": filename, "label_row": record["label_row"],
            "exact_pixel_mask_match": str(exact).lower(),
            "status": "review-only exact candidate" if exact else "rejected; do not import",
            "pixel_mismatch_count": next((c["pixel_mismatch_count"] for c in proposal_checks if c["split"] == split and c["filename"] == filename), ""),
            "repair_label": str(proposed.relative_to(ROOT)),
            "comparison_preview": str(out.relative_to(ROOT)),
            "note": "Candidate only; no active labels were changed. Non-identical candidates are excluded from automatic repair."
        })
    with (output / "polygon_repairs.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0]) if comparison_rows else [
            "split", "filename", "label_row", "exact_pixel_mask_match", "status", "pixel_mismatch_count", "repair_label", "comparison_preview", "note"
        ])
        writer.writeheader()
        writer.writerows(comparison_rows)

    scope_dir = output / "proposed_scope_cleanup"
    label_root = scope_dir / "labels"
    preview_root = scope_dir / "previews"
    label_root.mkdir(parents=True, exist_ok=True)
    preview_root.mkdir(parents=True, exist_ok=True)
    scope_rows = []
    for (split, filename), plan in SCOPE_CLEANUP.items():
        image_path = DATA / "images" / split / filename
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        source_label = args.stage / "labels" / split / f"{Path(filename).stem}.txt"
        source_lines = [line for line in source_label.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(row < 1 or row > len(source_lines) for row in plan["keep"] + plan["drop"]):
            raise ValueError(f"Scope cleanup row numbers do not match {split}/{filename}")
        planned_lines = [source_lines[row - 1] for row in plan["keep"]]
        proposed_label = label_root / split / f"{Path(filename).stem}.txt"
        proposed_label.parent.mkdir(parents=True, exist_ok=True)
        proposed_label.write_text("\n".join(planned_lines) + "\n", encoding="utf-8")

        h, w = image.shape[:2]
        old_polys = read_polygons(DATA / "labels" / split / f"{Path(filename).stem}.txt", w, h)
        current_polys = read_polygons(source_label, w, h)
        proposed_polys = read_polygons(proposed_label, w, h)
        marker_boxes = detect_markers(image)
        panels = [
            ("Previous active", old_polys),
            ("CVAT current", current_polys),
            ("Proposed scope cleanup", proposed_polys),
        ]
        preview_panels = []
        for title, polygons in panels:
            panel = make_overlay(image, polygons, marker_boxes)
            cv2.putText(panel, title, (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(panel, title, (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            panel = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
            panel_h, panel_w = panel.shape[:2]
            preview_panels.append(cv2.resize(panel, (680, round(panel_h * 680 / panel_w))))
        preview = np.concatenate(preview_panels, axis=1)
        preview_path = preview_root / f"{split}_{Path(filename).stem}_scope_cleanup_before_after.jpg"
        Image.fromarray(preview).save(preview_path, quality=94)
        scope_rows.append({
            "split": split,
            "filename": filename,
            "review_class": "A_high_confidence_scope_cleanup_proposal",
            "issue": "separate polygon(s) belong to background/unrelated person; retain marker-bearing target arm only",
            "proposed_action": f"Keep CVAT rows {','.join(map(str, plan['keep']))}; drop rows {','.join(map(str, plan['drop']))}.",
            "confidence": plan["confidence"],
            "reason": plan["reason"],
            "proposed_label": str(proposed_label.relative_to(ROOT)),
            "comparison_preview": str(preview_path.relative_to(ROOT)),
            "status": "staged proposal; human approval required before import",
        })
    with (output / "automatic_scope_cleanup_proposals.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(scope_rows[0]) if scope_rows else [
            "split", "filename", "review_class", "issue", "proposed_action", "confidence", "reason", "proposed_label", "comparison_preview", "status"
        ])
        writer.writeheader()
        writer.writerows(scope_rows)

    eligibility_rows = []
    for split, filename in sorted(EDGE_CASES):
        if (split, filename) == ("val", "54.png"):
            eligibility_rows.append({
                "split": split,
                "filename": filename,
                "native_resolution_finding": "The edge-touching row is the tiny left-edge polygon, not the selected infant arm.",
                "recommended_action": "Review the proposed removal of the unrelated row; keep the marker-bearing arm. No image exclusion is indicated by this edge touch alone.",
                "proposal": "val_54 scope cleanup preview",
            })
        else:
            eligibility_rows.append({
                "split": split,
                "filename": filename,
                "native_resolution_finding": "Image bottom crops the marked arm before the elbow is visible.",
                "recommended_action": "Do not invent the elbow boundary or end the mask at the frame. Provide a complete-arm replacement in the same split, or explicitly authorize an eligibility/split decision.",
                "proposal": "No polygon repair proposed; the missing anatomy is not present in the image.",
            })
    with (output / "dataset_eligibility_decisions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(eligibility_rows[0]))
        writer.writeheader()
        writer.writerows(eligibility_rows)

    marker_count = len(markers)
    unresolved_count = len(sheet_items)
    summary = {
        "staged_only": True,
        "active_annotations_modified": False,
        "split_files_modified": False,
        "cvat_audit_json_sha256": hashlib.sha256((args.audit / "cvat_export_audit.json").read_bytes()).hexdigest(),
        "unresolved_unique_images": unresolved_count,
        "polygon_rows_component_audited": len(component_rows),
        "disconnected_polygon_rows": sum(int(row["component_count"]) > 1 for row in component_rows),
        "images_with_disconnected_polygon_rows": len(disconnected_images),
        "marker_centers_outside_masks": marker_count,
        "exact_mask_preserving_repair_proposals": sum(1 for r in comparison_rows if r["exact_pixel_mask_match"] == "true"),
        "geometric_contour_candidates_compared": len(comparison_rows),
        "rejected_nonidentical_geometry_proposals": sum(1 for row in proposal_checks if row["exact_pixel_mask_match"] == "false"),
        "high_confidence_scope_cleanup_proposals": len(scope_rows),
        "images_cropped_before_elbow": sum(1 for r in eligibility_rows if "crops the marked arm" in r["native_resolution_finding"]),
        "edge_review_resolved_by_scope_proposal": sum(1 for r in eligibility_rows if "tiny left-edge" in r["native_resolution_finding"]),
        "all_geometric_candidates_pixel_identical": bool(comparison_rows) and all(r["exact_pixel_mask_match"] == "true" for r in comparison_rows),
        "contact_sheet": str(sheet_path.relative_to(ROOT)),
        "unresolved_csv": str(csv_path.relative_to(ROOT)),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
