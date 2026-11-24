"""Generate SAM2 review-only mask proposals from staged YOLO polygons.

This script never edits source images, active labels, split files, or CVAT
exports. SAM2 is class-agnostic; all outputs are proposals requiring review.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "yolo_working"
DEFAULT_AUDIT = ROOT / "results" / "cvat_import_audit_execution_check"
DEFAULT_OUT = DEFAULT_AUDIT / "human_review_package" / "automated_proposals" / "sam2"
DEFAULT_STAGE = ROOT / "results" / "cvat_import_staging_execution_check"
DEFAULT_CHECKPOINT = Path.home() / "AppData/Local/Temp/poshanscan_sam2_review/checkpoints/sam2.1_hiera_tiny.pt"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


def read_rows(path: Path) -> list[tuple[int, np.ndarray]]:
    rows = []
    if not path.exists():
        return rows
    for row_i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) < 7 or (len(fields) - 1) % 2 or fields[0] != "0":
            continue
        xy = np.asarray([float(v) for v in fields[1:]], dtype=np.float32).reshape(-1, 2)
        rows.append((row_i, xy))
    return rows


def detect_markers(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    try:
        a = cv2.aruco
        dictionary = a.getPredefinedDictionary(a.DICT_4X4_50)
        corners, _, _ = a.ArucoDetector(dictionary, a.DetectorParameters()).detectMarkers(image)
        return [cv2.boundingRect(c.astype(np.float32)) for c in corners]
    except (AttributeError, cv2.error):
        return []


def rasterize(points: np.ndarray, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    xy = (points * np.asarray([width, height], dtype=np.float32)).astype(np.int32)
    cv2.fillPoly(mask, [xy], 255)
    return mask


def prompts(seed: np.ndarray, points: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.clip((points[:, 0] * width).astype(int), 0, width - 1)
    y = np.clip((points[:, 1] * height).astype(int), 0, height - 1)
    x0, x1 = int(x.min()), int(x.max())
    y0, y1 = int(y.min()), int(y.max())
    pad_x, pad_y = max(4, int((x1 - x0 + 1) * 0.035)), max(4, int((y1 - y0 + 1) * 0.035))
    box = np.asarray([max(0, x0 - pad_x), max(0, y0 - pad_y), min(width - 1, x1 + pad_x), min(height - 1, y1 + pad_y)], dtype=np.float32)

    dist = cv2.distanceTransform((seed > 0).astype(np.uint8), cv2.DIST_L2, 5)
    if float(dist.max()) <= 0:
        positive = np.asarray([[int(x.mean()), int(y.mean())]], dtype=np.float32)
    else:
        py, px = np.unravel_index(int(dist.argmax()), dist.shape)
        positive = np.asarray([[px, py]], dtype=np.float32)
    # Negative clicks just outside the provided polygon box discourage SAM
    # from absorbing the sleeve/torso when those regions touch the prompt box.
    candidates = [
        (max(0, x0 - pad_x * 3), (y0 + y1) // 2),
        (min(width - 1, x1 + pad_x * 3), (y0 + y1) // 2),
        ((x0 + x1) // 2, max(0, y0 - pad_y * 3)),
        ((x0 + x1) // 2, min(height - 1, y1 + pad_y * 3)),
    ]
    neg = np.asarray(candidates, dtype=np.float32)
    good = np.asarray([seed[int(py), int(px)] == 0 for px, py in neg], dtype=bool)
    coords = np.concatenate([positive, neg[good]], axis=0)
    labels = np.asarray([1] + [0] * int(good.sum()), dtype=np.int32)
    return box, coords, labels


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.24) -> np.ndarray:
    out = image.copy()
    if np.any(mask):
        layer = out.copy()
        layer[mask > 0] = color
        out = cv2.addWeighted(layer, alpha, out, 1 - alpha, 0)
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2, cv2.LINE_AA)
    return out


def marker_exclusion(mask: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    keep = np.ones(mask.shape, dtype=np.uint8)
    for x, y, w, h in boxes:
        keep[max(0, y):min(mask.shape[0], y + h), max(0, x):min(mask.shape[1], x + w)] = 0
    return keep


def components(mask: np.ndarray) -> int:
    count, _, _, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    return int(count - 1)


def simple_polygon(points: list[tuple[float, float]]) -> bool:
    n = len(points)
    if n < 3:
        return False
    def orient(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        for j in range(i + 1, n):
            if j in (i, i + 1) or (i == 0 and j == n - 1):
                continue
            c, d = points[j], points[(j + 1) % n]
            o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
            if o1 * o2 < -1e-12 and o3 * o4 < -1e-12:
                return False
    return True


def largest_filled_component(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Drop SAM speckles, then fill internal holes for a single amodal outline."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(mask), 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    component_id = int(np.argmax(areas)) + 1
    largest = (labels == component_id).astype(np.uint8)
    contours, _ = cv2.findContours(largest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask), count - 1
    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
    return filled, count - 1


def to_yolo_polygon(mask: np.ndarray) -> str | None:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    h, w = mask.shape
    # Validate the exact six-decimal YOLO representation. If a simplification
    # crosses itself, try a coarser outline; never emit it as a candidate label.
    for fraction in (0.0015, 0.002, 0.003, 0.005, 0.008, 0.01):
        approx = cv2.approxPolyDP(contour, max(1.0, fraction * perimeter), True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        points = [
            (round(max(0.0, min(1.0, float(x) / w)), 6), round(max(0.0, min(1.0, float(y) / h)), 6))
            for x, y in approx
        ]
        if simple_polygon(points):
            coords = [value for point in points for value in point]
            return "0 " + " ".join(f"{v:.6f}" for v in coords)
    return None


def rebuild_candidate_labels(args: argparse.Namespace) -> None:
    proposal_csv = args.output / "sam2_proposals.csv"
    rows = list(csv.DictReader(proposal_csv.open(newline="", encoding="utf-8-sig")))
    rejected = args.output / "candidate_labels_rejected"
    rejected.mkdir(exist_ok=True)
    audit_rows = []
    for row in rows:
        mask_path = args.output / row["mask"]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        label_path = args.output / "candidate_labels" / f"{mask_path.stem}.txt"
        label_text = to_yolo_polygon(mask) if mask is not None else None
        if label_text:
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(label_text + "\n", encoding="utf-8")
            row["candidate_label"] = str(label_path.relative_to(args.output)).replace("\\", "/")
            row["candidate_label_valid"] = "true"
            row["candidate_label_geometry"] = "simple polygon; YOLO class 0"
            vertices = (len(label_text.split()) - 1) // 2
            status = "valid candidate polygon"
        else:
            if label_path.exists():
                shutil.move(str(label_path), str(rejected / label_path.name))
            row["candidate_label"] = ""
            row["candidate_label_valid"] = "false"
            row["candidate_label_geometry"] = "no simple serialization; use the mask preview only"
            vertices = 0
            status = "candidate polygon rejected; binary mask remains reviewable"
        audit_rows.append({
            "split": row["split"], "filename": row["filename"], "row": row["row"],
            "candidate_label": row["candidate_label"], "valid": row["candidate_label_valid"],
            "vertices": str(vertices), "status": status,
            "preview": row["preview"], "mask": row["mask"],
        })
    with proposal_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "sam2_candidate_label_audit.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit_rows[0]) if audit_rows else ["status"])
        writer.writeheader()
        writer.writerows(audit_rows)
    summary_path = args.output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        summary["valid_candidate_polygons"] = sum(r["valid"] == "true" for r in audit_rows)
        summary["invalid_candidate_polygons"] = sum(r["valid"] != "true" for r in audit_rows)
        summary["candidate_label_validation"] = "self-crossing serialized polygons excluded from candidate-label set"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "proposal_rows": len(audit_rows),
        "valid_candidate_polygons": sum(r["valid"] == "true" for r in audit_rows),
        "invalid_candidate_polygons": sum(r["valid"] != "true" for r in audit_rows),
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    ap.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--max-images", type=int, default=0, help="Optional smoke-test cap; 0 processes all unresolved images")
    ap.add_argument("--rebuild-labels-only", action="store_true", help="Regenerate/validate candidate polygons from saved masks without running SAM2 inference")
    args = ap.parse_args()

    if args.rebuild_labels_only:
        rebuild_candidate_labels(args)
        return

    unresolved = list(csv.DictReader((args.audit / "human_review_package" / "unresolved_cases.csv").open(newline="", encoding="utf-8-sig")))
    if args.max_images:
        keep = {(r["split"], r["filename"]) for r in unresolved[:args.max_images]}
        unresolved = [r for r in unresolved if (r["split"], r["filename"]) in keep]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "previews").mkdir(exist_ok=True)
    (args.output / "masks").mkdir(exist_ok=True)
    (args.output / "candidate_labels").mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(args.config, str(args.checkpoint), device=device)
    predictor = SAM2ImagePredictor(model)
    audit_rows: list[dict[str, str]] = []
    total = len(unresolved)

    for image_i, record in enumerate(unresolved, 1):
        split, filename = record["split"], record["filename"]
        image_path = DATA / "images" / split / filename
        image = cv2.imread(str(image_path))
        if image is None:
            audit_rows.append({"split": split, "filename": filename, "row": "", "status": "failed", "reason": "Original image could not be read."})
            continue
        h, w = image.shape[:2]
        staged = args.stage / "labels" / split / f"{Path(filename).stem}.txt"
        if staged.exists():
            label_path, prompt_source = staged, "staged_CVAT_export"
        else:
            label_path = DATA / "labels" / split / f"{Path(filename).stem}.txt"
            prompt_source = "previous_active_label_only_missing_from_CVAT"
        annotations = read_rows(label_path)
        if not annotations:
            audit_rows.append({"split": split, "filename": filename, "row": "", "status": "failed", "reason": "No usable polygon prompt exists."})
            continue

        predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        marker_boxes = detect_markers(image)
        exclusion = marker_exclusion(np.zeros((h, w), np.uint8), marker_boxes) == 0
        crop_issue = "cropped before elbow" in record.get("issue", "") or "polygon touches image boundary" in record.get("issue", "")
        for row_number, points in annotations:
            seed = rasterize(points, w, h)
            box, coords, labels = prompts(seed, points, w, h)
            try:
                with torch.inference_mode():
                    masks, scores, _ = predictor.predict(
                        box=box,
                        point_coords=coords,
                        point_labels=labels,
                        multimask_output=True,
                    )
            except Exception as exc:  # retain diagnostics and continue other rows
                audit_rows.append({"split": split, "filename": filename, "row": str(row_number), "status": "failed", "reason": f"SAM2 inference failed: {type(exc).__name__}: {exc}"})
                continue
            if masks.ndim == 2:
                masks = masks[None, ...]
            candidates = []
            seed_binary = seed > 0
            for idx, raw_mask in enumerate(masks):
                raw_mask = (raw_mask > 0).astype(np.uint8) * 255
                if raw_mask.shape != (h, w):
                    raw_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                mask, raw_component_count = largest_filled_component(raw_mask)
                candidate = mask > 0
                valid = ~exclusion
                inter = int(np.count_nonzero(candidate & seed_binary & valid))
                union = int(np.count_nonzero((candidate | seed_binary) & valid))
                overlap_iou = inter / max(1, union)
                score = float(scores[idx]) if idx < len(scores) else 0.0
                rank = 0.8 * overlap_iou + 0.2 * score
                candidates.append((rank, idx, score, overlap_iou, mask, raw_component_count))
            _, best_idx, score, overlap_iou, chosen, raw_component_count = max(candidates, key=lambda x: x[0])
            chosen_bool = chosen > 0
            comp_count = components(chosen)
            marker_centers = [(x + bw // 2, y + bh // 2) for x, y, bw, bh in marker_boxes]
            marker_covered = sum(1 for x, y in marker_centers if 0 <= x < w and 0 <= y < h and chosen[y, x] > 0)
            area_ratio = float(np.count_nonzero(chosen_bool) / max(1, np.count_nonzero(seed_binary)))
            touches_border = bool(chosen_bool[0].any() or chosen_bool[-1].any() or chosen_bool[:, 0].any() or chosen_bool[:, -1].any())
            name = f"{split}_{Path(filename).stem}_r{row_number}"
            cv2.imwrite(str(args.output / "masks" / f"{name}.png"), chosen)
            label_text = to_yolo_polygon(chosen)
            label_path_out = args.output / "candidate_labels" / f"{name}.txt"
            if label_text:
                label_path_out.write_text(label_text + "\n", encoding="utf-8")

            # A three-panel, native-resolution comparison: original mask,
            # selected SAM proposal, and differences. Nothing is auto-applied.
            before = overlay(image, seed, (0, 185, 255))
            after = overlay(image, chosen, (255, 170, 0))
            diff = image.copy()
            same = (seed_binary == chosen_bool)
            diff[(seed_binary & ~chosen_bool)] = (20, 20, 240)
            diff[(chosen_bool & ~seed_binary)] = (240, 60, 220)
            diff = cv2.addWeighted(diff, 0.55, image, 0.45, 0)
            panel_w = min(480, w)
            panels = []
            for panel in (before, after, diff):
                resized = cv2.resize(panel, (panel_w, max(1, round(h * panel_w / w))))
                panels.append(resized)
            preview = np.concatenate(panels, axis=1)
            title = f"{split}/{filename} row {row_number} | SAM2 tiny score={score:.3f} IoU-outside-tag={overlap_iou:.3f} | class-agnostic proposal only"
            cv2.putText(preview, title[:110], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(preview, title[:110], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            preview_path = args.output / "previews" / f"{name}_before_after.jpg"
            cv2.imwrite(str(preview_path), preview, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

            issues = record.get("issue", "")
            is_marker_only = issues.strip() == "detected ArUco center outside mask"
            high_match = overlap_iou >= 0.985 and 0.85 <= area_ratio <= 1.25
            all_marker_centers_covered = bool(marker_centers) and marker_covered == len(marker_centers)
            auto_eligible = bool(
                is_marker_only and high_match and comp_count == 1 and
                all_marker_centers_covered and score >= 0.88 and label_text
            )
            # This records an isolated candidate approval only where the source
            # was previously reviewed and the sole flagged issue is a marker gap.
            # The polygon is still staged, never installed as active ground truth.
            decision = "AUTO_APPROVED_PROPOSAL_ONLY_PENDING_FINAL_AUDIT" if auto_eligible else "PROPOSAL_ONLY_REQUIRES_HUMAN_REVIEW"
            reasons = ["SAM2 remains class-agnostic; confidence gates also require high agreement with the prior reviewed mask outside the marker.", "Amodal skin under the marker is inferred only by filling the external outline; it remains a proposal until final dataset audit."]
            if comp_count != 1:
                reasons.append(f"Filled candidate has {comp_count} connected components.")
            if raw_component_count > 1:
                reasons.append(f"Raw SAM output had {raw_component_count} components; only its largest component was retained for this review candidate.")
            if crop_issue:
                reasons.append("Image is cropped; SAM2 cannot infer invisible anatomy.")
            if prompt_source != "staged_CVAT_export":
                reasons.append("Prompt came from the old active label; this does not recover a CVAT correction.")
            if not auto_eligible:
                reasons.append("At least one strict automatic-approval condition failed.")
            audit_rows.append({
                "split": split, "filename": filename, "row": str(row_number),
                "issue": record.get("issue", ""), "prompt_source": prompt_source,
                "selected_candidate": str(best_idx + 1), "sam_predicted_iou": f"{score:.6f}",
                "seed_candidate_iou_excluding_marker": f"{overlap_iou:.6f}",
                "candidate_area_ratio": f"{area_ratio:.6f}",
                "candidate_components_8conn": str(comp_count), "raw_sam_components_8conn": str(raw_component_count),
                "candidate_touches_image_border": str(touches_border).lower(),
                "marker_centers": str(len(marker_centers)), "candidate_marker_centers_covered": str(marker_covered),
                "decision": decision,
                "confidence": "not a class confidence; SAM2 predicted mask quality only",
                "reason": " ".join(reasons),
                "preview": str(preview_path.relative_to(args.output)).replace("\\", "/"),
                "mask": str((args.output / "masks" / f"{name}.png").relative_to(args.output)).replace("\\", "/"),
                "candidate_label": str(label_path_out.relative_to(args.output)).replace("\\", "/") if label_text else "",
            })
        print(f"[{image_i}/{total}] {split}/{filename}: {len(annotations)} polygon prompt(s)", flush=True)

    columns = sorted({key for row in audit_rows for key in row})
    with (args.output / "sam2_proposals.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(audit_rows)
    summary = {
        "model": "Meta SAM2.1 Hiera Tiny",
        "device": device,
        "checkpoint": str(args.checkpoint),
        "input_images": total,
        "polygon_proposals": len(audit_rows),
        "successful_proposals": sum(r.get("decision", "").startswith("PROPOSAL_ONLY") for r in audit_rows),
        "automatically_accepted_proposals_pending_final_audit": sum(r.get("decision", "").startswith("AUTO_APPROVED") for r in audit_rows),
        "active_labels_modified": False,
        "source_images_modified": False,
        "splits_modified": False,
        "note": "All masks are class-agnostic review proposals; no output was installed as ground truth.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rebuild_candidate_labels(args)


if __name__ == "__main__":
    main()
