"""Offline diagnostic for real-photo YOLO arm segmentation failures.

This script does not modify the checkpoint, production thresholds, annotations,
or source photos. Outputs are derived, local-only diagnostics and should remain
private because they contain the supplied photograph.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parent
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))


def _json_write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def _read_bgr(path: Path) -> tuple[bytes, np.ndarray]:
    data = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode {path}")
    return data, image


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def _overlay(image: np.ndarray, mask: np.ndarray, color=(20, 220, 50), alpha=0.46) -> np.ndarray:
    output = image.copy()
    foreground = mask > 0
    if np.any(foreground):
        tint = np.zeros_like(output)
        tint[:] = color
        output[foreground] = cv2.addWeighted(output[foreground], 1.0 - alpha, tint[foreground], alpha, 0)
    return output


def _component_stats(mask: np.ndarray) -> dict[str, Any]:
    binary = (mask > 0).astype(np.uint8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)]
    return {"count": len(areas), "areas_px": sorted(areas, reverse=True)}


def _contour_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = image.copy()
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (0, 230, 255), 2, cv2.LINE_AA)
    return output


def _component_visualization(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    output = np.zeros_like(image)
    palette = [(40, 220, 40), (255, 100, 40), (40, 180, 255), (220, 40, 220), (255, 220, 40)]
    for label in range(1, count):
        color = palette[(label - 1) % len(palette)]
        output[labels == label] = color
        x, y, w, h, area = stats[label]
        cv2.putText(output, f"C{label}: {area}px", (int(x), max(18, int(y) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return cv2.addWeighted(image, 0.42, output, 0.58, 0)


def _mask_data(result: Any) -> list[np.ndarray]:
    masks = getattr(result, "masks", None)
    data = getattr(masks, "data", None) if masks is not None else None
    if data is None:
        return []
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    array = np.asarray(data)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        return []
    return [(np.asarray(item) > 0).astype(np.uint8) * 255 for item in array]


def _mask_stats(mask: np.ndarray) -> dict[str, Any]:
    area = int(np.count_nonzero(mask))
    height, width = mask.shape[:2]
    ys, xs = np.where(mask > 0)
    bbox = None if not xs.size else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return {
        "dimensions_wh": [int(width), int(height)],
        "area_px": area,
        "coverage_percent": 100.0 * area / float(width * height),
        "bbox_xyxy": bbox,
        "connected_components": _component_stats(mask),
    }


def _draw_boxes(image: np.ndarray, result: Any) -> np.ndarray:
    output = image.copy()
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return output
    def cpu_np(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)
    xyxy = cpu_np(boxes.xyxy).reshape(-1, 4)
    classes = cpu_np(boxes.cls).reshape(-1)
    scores = cpu_np(boxes.conf).reshape(-1)
    for box, cls_id, score in zip(xyxy, classes, scores):
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 150, 255), 3)
        cv2.putText(output, f"class={int(cls_id)} conf={float(score):.4f}", (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 150, 255), 2, cv2.LINE_AA)
    return output


def _fit_panel(image: np.ndarray, width: int, height: int, title: str) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
    panel = np.full((height + 38, width, 3), 248, dtype=np.uint8)
    cv2.putText(panel, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 30, 40), 2, cv2.LINE_AA)
    y = 38 + (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return panel


def _write_raw_visuals(out_dir: Path, image: np.ndarray, result: Any, threshold: float) -> tuple[list[dict[str, Any]], np.ndarray]:
    height, width = image.shape[:2]
    stem = f"conf_{threshold:.3f}".replace(".", "p")
    directory = out_dir / stem
    directory.mkdir(parents=True, exist_ok=True)
    masks = _mask_data(result)
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        class_ids, scores, boxes_xyxy = [], [], []
    else:
        def cpu_np(value):
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value)
        class_ids = cpu_np(boxes.cls).reshape(-1).astype(int).tolist()
        scores = cpu_np(boxes.conf).reshape(-1).astype(float).tolist()
        boxes_xyxy = cpu_np(boxes.xyxy).reshape(-1, 4).astype(float).tolist()

    # Keep arrays on the original image grid for all metrics and diagnostic PNGs.
    normalized_masks: list[np.ndarray] = []
    for mask in masks:
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        normalized_masks.append(mask)

    rows = []
    combined = np.zeros((height, width), dtype=np.uint8)
    for index, (class_id, score, box) in enumerate(zip(class_ids, scores, boxes_xyxy)):
        mask = normalized_masks[index] if index < len(normalized_masks) else np.zeros((height, width), dtype=np.uint8)
        combined = cv2.bitwise_or(combined, mask)
        row = {
            "confidence_threshold": threshold,
            "instance_index": index,
            "class_id": class_id,
            "class_name": "upper_arm" if class_id == 0 else str(class_id),
            "confidence": score,
            "bbox_xyxy": [round(v, 4) for v in box],
            "mask": _mask_stats(mask),
            "overlap_with_ground_truth": None,
            "overlap_note": "No ground-truth mask exists for adult2.jpeg; IoU/Dice not computed.",
        }
        rows.append(row)
        _write_image(directory / f"raw_mask_instance_{index:02d}.png", mask)
    _write_image(directory / "original.png", image)
    _write_image(directory / "raw_boxes.png", _draw_boxes(image, result))
    _write_image(directory / "raw_mask.png", combined)
    _write_image(directory / "raw_mask_overlay.png", _overlay(image, combined))
    _write_image(directory / "raw_contours.png", _contour_overlay(image, combined))
    _write_image(directory / "raw_connected_components.png", _component_visualization(image, combined))
    return rows, combined


def _diagnose_training_subset(dataset_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Summarize the exact 40-image audited subset and render polygon overlays."""
    rows: list[dict[str, Any]] = []
    tiles: list[np.ndarray] = []
    for split in ("train", "val", "test"):
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        for label_path in sorted(label_dir.glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
            matches = list(image_dir.glob(label_path.stem + ".*"))
            if not matches:
                rows.append({"split": split, "image": label_path.stem, "error": "image_missing"})
                continue
            image_path = matches[0]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                rows.append({"split": split, "image": image_path.name, "error": "decode_failed"})
                continue
            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            class_ids, vertex_count = [], 0
            polygon_areas = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                values = [float(v) for v in line.split()]
                if len(values) < 7 or len(values[1:]) % 2:
                    continue
                class_id = int(values[0])
                pts = np.asarray(values[1:], dtype=np.float64).reshape(-1, 2)
                pts[:, 0] *= width
                pts[:, 1] *= height
                polygon = np.round(pts).astype(np.int32)
                instance = np.zeros_like(mask)
                cv2.fillPoly(instance, [polygon], 255)
                mask = cv2.bitwise_or(mask, instance)
                class_ids.append(class_id)
                vertex_count += len(pts)
                polygon_areas.append(int(np.count_nonzero(instance)))
            ys, xs = np.where(mask > 0)
            if xs.size:
                bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                points = np.column_stack((xs, ys)).astype(np.float32)
                _mean, eigenvectors, eigenvalues = cv2.PCACompute2(points, mean=None)
                axis = float(np.degrees(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])) % 180)
                bbox_wh = [bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1]
                coverage = float(xs.size / (width * height))
            else:
                bbox, bbox_wh, axis, coverage = None, None, None, 0.0
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            outside = cv2.bitwise_not(cv2.dilate(mask, np.ones((11, 11), np.uint8)))
            edges = cv2.Canny(gray, 80, 160)
            outside_pixels = max(1, int(np.count_nonzero(outside)))
            detected = None
            try:
                from app.pipeline.reference_detect import detect_reference_marker
                marker = detect_reference_marker(image, 50.0)
                if marker.get("detected") and marker.get("marker_corners") is not None:
                    corners = np.asarray(marker["marker_corners"], dtype=np.float64).reshape(4, 2)
                    detected = {
                        "id": marker.get("marker_id"),
                        "center_xy_norm": [float(corners[:, 0].mean() / width), float(corners[:, 1].mean() / height)],
                        "confidence_proxy": float(marker.get("confidence", 0.0)),
                    }
            except Exception:
                pass
            rows.append({
                "split": split,
                "image": image_path.name,
                "dimensions_wh": [width, height],
                "class_ids": class_ids,
                "polygon_count": len(class_ids),
                "total_polygon_vertices": vertex_count,
                "polygon_area_px": polygon_areas,
                "mask_area_px": int(xs.size),
                "mask_coverage_percent": 100.0 * coverage,
                "mask_bbox_xyxy": bbox,
                "mask_bbox_width_height": bbox_wh,
                "mask_bbox_relative_wh": None if bbox_wh is None else [bbox_wh[0] / width, bbox_wh[1] / height],
                "mask_pca_axis_degrees": axis,
                "image_mean_luma": float(gray.mean()),
                "mask_mean_luma": float(gray[mask > 0].mean()) if xs.size else None,
                "mask_mean_saturation": float(hsv[:, :, 1][mask > 0].mean()) if xs.size else None,
                "background_edge_density_percent": 100.0 * np.count_nonzero((edges > 0) & (outside > 0)) / outside_pixels,
                "aruco_marker_detected": detected is not None,
                "aruco_marker": detected,
            })
            overlay = _overlay(image, mask)
            cv2.drawContours(overlay, cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 255, 255), 1)
            tiles.append(_fit_panel(overlay, 250, 185, f"{split}: {image_path.name}"))

    cols = 4
    tile_w, tile_h = 250, 223
    rows_count = max(1, (len(tiles) + cols - 1) // cols)
    sheet = np.full((rows_count * tile_h, cols * tile_w, 3), 245, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        y, x = divmod(index, cols)
        sheet[y * tile_h:(y + 1) * tile_h, x * tile_w:(x + 1) * tile_w] = tile
    _write_image(output_dir / "training_subset_contact_sheet.jpg", sheet)
    with (output_dir / "training_subset_image_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid = [r for r in rows if "error" not in r]
    def describe(key: str) -> dict[str, float | None]:
        values = [float(r[key]) for r in valid if r.get(key) is not None]
        return {
            "count": len(values),
            "min": min(values) if values else None,
            "median": float(np.median(values)) if values else None,
            "max": max(values) if values else None,
        }
    summary = {
        "dataset_dir": str(dataset_dir),
        "counts_by_split": {s: sum(r.get("split") == s and "error" not in r for r in rows) for s in ("train", "val", "test")},
        "image_count": len(valid),
        "class_id_counts": {str(i): sum(i in r.get("class_ids", []) for r in valid) for i in range(3)},
        "metric_summary": {
            "mask_coverage_percent": describe("mask_coverage_percent"),
            "mask_bbox_relative_width": {
                "count": sum(r.get("mask_bbox_relative_wh") is not None for r in valid),
                "median": float(np.median([r["mask_bbox_relative_wh"][0] for r in valid if r.get("mask_bbox_relative_wh") is not None])) if any(r.get("mask_bbox_relative_wh") is not None for r in valid) else None,
            },
            "mask_bbox_relative_height": {
                "count": sum(r.get("mask_bbox_relative_wh") is not None for r in valid),
                "median": float(np.median([r["mask_bbox_relative_wh"][1] for r in valid if r.get("mask_bbox_relative_wh") is not None])) if any(r.get("mask_bbox_relative_wh") is not None for r in valid) else None,
            },
            "mask_pca_axis_degrees": describe("mask_pca_axis_degrees"),
            "background_edge_density_percent": describe("background_edge_density_percent"),
            "image_mean_luma": describe("image_mean_luma"),
            "mask_mean_luma": describe("mask_mean_luma"),
            "mask_mean_saturation": describe("mask_mean_saturation"),
        },
        "aruco_detection_count": sum(bool(r.get("aruco_marker_detected")) for r in valid),
        "note": "Color and edge statistics are descriptive proxies only. They are not ground truth or proof of domain coverage.",
    }
    _json_write(output_dir / "training_subset_summary.json", summary)
    return summary


def _evaluate_annotated_test_resolution(
    dataset_dir: Path,
    output_dir: Path,
    segment_module: Any,
    image_sizes: list[int],
) -> dict[str, Any]:
    """Compare candidate inference sizes against the existing five held-out labels."""
    rows: list[dict[str, Any]] = []
    for label_path in sorted((dataset_dir / "labels" / "test").glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
        matches = list((dataset_dir / "images" / "test").glob(label_path.stem + ".*"))
        if not matches:
            continue
        image_path = matches[0]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        gt = np.zeros((height, width), dtype=np.uint8)
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = [float(v) for v in line.split()]
            if len(parts) < 7 or int(parts[0]) != 0 or len(parts[1:]) % 2:
                continue
            points = np.asarray(parts[1:], dtype=np.float64).reshape(-1, 2)
            points[:, 0] *= width
            points[:, 1] *= height
            cv2.fillPoly(gt, [np.round(points).astype(np.int32)], 255)
        for image_size in image_sizes:
            started = time.perf_counter()
            result = segment_module.predict_yolo_image(
                image, imgsz=image_size, confidence_threshold=0.25, mask_threshold=0.5,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            pred = (result["mask"] > 0).astype(np.uint8)
            target = (gt > 0).astype(np.uint8)
            intersection = int(np.count_nonzero(pred & target))
            union = int(np.count_nonzero(pred | target))
            pred_area = int(pred.sum())
            gt_area = int(target.sum())
            iou = intersection / union if union else None
            dice = (2 * intersection / (pred_area + gt_area)) if pred_area + gt_area else None
            height0, width0 = image.shape[:2]
            color = np.zeros((height0, width0, 3), dtype=np.uint8)
            color[target > 0] = (0, 0, 255)  # ground truth red
            color[pred > 0] = (0, 220, 0)   # prediction green
            color[(target > 0) & (pred > 0)] = (0, 220, 220)  # overlap yellow
            preview = cv2.addWeighted(image, 0.48, color, 0.52, 0)
            cv2.putText(preview, f"test {image_path.name} imgsz={image_size} conf=0.25 IoU={iou if iou is not None else 'n/a'}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            preview_dir = output_dir / "annotated_test_resolution" / f"imgsz_{image_size}"
            _write_image(preview_dir / f"{image_path.stem}_ground_truth_vs_prediction.png", preview)
            rows.append({
                "split": "test",
                "image": image_path.name,
                "image_size": image_size,
                "confidence_threshold": 0.25,
                "prediction_detected": bool(result["detected"]),
                "prediction_count": len(result["instances"]),
                "prediction_scores": [item["confidence"] for item in result["instances"]],
                "prediction_mask_dimensions_wh": [width0, height0],
                "prediction_area_px": pred_area,
                "ground_truth_area_px": gt_area,
                "intersection_px": intersection,
                "union_px": union,
                "iou": iou,
                "dice": dice,
                "inference_ms": elapsed_ms,
            })
    with (output_dir / "annotated_test_resolution_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["image_size", "image"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()})
    aggregate = {}
    for image_size in image_sizes:
        subset = [r for r in rows if r["image_size"] == image_size and r["iou"] is not None]
        aggregate[str(image_size)] = {
            "image_count": len(subset),
            "detections": sum(bool(r["prediction_detected"]) for r in subset),
            "mean_image_iou": float(np.mean([r["iou"] for r in subset])) if subset else None,
            "mean_image_dice": float(np.mean([r["dice"] for r in subset])) if subset else None,
            "note": "Existing annotated synthetic held-out split only; not real-image validation.",
        }
    result = {
        "threshold": 0.25,
        "image_sizes": image_sizes,
        "by_image_size": aggregate,
        "image_count": len({r["image"] for r in rows}),
        "note": "These five quarantined-subset test labels are annotation references for a configuration comparison, not real-world or clinical evidence.",
    }
    _json_write(output_dir / "annotated_test_resolution_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Exact real photograph; read-only input")
    parser.add_argument("--output", type=Path, default=SERVICE_ROOT / "results" / "real_world_segmentation_failure")
    parser.add_argument("--weights", type=Path, default=SERVICE_ROOT / "app" / "models" / "arm_segmentation" / "best.pt")
    parser.add_argument("--dataset", type=Path, default=SERVICE_ROOT / "results" / "verified_quarantine_training" / "dataset")
    parser.add_argument("--confidences", default="0.25,0.15,0.10,0.05,0.01")
    parser.add_argument("--image-sizes", default="416,640", help="Compare production and training image sizes")
    args = parser.parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(output_dir / "ultralytics_settings"))

    from ultralytics import YOLO
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor
    from app.pipeline import segment as segment_module
    from app.pipeline.pose_localize import locate_from_marker
    from app.pipeline.reference_detect import detect_reference_marker
    from app.pipeline.run import _crop_size_from_marker

    source_bytes, image = _read_bgr(args.image)
    height, width = image.shape[:2]
    checkpoint_sha256 = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    checkpoint = YOLO(str(args.weights))
    if checkpoint.task != "segment" or checkpoint.names != {0: "upper_arm"}:
        raise ValueError(f"Unexpected checkpoint metadata task={checkpoint.task!r}, names={checkpoint.names!r}")

    image_sizes = [int(part) for part in args.image_sizes.split(",") if part.strip()]
    production_imgsz = image_sizes[0]
    configured = segment_module.configure_segmentation_model(
        "yolo", args.weights, confidence_threshold=0.25, mask_threshold=0.5,
        min_mask_coverage=0.005, max_mask_coverage=0.95, image_size=production_imgsz,
    )
    if not configured:
        raise RuntimeError("Production segmentation model configuration failed.")
    model = segment_module._YOLO_MODEL
    device = str(segment_module._YOLO_DEVICE)

    # A single warm-up is excluded from threshold timing.
    model.predict(source=image, imgsz=production_imgsz, conf=0.25, iou=0.7, classes=[0], retina_masks=True, device=device, predictor=SegmentationPredictor, verbose=False)

    ref = detect_reference_marker(image, 50.0)
    crop_context = {"available": False}
    midpoint = (width / 2.0, height / 2.0)
    crop_size = 400
    if ref.get("detected") and ref.get("marker_corners") is not None:
        pose = locate_from_marker(image, ref["marker_corners"])
        midpoint = tuple(float(v) for v in pose.get("midpoint_px", midpoint))
        crop_size = _crop_size_from_marker(image, ref["marker_corners"])
        crop_context = {
            "available": True,
            "source": "production marker-based fallback localization for post-processing reproduction only",
            "marker_id": ref.get("marker_id"),
            "marker_center_xy": np.asarray(ref["marker_corners"]).reshape(4, 2).mean(axis=0).tolist(),
            "midpoint_xy": list(midpoint),
            "crop_size_px": int(crop_size),
            "crop_origin_xy": [int(round(midpoint[0])) - crop_size // 2, int(round(midpoint[1])) - crop_size // 2],
        }

    shutil.copyfile(args.image, output_dir / f"original{args.image.suffix.lower()}")
    rows: list[dict[str, Any]] = []
    thresholds = [float(part) for part in args.confidences.split(",") if part.strip()]
    base_config = segment_module._YOLO_CONFIG
    threshold_summaries = []
    try:
        for image_size in image_sizes:
            image_size_dir = output_dir / f"imgsz_{image_size}"
            for threshold in thresholds:
                start = time.perf_counter()
                raw_result = model.predict(
                    source=image, imgsz=image_size, conf=threshold, iou=0.7, classes=[0],
                    retina_masks=True, device=device, predictor=SegmentationPredictor, verbose=False,
                )[0]
                raw_ms = (time.perf_counter() - start) * 1000.0
                instance_rows, raw_combined = _write_raw_visuals(image_size_dir, image, raw_result, threshold)
                rows.extend([{**row, "image_size": image_size} for row in instance_rows])

                start = time.perf_counter()
                adapter_result = segment_module.predict_yolo_image(
                    image, imgsz=image_size, confidence_threshold=threshold, mask_threshold=0.5,
                )
                adapter_ms = (time.perf_counter() - start) * 1000.0
                subdir = image_size_dir / f"conf_{threshold:.3f}".replace(".", "p")
                processed_mask = adapter_result["mask"]
                _write_image(subdir / "postprocessed_full_image_mask.png", processed_mask)
                _write_image(subdir / "postprocessed_full_image_overlay.png", adapter_result["visualization"])
                union = cv2.bitwise_or(raw_combined, processed_mask)
                intersection = cv2.bitwise_and(raw_combined, processed_mask)
                raw_processed_iou = float(np.count_nonzero(intersection) / np.count_nonzero(union)) if np.any(union) else None
                _write_image(subdir / "raw_vs_postprocessed.png", np.hstack((
                    _fit_panel(_overlay(image, raw_combined), 480, 640, "Raw Ultralytics result mask"),
                    _fit_panel(_overlay(image, processed_mask), 480, 640, "PoshanScan /segment mask after filters"),
                )))

                previous_config = segment_module._YOLO_CONFIG
                segment_module._YOLO_CONFIG = replace(base_config, confidence=threshold, image_size=image_size)
                start = time.perf_counter()
                crop_result = segment_module.segment_arm(image, midpoint, crop_size_px=crop_size)
                crop_ms = (time.perf_counter() - start) * 1000.0
                segment_module._YOLO_CONFIG = previous_config
                crop_mask = crop_result["mask"]
                _write_image(subdir / "postprocessed_crop_mask.png", crop_mask)
                _write_image(subdir / "postprocessed_crop_overlay.png", _overlay(crop_result["crop_bgr"], crop_mask))
                _write_image(subdir / "postprocessed_crop_contours.png", _contour_overlay(crop_result["crop_bgr"], crop_mask))
                center = crop_size // 2
                half = max(16, min(50, crop_size // 4))
                target = crop_mask[max(0, center - half):min(crop_size, center + half + 1), max(0, center - half):min(crop_size, center + half + 1)]
                predictor_args = getattr(getattr(model, "predictor", None), "args", None)
                summary = {
                    "confidence_threshold": threshold,
                    "raw_prediction_iou_threshold": 0.7,
                    "adapter_effective_iou_threshold": None if predictor_args is None else getattr(predictor_args, "iou", None),
                    "class_filter": [0],
                    "image_size": image_size,
                    "mask_threshold": 0.5,
                    "device": device,
                    "model_orig_shape_hw": list(getattr(raw_result, "orig_shape", [])),
                    "raw_ultralytics_inference_ms": raw_ms,
                    "poshanscan_segment_adapter_inference_ms": adapter_ms,
                    "segment_arm_full_crop_path_ms": crop_ms,
                    "prediction_count": len(instance_rows),
                    "class_ids": [r["class_id"] for r in instance_rows],
                    "confidence_scores": [r["confidence"] for r in instance_rows],
                    "raw_combined_mask": _mask_stats(raw_combined),
                    "raw_postprocessed_binary_mask_iou": raw_processed_iou,
                    "postprocessed_full_image_detected": bool(adapter_result["detected"]),
                    "postprocessed_full_image_instances": adapter_result["instances"],
                    "postprocessed_full_image_mask": _mask_stats(processed_mask),
                    "crop_postprocessing_method": crop_result["segmentation_method"],
                    "crop_origin_xy": list(crop_result["crop_origin_px"]),
                    "crop_dimensions_wh": [int(crop_size), int(crop_size)],
                    "crop_mask": _mask_stats(crop_mask),
                    "crop_target_center_has_foreground": bool(np.any(target)),
                    "crop_output_is_legacy_fallback_and_not_accepted_as_yolo": crop_result["segmentation_method"] == "legacy_fallback",
                    "adapter_predictor_args": None if predictor_args is None else {"imgsz": getattr(predictor_args, "imgsz", None), "conf": getattr(predictor_args, "conf", None), "iou": getattr(predictor_args, "iou", None), "retina_masks": getattr(predictor_args, "retina_masks", None), "classes": getattr(predictor_args, "classes", None)},
                }
                threshold_summaries.append(summary)
    finally:
        segment_module._YOLO_CONFIG = base_config

    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["image_size", "confidence_threshold", "instance_index", "class_id", "class_name", "confidence", "bbox_xyxy", "mask", "overlap_with_ground_truth", "overlap_note"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})

    training_summary = _diagnose_training_subset(args.dataset, output_dir)
    annotated_test_resolution = _evaluate_annotated_test_resolution(args.dataset, output_dir, segment_module, image_sizes)
    summary = {
        "source_image": {"filename": args.image.name, "sha256": hashlib.sha256(source_bytes).hexdigest(), "bytes": len(source_bytes), "dimensions_wh": [width, height], "decoded_color_order": "OpenCV BGR", "decoder": "cv2.imdecode(IMREAD_COLOR)"},
        "checkpoint": {"path": str(args.weights), "sha256": checkpoint_sha256, "task": checkpoint.task, "names": checkpoint.names, "ultralytics_version": getattr(checkpoint, "ckpt", {}).get("version") if isinstance(getattr(checkpoint, "ckpt", None), dict) else None, "train_args": getattr(checkpoint, "ckpt", {}).get("train_args") if isinstance(getattr(checkpoint, "ckpt", None), dict) else None},
        "runtime": {"opencv_version": cv2.__version__, "ultralytics_version": __import__("ultralytics").__version__, "device": device},
        "preprocessing": {"source": "original BGR NumPy array", "production_imgsz": production_imgsz, "tested_image_sizes": image_sizes, "ultralytics_letterbox": "Ultralytics predictor default, aspect-preserving letterbox to requested inference size", "retina_masks": True, "output_mask_dimensions": [width, height], "confidence_thresholds": thresholds},
        "marker_crop_context": crop_context,
        "threshold_results": threshold_summaries,
        "training_subset": training_summary,
        "annotated_test_resolution_comparison": annotated_test_resolution,
        "ground_truth": {"available_for_real_image": False, "real_image_mask_overlap": None, "reason": "No reviewed ground-truth segmentation mask exists for adult2.jpeg. IoU/Dice and overlap against anatomy cannot be calculated without one; no mask was invented."},
        "outputs": {"folder": str(output_dir), "training_contact_sheet": "training_subset_contact_sheet.jpg", "training_image_audit": "training_subset_image_audit.csv", "threshold_predictions": "predictions.csv"},
    }
    _json_write(output_dir / "diagnostic_summary.json", summary)
    print(json.dumps({
        "output_dir": str(output_dir),
        "source_sha256": summary["source_image"]["sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "task": checkpoint.task,
        "names": checkpoint.names,
        "runs": [{"imgsz": x["image_size"], "confidence": x["confidence_threshold"], "count": x["prediction_count"], "scores": x["confidence_scores"], "mask_area": x["raw_combined_mask"]["area_px"], "crop_method": x["crop_postprocessing_method"], "raw_ms": round(x["raw_ultralytics_inference_ms"], 2)} for x in threshold_summaries],
        "training_counts": training_summary["counts_by_split"],
        "annotated_test_resolution": annotated_test_resolution["by_image_size"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
