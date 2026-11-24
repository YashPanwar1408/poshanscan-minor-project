"""Diagnostic overlay for inspecting MUAC axis and width geometry."""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np


def draw_measurement_geometry_debug(
    image: np.ndarray,
    marker_corners: Optional[np.ndarray],
    pose: Optional[dict[str, Any]],
    crop_origin_px: tuple[int, int],
    mask: Optional[np.ndarray],
    measurement_center_px: Optional[tuple[float, float]],
    arm_axis_degrees: Optional[float],
    axis_source: str,
    width_px: float,
    width_diagnostics: Optional[list[dict[str, Any]]] = None,
    mask_axis_degrees: Optional[float] = None,
    mask_axis_ratio: Optional[float] = None,
    axis_agreement_degrees: Optional[float] = None,
    segmentation_status: str = "not_run",
    measurement_valid: bool = False,
    rejection_reason: Optional[str] = None,
) -> np.ndarray:
    """Draw marker, pose, mask, axis, perpendicular sample lines and widths.

    ``width_diagnostics`` coordinates are local to ``mask`` and are translated
    by ``crop_origin_px`` before drawing on the original-resolution image.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty numpy array")
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    if canvas.ndim != 3 or canvas.shape[2] != 3:
        raise ValueError("image must have one or three channels")
    height, width = canvas.shape[:2]
    ox, oy = int(crop_origin_px[0]), int(crop_origin_px[1])

    # Tint only the in-frame area covered by the segmentation crop.
    if mask is not None and isinstance(mask, np.ndarray) and mask.ndim == 2 and mask.size:
        x0, y0 = max(0, ox), max(0, oy)
        x1, y1 = min(width, ox + mask.shape[1]), min(height, oy + mask.shape[0])
        if x1 > x0 and y1 > y0:
            local = mask[y0 - oy : y1 - oy, x0 - ox : x1 - ox] > 0
            roi = canvas[y0:y1, x0:x1]
            tint = np.zeros_like(roi)
            tint[:] = (255, 80, 40)  # BGR blue overlay
            roi[local] = cv2.addWeighted(roi[local], 0.62, tint[local], 0.38, 0.0)

    marker_center: Optional[tuple[float, float]] = None
    if marker_corners is not None:
        points = np.asarray(marker_corners, dtype=np.float32).reshape(-1, 2)
        if points.shape[0] >= 4:
            polygon = np.round(points).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [polygon], True, (0, 165, 255), 3, cv2.LINE_AA)
            marker_center = (float(points[:, 0].mean()), float(points[:, 1].mean()))
            marker_pixel = tuple(np.round(marker_center).astype(int))
            cv2.drawMarker(canvas, marker_pixel, (0, 255, 255), cv2.MARKER_CROSS, 22, 2)

    pose = pose or {}
    shoulder = pose.get("shoulder_px")
    elbow = pose.get("elbow_px")
    if shoulder is not None and elbow is not None:
        shoulder_pixel = tuple(np.round(shoulder).astype(int))
        elbow_pixel = tuple(np.round(elbow).astype(int))
        cv2.circle(canvas, shoulder_pixel, 7, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, elbow_pixel, 7, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.line(canvas, shoulder_pixel, elbow_pixel, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, "shoulder", (shoulder_pixel[0] + 6, shoulder_pixel[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, "elbow", (elbow_pixel[0] + 6, elbow_pixel[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)

    if measurement_center_px is not None:
        center = np.array(
            [measurement_center_px[0] + ox, measurement_center_px[1] + oy],
            dtype=np.float64,
        )
        center_pixel = tuple(np.round(center).astype(int))
        cv2.circle(canvas, center_pixel, 7, (255, 255, 0), -1, cv2.LINE_AA)
        if arm_axis_degrees is not None and np.isfinite(arm_axis_degrees):
            radians = np.deg2rad(float(arm_axis_degrees))
            half_length = 0.35 * float(min(height, width))
            direction = np.array([np.cos(radians), np.sin(radians)])
            p1 = tuple(np.round(center - half_length * direction).astype(int))
            p2 = tuple(np.round(center + half_length * direction).astype(int))
            axis_color = (0, 255, 0) if axis_source == "pose_landmarks" else (0, 0, 255)
            cv2.arrowedLine(canvas, p1, p2, axis_color, 3, cv2.LINE_AA, tipLength=0.06)

    diagnostics = width_diagnostics or []
    sample_rows = [item for item in diagnostics if item.get("start_px") and item.get("end_px")]
    if sample_rows:
        # Keep both accepted and rejected examples visible without obscuring
        # the photograph. The sidecar JSON retains every candidate line.
        accepted_indexes = [index for index, item in enumerate(sample_rows) if item.get("valid")]
        rejected_indexes = [index for index, item in enumerate(sample_rows) if not item.get("valid")]

        def spread(indexes: list[int], limit: int) -> list[int]:
            if not indexes:
                return []
            chosen = np.linspace(0, len(indexes) - 1, min(limit, len(indexes))).round().astype(int)
            return [indexes[int(index)] for index in chosen]

        selected_indexes = sorted(set(spread(accepted_indexes, 7) + spread(rejected_indexes, 7)))
        for index in selected_indexes:
            item = sample_rows[index]
            start = (int(round(item["start_px"][0] + ox)), int(round(item["start_px"][1] + oy)))
            end = (int(round(item["end_px"][0] + ox)), int(round(item["end_px"][1] + oy)))
            color = (255, 255, 0) if item.get("valid") else (150, 150, 150)
            cv2.line(canvas, start, end, color, 2, cv2.LINE_AA)
            cv2.circle(canvas, start, 5, (255, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, end, 5, (255, 0, 255), -1, cv2.LINE_AA)
            midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
            cv2.putText(canvas, f"{float(item['width_px']):.0f}px", (midpoint[0] + 5, midpoint[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    panel_width = 500
    output = np.full((height, width + panel_width, 3), (28, 28, 28), dtype=np.uint8)
    output[:, :width] = canvas
    panel_x = width + 16
    panel_lines = [
        ("MUAC geometry diagnostics", (255, 255, 255)),
        (f"Marker center: {marker_center if marker_center else 'not detected'}", (0, 255, 255)),
        (f"Pose landmarks: {'available' if shoulder is not None and elbow is not None else 'unavailable'}", (0, 255, 0)),
        (f"Selected axis: {float(arm_axis_degrees):.1f} deg / {axis_source}" if arm_axis_degrees is not None else "Selected axis: unavailable", (0, 0, 255)),
        (f"Mask axis: {float(mask_axis_degrees):.1f} deg" if mask_axis_degrees is not None else "Mask axis: unavailable", (255, 255, 255)),
        (f"Mask shape ratio: {float(mask_axis_ratio):.2f}" if mask_axis_ratio is not None else "Mask shape ratio: unavailable", (255, 255, 255)),
        (f"Pose/mask difference: {float(axis_agreement_degrees):.1f} deg" if axis_agreement_degrees is not None else "Pose/mask difference: unavailable", (255, 255, 255)),
        (f"Segmentation: {segmentation_status}", (255, 255, 255)),
        (f"Final sampled width: {float(width_px):.2f} px", (255, 255, 255)),
        (f"Sampling lines logged: {len(sample_rows)}", (255, 255, 0)),
        (f"Measurement: {'VALID' if measurement_valid else 'INVALID'}", (0, 255, 0) if measurement_valid else (0, 165, 255)),
        (f"Reason: {rejection_reason or 'none'}", (220, 220, 220)),
        ("Legend", (200, 200, 200)),
        ("Green: pose landmarks", (0, 255, 0)),
        ("Red: selected arm axis", (0, 0, 255)),
        ("Cyan: perpendicular sample", (255, 255, 0)),
        ("Magenta: selected boundaries", (255, 0, 255)),
        ("Blue tint: segmented arm mask", (255, 80, 40)),
    ]
    line_height = 25
    panel_y = 30
    for index, (text, color) in enumerate(panel_lines):
        if text == "Legend":
            panel_y += 10
        cv2.putText(output, text, (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.44, color, 1, cv2.LINE_AA)
        panel_y += line_height
    return output
