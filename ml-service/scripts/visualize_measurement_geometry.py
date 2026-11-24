"""Create an original-resolution diagnostic overlay for a MUAC input image."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))

from app.pipeline.geometry_debug import draw_measurement_geometry_debug  # noqa: E402
from app.pipeline.pose_localize import estimate_arm_axis_from_mask, locate_from_marker  # noqa: E402
from app.pipeline.pose_localize import axis_angle_difference_degrees  # noqa: E402
from app.pipeline.reference_detect import detect_reference_marker  # noqa: E402
from app.pipeline.run import _crop_size_from_marker, locate_either_arm  # noqa: E402
from app.pipeline.segment import (  # noqa: E402
    configure_segmentation_model,
    measure_width_perpendicular,
    remove_polygon_from_mask,
    segment_arm,
)


def _json_safe_pose(pose: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pose.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, tuple):
            result[key] = list(value)
        elif isinstance(value, (np.floating, np.integer)):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Original image sent to /infer")
    parser.add_argument("--output", type=Path, help="Overlay image path")
    parser.add_argument("--reference-size-mm", type=float, default=50.0)
    parser.add_argument("--age-category", choices=("child_6_59m", "adult"), default="child_6_59m")
    parser.add_argument("--weights", type=Path, help="Optional YOLO segmentation checkpoint path")
    parser.add_argument("--legacy-segmentation", action="store_true", help="Use HSV/GrabCut instead of YOLO")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    image_path = args.image.resolve()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"Could not decode image: {image_path}")

    mode = "legacy" if args.legacy_segmentation else "yolo"
    configure_segmentation_model(mode, str(args.weights) if args.weights else None)

    reference = detect_reference_marker(image, args.reference_size_mm)
    if not reference["detected"]:
        parser.error("No ArUco marker was detected; cannot reproduce the reference scale or marker center.")
    pose = locate_either_arm(image)
    pose_source = "mediapipe" if pose.get("detected") and pose.get("midpoint_px") is not None else "marker_fallback"
    if pose_source == "marker_fallback":
        pose = locate_from_marker(image, reference.get("marker_corners"))
    if not pose.get("detected") or pose.get("midpoint_px") is None:
        parser.error("Neither pose nor marker fallback produced a measurement center.")

    crop_size = _crop_size_from_marker(image, reference.get("marker_corners"))
    segmented = segment_arm(image, pose["midpoint_px"], crop_size_px=crop_size)
    mask = segmented["mask"]
    origin = segmented["crop_origin_px"]
    corners = reference.get("marker_corners")
    marker_side_px = 0.0
    if corners is not None:
        points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        marker_side_px = float(np.mean([
            np.linalg.norm(points[(index + 1) % 4] - points[index]) for index in range(4)
        ]))
        mask = remove_polygon_from_mask(mask, corners, origin)

    center = (
        float(pose["midpoint_px"][0] - origin[0]),
        float(pose["midpoint_px"][1] - origin[1]),
    )
    mask_axis = estimate_arm_axis_from_mask(mask, center)
    mask_angle = mask_axis.get("angle_degrees")
    pose_angle = pose.get("arm_angle_degrees")
    axis_source = "unavailable"
    angle = None
    if pose_source == "marker_fallback":
        if mask_angle is not None:
            angle, axis_source = float(mask_angle), "segmentation_mask"
    elif pose_angle is not None and np.isfinite(float(pose_angle)):
        if mask_angle is None or axis_angle_difference_degrees(float(pose_angle), float(mask_angle)) <= 45:
            angle, axis_source = float(pose_angle), "pose_landmarks"
    elif mask_angle is not None:
        angle, axis_source = float(mask_angle), "segmentation_mask"

    width_diagnostics: list[dict[str, Any]] = []
    width_px = 0.0
    if angle is not None:
        width_px = measure_width_perpendicular(
            mask,
            center,
            angle,
            bridge_internal_gaps=corners is not None,
            search_radius_px=int(marker_side_px / 2 + 25) if corners is not None else 0,
            min_width_px=marker_side_px * 0.2 if corners is not None else 0.0,
            diagnostics=width_diagnostics,
            max_bridge_gap_px=marker_side_px * 1.25 if marker_side_px > 0 else None,
        )
        if args.age_category == "child_6_59m" and marker_side_px > 0:
            visible_width_px = measure_width_perpendicular(
                mask,
                center,
                angle,
                bridge_internal_gaps=False,
                search_radius_px=int(marker_side_px / 2 + 25),
                min_width_px=marker_side_px * 0.2,
            )
            if visible_width_px > 0:
                width_px = min(width_px, visible_width_px * 2.0)
    else:
        width_diagnostics.append({"rejected": True, "reason": "no_reliable_axis"})

    overlay = draw_measurement_geometry_debug(
        image=image,
        marker_corners=corners,
        pose=pose,
        crop_origin_px=origin,
        mask=mask,
        measurement_center_px=center,
        arm_axis_degrees=angle,
        axis_source=axis_source,
        width_px=width_px,
        width_diagnostics=width_diagnostics,
        mask_axis_degrees=float(mask_angle) if mask_angle is not None else None,
        mask_axis_ratio=float(mask_axis.get("eigenvalue_ratio") or 0.0),
    )

    output = args.output.resolve() if args.output else image_path.with_name(f"{image_path.stem}_geometry_debug.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), overlay):
        raise RuntimeError(f"Could not write overlay: {output}")
    json_path = output.with_suffix(".json")
    report = {
        "input_image": str(image_path),
        "image_dimensions_wh": [int(image.shape[1]), int(image.shape[0])],
        "marker_detected": bool(reference["detected"]),
        "marker_id": reference.get("marker_id"),
        "marker_corners_xy": np.asarray(corners).reshape(-1, 2).tolist() if corners is not None else None,
        "marker_center_px": np.asarray(corners).reshape(-1, 2).mean(axis=0).tolist() if corners is not None else None,
        "scale_px_per_mm": reference.get("scale_px_per_mm"),
        "pose_source": pose_source,
        "pose_result": _json_safe_pose(pose),
        "crop_origin_xy": list(origin),
        "measurement_center_crop_px": list(center),
        "segmentation_method": segmented.get("segmentation_method"),
        "segmentation_confidence": segmented.get("segmentation_confidence"),
        "segmentation_foreground_px": int(np.count_nonzero(mask)),
        "mask_axis": mask_axis,
        "selected_axis_degrees": angle,
        "selected_axis_source": axis_source,
        "sampled_width_px": float(width_px),
        "width_sample_diagnostics": width_diagnostics,
        "overlay_image": str(output),
    }
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Overlay: {output}")
    print(f"Diagnostics: {json_path}")
    print(f"pose={pose_source}; axis={angle} ({axis_source}); width_px={width_px:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
