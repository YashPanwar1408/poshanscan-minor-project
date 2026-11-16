"""Shared MUAC pipeline used by the HTTP API and the evaluation script."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from app.pipeline.classify import classify_risk_band, compute_confidence_score
from app.pipeline.estimate import CalibrationCorrector, estimate_muac
from app.pipeline.pose_localize import locate_arm_midpoint, locate_from_marker
from app.pipeline.reference_detect import detect_reference_marker
from app.pipeline.segment import (
    measure_width_perpendicular,
    remove_polygon_from_mask,
    segment_arm,
)

logger = logging.getLogger("poshanscan-cv")

MIN_FOREGROUND_PX = 100
MIN_WIDTH_PX = 5.0
MUAC_RANGES_MM = {
    "child_6_59m": (70.0, 180.0),
    "adult": (100.0, 450.0),
}
_calibrator = CalibrationCorrector()


def locate_either_arm(image: np.ndarray) -> dict[str, Any]:
    left = locate_arm_midpoint(image, side="left")
    if left.get("detected"):
        return left
    return locate_arm_midpoint(image, side="right")


def _crop_size_from_marker(image: np.ndarray, marker_corners: Optional[np.ndarray]) -> int:
    height, width = image.shape[:2]
    cap = int(min(height, width))
    if marker_corners is None:
        return int(np.clip(400, 240, cap))
    pts = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
    sides = [
        float(np.linalg.norm(pts[1] - pts[0])),
        float(np.linalg.norm(pts[2] - pts[1])),
        float(np.linalg.norm(pts[3] - pts[2])),
        float(np.linalg.norm(pts[0] - pts[3])),
    ]
    mean_side = float(np.mean(sides))
    # The marker is already placed at the MUAC site. A large crop causes the
    # skin baseline to include the torso and background when pose detection
    # misses, which produces impossible widths.
    return int(np.clip(mean_side * 2.5, 280, cap))


def _segmentation_ok(mask: np.ndarray, width_px: float) -> bool:
    if mask is None or mask.size == 0:
        return False
    return int(np.count_nonzero(mask)) >= MIN_FOREGROUND_PX and width_px >= MIN_WIDTH_PX


def _segmentation_quality(mask: np.ndarray) -> float:
    if mask is None or mask.size == 0:
        return 0.0
    return float(np.clip(np.count_nonzero(mask) / float(mask.size), 0.0, 1.0))


def run_muac_pipeline(
    image: np.ndarray,
    reference_size_mm: float,
    age_category: str = "child_6_59m",
) -> dict[str, Any]:
    """Run reference → pose (or marker fallback) → segment → estimate → classify.

    Parameters
    ----------
    image:
        BGR image as a numpy array.
    reference_size_mm:
        Physical size of the ArUco marker edge in millimetres.
    age_category:
        Must be ``"child_6_59m"`` or ``"adult"``.  Passed directly to
        :func:`classify_risk_band`; a ``ValueError`` from that function
        propagates up to the caller.

    Returns a dict with ``ok`` True on success, or ``ok`` False and ``error`` set.
    """
    if age_category not in MUAC_RANGES_MM:
        raise ValueError("age_category must be 'child_6_59m' or 'adult'")

    reference = detect_reference_marker(image, float(reference_size_mm))
    logger.info(
        "reference_detect: detected=%s scale_px_per_mm=%s conf=%.3f marker_id=%s",
        reference["detected"],
        reference.get("scale_px_per_mm"),
        reference.get("confidence") or 0.0,
        reference.get("marker_id"),
    )
    if not reference["detected"] or not reference.get("scale_px_per_mm"):
        return {
            "ok": False,
            "error": "reference_object_not_detected",
            "message": "No ArUco reference marker was found in the image.",
        }

    pose = locate_either_arm(image)
    logger.info(
        "pose_localize: detected=%s midpoint=%s angle=%s conf=%.3f",
        pose.get("detected"),
        pose.get("midpoint_px"),
        pose.get("arm_angle_degrees"),
        pose.get("confidence") or 0.0,
    )
    if not pose.get("detected") or pose.get("midpoint_px") is None:
        pose = locate_from_marker(image, reference.get("marker_corners"))
        logger.info(
            "pose_localize: MediaPipe missed; marker fallback detected=%s midpoint=%s angle=%s",
            pose.get("detected"),
            pose.get("midpoint_px"),
            pose.get("arm_angle_degrees"),
        )
    if not pose.get("detected") or pose.get("midpoint_px") is None:
        return {
            "ok": False,
            "error": "arm_not_detected",
            "message": "Could not localise a shoulder–elbow pair for MUAC measurement.",
        }

    corners = reference.get("marker_corners")
    crop_size = _crop_size_from_marker(image, corners)
    segmented = segment_arm(image, pose["midpoint_px"], crop_size_px=crop_size)
    mask = segmented["mask"]
    origin = segmented["crop_origin_px"]
    marker_side_px = 0.0
    marker_points: Optional[np.ndarray] = None
    if corners is not None:
        mask = remove_polygon_from_mask(mask, corners, origin)
        marker_points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        marker_side_px = float(
            np.mean(
                [
                    np.linalg.norm(marker_points[(i + 1) % 4] - marker_points[i])
                    for i in range(4)
                ]
            )
        )

    centroid = segmented.get("centroid_px")
    if pose.get("shoulder_px") is None and corners is not None:
        assert marker_points is not None
        marker_center = marker_points.mean(axis=0)
        centroid = (
            float(marker_center[0] - origin[0]),
            float(marker_center[1] - origin[1]),
        )
    if centroid is None:
        h, w = mask.shape[:2]
        centroid = (w / 2.0, h / 2.0)
    angle = float(pose.get("arm_angle_degrees") if pose.get("arm_angle_degrees") is not None else 90.0)
    width_px = measure_width_perpendicular(
        mask,
        centroid,
        angle,
        bridge_internal_gaps=corners is not None,
        search_radius_px=int(marker_side_px / 2 + 25) if corners is not None else 0,
        min_width_px=marker_side_px * 0.2 if corners is not None else 0.0,
        # The final physical MUAC range is the safety bound. A marker-pixel
        # ratio is unreliable when the upload has been resized by a camera or
        # browser before reaching the service.
        max_width_px=None,
    )
    if age_category == "child_6_59m" and marker_side_px > 0:
        # A child upper arm cannot be wider than the 50 mm reference marker
        # used at the same site. The outer-span fallback can include the torso
        # when the marker hides the complete arm cross-section.
        width_px = min(width_px, marker_side_px * 0.85)
    seg_ok = _segmentation_ok(mask, width_px)
    seg_quality = _segmentation_quality(mask)
    logger.info(
        "segment: crop=%s centroid=%s width_px=%.2f foreground=%d ok=%s quality=%.3f",
        crop_size,
        centroid,
        width_px,
        int(np.count_nonzero(mask)),
        seg_ok,
        seg_quality,
    )

    estimated = estimate_muac(width_px, float(reference["scale_px_per_mm"]))
    baseline_mm = estimated.get("muac_estimate_mm")
    if baseline_mm is None:
        baseline_mm = 0.0
    muac_mm = _calibrator.predict(float(baseline_mm))
    min_muac, max_muac = MUAC_RANGES_MM[age_category]
    if not min_muac <= muac_mm <= max_muac:
        logger.warning(
            "estimate rejected: muac_mm=%.3f outside %s range %.1f-%.1f",
            muac_mm,
            age_category,
            min_muac,
            max_muac,
        )
        if age_category == "child_6_59m" and muac_mm > max_muac:
            message = (
                "This photo appears to be outside the child range. "
                "If this is an adult, return to the list and choose Adults "
                "before screening."
            )
        else:
            message = (
                f"The measured MUAC is outside the valid {age_category} range. "
                "Retake the photo with the full arm and marker visible."
            )
        return {
            "ok": False,
            "error": "measurement_out_of_range",
            "message": message,
        }
    classified = classify_risk_band(muac_mm, age_category)
    confidence = compute_confidence_score(
        float(reference.get("confidence") or 0.0),
        float(pose.get("confidence") or 0.0),
        seg_quality,
    )
    logger.info(
        "estimate/classify: width_mm=%s muac_mm=%.3f band=%s conf=%.3f",
        estimated.get("width_mm"),
        muac_mm,
        classified["risk_band"],
        confidence,
    )
    return {
        "ok": True,
        "error": None,
        "message": "",
        "muac_estimate_mm": float(muac_mm),
        "risk_band": classified["risk_band"],
        "age_category": classified["age_category"],
        "confidence_score": confidence,
        "segmentation_ok": seg_ok,
        "arm_detected": True,
        "reference_detected": True,
    }
