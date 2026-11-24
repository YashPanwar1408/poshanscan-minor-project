"""Shared MUAC pipeline used by the HTTP API and the evaluation script."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from app.pipeline.classify import classify_risk_band, compute_confidence_score
from app.pipeline.estimate import CalibrationCorrector, estimate_muac
from app.pipeline.pose_localize import (
    axis_angle_difference_degrees,
    estimate_arm_axis_from_mask,
    locate_arm_midpoint,
    locate_from_marker,
)
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


def set_calibrator(calibrator: Optional[CalibrationCorrector]) -> None:
    """Replace the inference corrector; absent models use the pass-through."""
    global _calibrator
    if calibrator is None or not calibrator.is_fitted or calibrator.model is None:
        _calibrator = CalibrationCorrector()
        return
    _calibrator = calibrator


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
    diagnostics: Optional[dict[str, Any]] = None,
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

    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({
            "image_dimensions_wh": [int(image.shape[1]), int(image.shape[0])]
            if isinstance(image, np.ndarray) and image.ndim >= 2
            else None,
            "marker_detected": False,
            "segmentation_status": "not_run",
            "pose_available": False,
            "pose_source": None,
            "axis_method": None,
            "axis_degrees": None,
            "axis_confidence": None,
            "axis_agreement_degrees": None,
            "accepted_samples": 0,
            "rejected_samples": 0,
            "measurement_valid": False,
            "estimated_muac_mm": None,
            "rejection_reason": None,
        })

    stage_status = {
        "reference": "not_run",
        "pose": "not_run",
        "segmentation": "not_run",
        "muac_estimation": "not_run",
    }

    reference = detect_reference_marker(image, float(reference_size_mm))
    if diagnostics is not None:
        corners = reference.get("marker_corners")
        corner_array = np.asarray(corners).reshape(-1, 2) if corners is not None else None
        diagnostics.update({
            "marker_detected": bool(reference.get("detected")),
            "marker_id": reference.get("marker_id"),
            "marker_corners_xy": corner_array.tolist() if corner_array is not None else None,
            "marker_center_px": corner_array.mean(axis=0).tolist() if corner_array is not None else None,
            "marker_scale_px_per_mm": reference.get("scale_px_per_mm"),
            "marker_confidence": reference.get("confidence"),
        })
    stage_status["reference"] = "detected" if reference["detected"] else "not_detected"
    logger.info(
        "reference_detect: detected=%s scale_px_per_mm=%s conf=%.3f marker_id=%s",
        reference["detected"],
        reference.get("scale_px_per_mm"),
        reference.get("confidence") or 0.0,
        reference.get("marker_id"),
    )
    if not reference["detected"] or not reference.get("scale_px_per_mm"):
        if diagnostics is not None:
            diagnostics["rejection_reason"] = "reference_object_not_detected"
        return {
            "ok": False,
            "error": "reference_object_not_detected",
            "message": "No ArUco reference marker was found in the image.",
            "stage_status": stage_status,
        }

    pose = locate_either_arm(image)
    pose_source = "mediapipe"
    logger.info(
        "pose_localize: detected=%s midpoint=%s angle=%s conf=%.3f",
        pose.get("detected"),
        pose.get("midpoint_px"),
        pose.get("arm_angle_degrees"),
        pose.get("confidence") or 0.0,
    )
    if not pose.get("detected") or pose.get("midpoint_px") is None:
        pose_source = "marker_fallback"
        pose = locate_from_marker(image, reference.get("marker_corners"))
        logger.info(
            "pose_localize: MediaPipe missed; marker fallback detected=%s midpoint=%s angle=%s",
            pose.get("detected"),
            pose.get("midpoint_px"),
            pose.get("arm_angle_degrees"),
        )
    stage_status["pose"] = (
        pose_source
        if pose.get("detected") and pose.get("midpoint_px") is not None
        else "not_detected"
    )
    if not pose.get("detected") or pose.get("midpoint_px") is None:
        if diagnostics is not None:
            diagnostics.update({
                "pose_available": False,
                "pose_source": "marker_fallback",
                "pose": pose,
                "rejection_reason": "arm_not_detected",
            })
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
    if diagnostics is not None:
        diagnostics.update({
            "pose_available": pose_source == "mediapipe",
            "pose_source": pose_source,
            "pose": pose,
            "segmentation_method": segmented.get("segmentation_method"),
            "segmentation_confidence": segmented.get("segmentation_confidence"),
            "segmentation_mask": mask.copy() if isinstance(mask, np.ndarray) else None,
            "crop_origin_px": tuple(int(v) for v in origin),
            "segmentation_foreground_px": int(np.count_nonzero(mask)) if isinstance(mask, np.ndarray) else 0,
        })
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

    # segment_arm crops around the pose midpoint (or, in marker fallback, the
    # ArUco center). Measure at that anatomical site rather than at the mask's
    # overall centroid, which can shift toward a broad shoulder or stray pixels.
    measurement_center = (
        float(pose["midpoint_px"][0] - origin[0]),
        float(pose["midpoint_px"][1] - origin[1]),
    )
    mask_axis = estimate_arm_axis_from_mask(mask, measurement_center)
    mask_angle = mask_axis.get("angle_degrees")
    pose_angle: Optional[float]
    try:
        candidate_pose_angle = pose.get("arm_angle_degrees")
        pose_angle = float(candidate_pose_angle) if candidate_pose_angle is not None else None
        if pose_angle is not None and not np.isfinite(pose_angle):
            pose_angle = None
    except (TypeError, ValueError, OverflowError):
        pose_angle = None
    axis_disagreement = (
        axis_angle_difference_degrees(pose_angle, float(mask_angle))
        if pose_angle is not None and mask_angle is not None
        else None
    )

    if pose_source == "marker_fallback":
        if mask_angle is None:
            stage_status["segmentation"] = "axis_unreliable"
            if diagnostics is not None:
                diagnostics.update({
                    "segmentation_status": "axis_unreliable",
                    "measurement_mask": mask.copy(),
                    "measurement_center_crop_px": measurement_center,
                    "mask_axis_degrees": None,
                    "mask_axis_eigenvalue_ratio": float(mask_axis.get("eigenvalue_ratio") or 0.0),
                    "axis_confidence": None,
                    "rejection_reason": "arm_axis_unreliable",
                })
            return {
                "ok": False,
                "error": "arm_axis_unreliable",
                "message": (
                    "Could not estimate a reliable local arm axis from the segmented mask. "
                    "Retake the image with the upper arm clearly visible."
                ),
                "segmentation_ok": False,
                "stage_status": stage_status,
            }
        angle = float(mask_angle)
        axis_source = "segmentation_mask"
    elif pose_angle is not None:
        if mask_angle is not None:
            if axis_disagreement > 45.0:
                stage_status["segmentation"] = "axis_conflict"
                if diagnostics is not None:
                    diagnostics.update({
                        "segmentation_status": "axis_conflict",
                        "measurement_mask": mask.copy(),
                        "measurement_center_crop_px": measurement_center,
                        "pose_axis_degrees": pose_angle,
                        "mask_axis_degrees": float(mask_angle),
                        "mask_axis_eigenvalue_ratio": float(mask_axis.get("eigenvalue_ratio") or 0.0),
                        "axis_method": "pose_mask_conflict",
                        "axis_degrees": pose_angle,
                        "axis_confidence": float(np.clip(pose.get("confidence") or 0.0, 0.0, 1.0)),
                        "axis_agreement_degrees": axis_disagreement,
                        "rejection_reason": "arm_axis_unreliable",
                    })
                return {
                    "ok": False,
                    "error": "arm_axis_unreliable",
                    "message": (
                        "Pose landmarks and the segmented arm disagree on the local arm direction. "
                        "Retake the image with the whole upper arm visible."
                    ),
                    "segmentation_ok": False,
                    "stage_status": stage_status,
                }
        angle = pose_angle
        axis_source = "pose_landmarks"
    elif mask_angle is not None:
        angle = float(mask_angle)
        axis_source = "segmentation_mask"
    else:
        stage_status["segmentation"] = "axis_unreliable"
        if diagnostics is not None:
            diagnostics.update({
                "segmentation_status": "axis_unreliable",
                "measurement_mask": mask.copy(),
                "measurement_center_crop_px": measurement_center,
                "mask_axis_degrees": None,
                "mask_axis_eigenvalue_ratio": float(mask_axis.get("eigenvalue_ratio") or 0.0),
                "rejection_reason": "arm_axis_unreliable",
            })
        return {
            "ok": False,
            "error": "arm_axis_unreliable",
            "message": "Could not estimate a reliable local arm axis from pose or segmentation.",
            "segmentation_ok": False,
            "stage_status": stage_status,
        }

    axis_confidence = (
        float(np.clip(pose.get("confidence") or 0.0, 0.0, 1.0))
        if axis_source == "pose_landmarks"
        else float(np.clip(1.0 - 1.0 / max(float(mask_axis.get("eigenvalue_ratio") or 0.0), 1.0), 0.0, 1.0))
    )
    width_diagnostics: list[dict[str, Any]] = []
    if diagnostics is not None:
        diagnostics.update({
            "measurement_mask": mask.copy(),
            "measurement_center_crop_px": measurement_center,
            "pose_axis_degrees": pose_angle,
            "mask_axis_degrees": float(mask_angle) if mask_angle is not None else None,
            "mask_axis_eigenvalue_ratio": float(mask_axis.get("eigenvalue_ratio") or 0.0),
            "axis_method": axis_source,
            "axis_degrees": float(angle),
            "axis_confidence": axis_confidence,
            "axis_agreement_degrees": axis_disagreement,
        })
    width_px = measure_width_perpendicular(
        mask,
        measurement_center,
        angle,
        bridge_internal_gaps=corners is not None,
        search_radius_px=int(marker_side_px / 2 + 25) if corners is not None else 0,
        min_width_px=marker_side_px * 0.2 if corners is not None else 0.0,
        diagnostics=width_diagnostics,
        max_bridge_gap_px=marker_side_px * 1.25 if marker_side_px > 0 else None,
        # The final physical MUAC range is the safety bound. A marker-pixel
        # ratio is unreliable when the upload has been resized by a camera or
        # browser before reaching the service.
        max_width_px=None,
    )
    visible_width_diagnostics: list[dict[str, Any]] = []
    if age_category == "child_6_59m" and marker_side_px > 0:
        # When the marker hides the cross-section, the outer span can include
        # nearby torso pixels. Reconstruct the arm from the visible side runs
        # instead of forcing every child image to the same marker-based value.
        visible_width_px = measure_width_perpendicular(
            mask,
            measurement_center,
            angle,
            bridge_internal_gaps=False,
            search_radius_px=int(marker_side_px / 2 + 25),
            min_width_px=marker_side_px * 0.2,
            diagnostics=visible_width_diagnostics,
            max_width_px=None,
        )
        if visible_width_px > 0:
            width_px = min(width_px, visible_width_px * 2.0)
    seg_ok = _segmentation_ok(mask, width_px)
    stage_status["segmentation"] = "ok" if seg_ok else "low_quality"
    if diagnostics is not None:
        diagnostics.update({
            "segmentation_status": "ok" if seg_ok else "low_quality",
            "width_samples": width_diagnostics,
            "visible_width_samples": visible_width_diagnostics,
            "accepted_samples": sum(bool(item.get("valid")) for item in width_diagnostics),
            "rejected_samples": sum(not bool(item.get("valid")) for item in width_diagnostics),
            "sampled_width_px": float(width_px),
        })
    seg_quality = _segmentation_quality(mask)
    segmentation_confidence = float(segmented.get("segmentation_confidence") or 0.0)
    segmentation_method = str(segmented.get("segmentation_method") or "legacy")
    if segmentation_method == "yolo":
        seg_quality = min(seg_quality, segmentation_confidence)
    logger.info(
        "segment: method=%s crop=%s center=%s axis=%.2f axis_source=%s mask_axis=%s mask_axis_ratio=%.3f width_px=%.2f foreground=%d ok=%s quality=%.3f model_conf=%.3f",
        segmentation_method,
        crop_size,
        measurement_center,
        angle,
        axis_source,
        mask_angle,
        float(mask_axis.get("eigenvalue_ratio") or 0.0),
        width_px,
        int(np.count_nonzero(mask)),
        seg_ok,
        seg_quality,
        segmentation_confidence,
    )

    estimated = estimate_muac(width_px, float(reference["scale_px_per_mm"]))
    raw_baseline_mm = estimated.get("muac_estimate_mm")
    try:
        baseline_mm = 0.0 if raw_baseline_mm is None else float(raw_baseline_mm)
    except (TypeError, ValueError, OverflowError):
        baseline_mm = float("nan")
    # Do not let a learned calibration map turn a malformed, zero, or negative
    # geometric estimate into a seemingly valid positive measurement.
    if not np.isfinite(baseline_mm) or baseline_mm <= 0:
        muac_mm = baseline_mm
    else:
        muac_mm = _calibrator.predict(baseline_mm)
    min_muac, max_muac = MUAC_RANGES_MM[age_category]
    try:
        muac_mm = float(muac_mm)
    except (TypeError, ValueError, OverflowError):
        muac_mm = float("nan")

    measurement_failure: Optional[str] = None
    if not np.isfinite(muac_mm):
        measurement_failure = "non_finite"
        stage_status["muac_estimation"] = "invalid"
        message = (
            "The image did not produce a finite MUAC measurement. "
            "Retake the photo with the arm and reference marker clearly visible."
        )
    elif muac_mm <= 0:
        measurement_failure = "non_positive"
        stage_status["muac_estimation"] = "invalid"
        message = (
            "The image did not produce a positive MUAC measurement. "
            "Retake the photo with the arm and reference marker clearly visible."
        )
    elif muac_mm < min_muac or muac_mm > max_muac:
        measurement_failure = "outside_expected_range"
        stage_status["muac_estimation"] = "outside_expected_range"
        message = (
            f"The estimated MUAC is outside the supported validity range "
            f"({min_muac:g}–{max_muac:g} mm) for {age_category}. No risk category was assigned. "
            "Retake the photo and confirm the measurement with a standard MUAC tape."
        )
    else:
        stage_status["muac_estimation"] = "ok"

    if measurement_failure is not None:
        logger.warning(
            "estimate rejected: reason=%s muac_mm=%s category=%s range=%.1f-%.1f segmentation_ok=%s",
            measurement_failure,
            muac_mm,
            age_category,
            min_muac,
            max_muac,
            seg_ok,
        )
        if diagnostics is not None:
            diagnostics.update({
                "measurement_valid": False,
                "measurement_status": "invalid",
                "estimated_muac_mm": None,
                "uncorrected_muac_mm": float(baseline_mm) if np.isfinite(baseline_mm) else None,
                "rejection_reason": measurement_failure,
            })
        return {
            "ok": False,
            "error": "invalid_measurement",
            "measurement_status": "invalid",
            "measurement_valid": False,
            "measurement_failure_reason": measurement_failure,
            "segmentation_ok": seg_ok,
            "message": message,
            "stage_status": stage_status,
        }

    classified = classify_risk_band(muac_mm, age_category)
    if diagnostics is not None:
        diagnostics.update({
            "measurement_valid": True,
            "measurement_status": "valid",
            "estimated_muac_mm": float(muac_mm),
            "uncorrected_muac_mm": float(baseline_mm) if np.isfinite(baseline_mm) else None,
            "rejection_reason": None,
        })
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
        "measurement_status": "valid",
        "measurement_valid": True,
        "message": "",
        "muac_estimate_mm": float(muac_mm),
        "risk_band": classified["risk_band"],
        "age_category": classified["age_category"],
        "confidence_score": confidence,
        "segmentation_ok": seg_ok,
        "arm_detected": True,
        "reference_detected": True,
        "baseline_muac_estimate_mm": float(baseline_mm),
        "stage_status": stage_status,
    }
