"""Localise the mid-upper-arm measurement site with MediaPipe Pose."""

from __future__ import annotations

import threading
from typing import Any, Optional, TypedDict

import cv2
import numpy as np

# MediaPipe Pose landmark indices (subject's own left/right).
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_ELBOW = 13
_RIGHT_ELBOW = 14

_MIN_LANDMARK_VISIBILITY = 0.15

_pose_lock = threading.Lock()
_pose_model: Any = None


class ArmMidpointResult(TypedDict):
    """Result of :func:`locate_arm_midpoint`."""

    detected: bool
    midpoint_px: Optional[tuple[float, float]]
    shoulder_px: Optional[tuple[float, float]]
    elbow_px: Optional[tuple[float, float]]
    arm_angle_degrees: Optional[float]
    confidence: float


def _empty_result() -> ArmMidpointResult:
    return {
        "detected": False,
        "midpoint_px": None,
        "shoulder_px": None,
        "elbow_px": None,
        "arm_angle_degrees": None,
        "confidence": 0.0,
    }


def _is_empty_image(image: Any) -> bool:
    if image is None:
        return True
    array = np.asarray(image)
    return array.size == 0 or array.ndim < 2 or array.shape[0] == 0 or array.shape[1] == 0


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _get_pose() -> Any:
    """Lazy singleton so the Pose graph is not rebuilt on every call."""
    global _pose_model
    if _pose_model is not None:
        return _pose_model
    with _pose_lock:
        if _pose_model is None:
            import mediapipe as mp

            _pose_model = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3,
            )
    return _pose_model


def _landmark_to_px(landmark: Any, width: int, height: int) -> tuple[float, float]:
    return float(landmark.x * width), float(landmark.y * height)


def _landmark_visibility(landmark: Any) -> float:
    visibility = getattr(landmark, "visibility", None)
    if visibility is None:
        visibility = getattr(landmark, "presence", 0.0)
    return float(visibility)


def midpoint_px(
    shoulder_px: tuple[float, float],
    elbow_px: tuple[float, float],
) -> tuple[float, float]:
    """Pixel midpoint between shoulder and elbow (MUAC tape location)."""
    return (
        (shoulder_px[0] + elbow_px[0]) / 2.0,
        (shoulder_px[1] + elbow_px[1]) / 2.0,
    )


def arm_axis_angle_degrees(
    shoulder_px: tuple[float, float],
    elbow_px: tuple[float, float],
) -> float:
    """Angle of the shoulder→elbow axis in degrees (OpenCV image coords).

    0° points right, 90° points down. Downstream width measurement should
    be taken perpendicular to this axis, not along image rows.
    """
    dx = elbow_px[0] - shoulder_px[0]
    dy = elbow_px[1] - shoulder_px[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def normalize_axis_angle_degrees(angle_degrees: float) -> float:
    """Normalize an unoriented image axis to [0, 180) degrees."""
    return float(float(angle_degrees) % 180.0)


def axis_angle_difference_degrees(first: float, second: float) -> float:
    """Smallest angle between unoriented axes (range 0 through 90 degrees)."""
    difference = abs(normalize_axis_angle_degrees(first) - normalize_axis_angle_degrees(second))
    return float(min(difference, 180.0 - difference))


def estimate_arm_axis_from_mask(
    mask: np.ndarray,
    center_px: Optional[tuple[float, float]] = None,
    min_foreground_px: int = 80,
    min_eigenvalue_ratio: float = 1.2,
) -> dict[str, Any]:
    """Estimate the local arm axis from nearby segmented foreground pixels.

    Connected components far from the measurement center are excluded. This
    permits the two visible arm pieces around an ArUco occlusion to contribute
    together while reducing the influence of unrelated crop contents.
    """
    if mask is None or not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.size == 0:
        return {"angle_degrees": None, "eigenvalue_ratio": 0.0, "foreground_px": 0}
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return {"angle_degrees": None, "eigenvalue_ratio": 0.0, "foreground_px": 0}

    height, width = binary.shape
    if center_px is None:
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    else:
        cx, cy = (float(center_px[0]), float(center_px[1]))
        if not np.isfinite(cx) or not np.isfinite(cy):
            return {"angle_degrees": None, "eigenvalue_ratio": 0.0, "foreground_px": 0}

    components: list[tuple[int, float, int]] = []
    for label in range(1, count):
        x, y, component_width, component_height, area = (int(v) for v in stats[label])
        if area < max(20, min_foreground_px // 8):
            continue
        dx = max(float(x) - cx, 0.0, cx - float(x + component_width - 1))
        dy = max(float(y) - cy, 0.0, cy - float(y + component_height - 1))
        components.append((label, float(np.hypot(dx, dy)), area))
    if not components:
        return {"angle_degrees": None, "eigenvalue_ratio": 0.0, "foreground_px": 0}

    nearest_distance = min(item[1] for item in components)
    proximity = max(12.0, min(height, width) * 0.12)
    largest_area = max(item[2] for item in components)
    selected_labels = {
        label
        for label, distance, area in components
        if distance <= nearest_distance + proximity and area >= max(20, largest_area * 0.05)
    }
    ys, xs = np.where(np.isin(labels, list(selected_labels)))
    if xs.size < min_foreground_px:
        return {"angle_degrees": None, "eigenvalue_ratio": 0.0, "foreground_px": int(xs.size)}

    points = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    covariance = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)
    minor = max(float(eigenvalues[order[0]]), 1e-9)
    major = float(eigenvalues[order[-1]])
    ratio = major / minor
    if not np.isfinite(ratio) or ratio < float(min_eigenvalue_ratio):
        return {"angle_degrees": None, "eigenvalue_ratio": float(ratio), "foreground_px": int(xs.size)}
    vector = eigenvectors[:, order[-1]]
    angle = normalize_axis_angle_degrees(float(np.degrees(np.arctan2(vector[1], vector[0]))))
    return {"angle_degrees": angle, "eigenvalue_ratio": float(ratio), "foreground_px": int(xs.size)}


def locate_arm_midpoint(
    image: np.ndarray,
    side: str = "left",
) -> ArmMidpointResult:
    """Detect shoulder/elbow and the mid-upper-arm point for one arm.

    Args:
        image: BGR (or grayscale) numpy array.
        side: ``"left"`` or ``"right"`` from the subject's perspective
            (MediaPipe Pose convention).

    Returns:
        Dict with ``detected``, ``midpoint_px``, ``shoulder_px``,
        ``elbow_px``, ``arm_angle_degrees``, and ``confidence``.
        On failure ``detected`` is False and coordinate fields are None.
    """
    if _is_empty_image(image):
        return _empty_result()

    side_key = (side or "left").strip().lower()
    if side_key not in {"left", "right"}:
        return _empty_result()

    bgr = _to_bgr(np.asarray(image))
    height, width = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    try:
        results = _get_pose().process(rgb)
    except Exception:
        return _empty_result()

    if results.pose_landmarks is None:
        return _empty_result()

    landmarks = results.pose_landmarks.landmark
    if side_key == "left":
        shoulder_lm = landmarks[_LEFT_SHOULDER]
        elbow_lm = landmarks[_LEFT_ELBOW]
    else:
        shoulder_lm = landmarks[_RIGHT_SHOULDER]
        elbow_lm = landmarks[_RIGHT_ELBOW]

    shoulder_vis = _landmark_visibility(shoulder_lm)
    elbow_vis = _landmark_visibility(elbow_lm)
    confidence = (shoulder_vis + elbow_vis) / 2.0

    if confidence < _MIN_LANDMARK_VISIBILITY:
        return {
            "detected": False,
            "midpoint_px": None,
            "shoulder_px": None,
            "elbow_px": None,
            "arm_angle_degrees": None,
            "confidence": float(confidence),
        }

    shoulder_px = _landmark_to_px(shoulder_lm, width, height)
    elbow_px = _landmark_to_px(elbow_lm, width, height)
    mid = midpoint_px(shoulder_px, elbow_px)
    angle = arm_axis_angle_degrees(shoulder_px, elbow_px)

    return {
        "detected": True,
        "midpoint_px": mid,
        "shoulder_px": shoulder_px,
        "elbow_px": elbow_px,
        "arm_angle_degrees": angle,
        "confidence": float(confidence),
    }


def localize_measurement_site(
    image: np.ndarray,
    side: str = "left",
) -> Optional[tuple[float, float]]:
    """Compatibility wrapper returning only the MUAC midpoint, or None."""
    result = locate_arm_midpoint(image, side=side)
    return result["midpoint_px"] if result["detected"] else None


def _marker_center(marker_corners: np.ndarray) -> tuple[float, float]:
    pts = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def _skin_axis_angle_degrees(
    image: np.ndarray,
    center_px: tuple[float, float],
    window_px: int = 400,
) -> Optional[float]:
    """Estimate a local axis from the nearest connected skin component.

    This is retained as a fallback diagnostic only. The measurement pipeline
    requires a usable segmentation axis when pose landmarks are absent.
    """
    bgr = _to_bgr(np.asarray(image))
    height, width = bgr.shape[:2]
    cx, cy = int(round(center_px[0])), int(round(center_px[1]))
    half = max(window_px // 2, 80)
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(width, cx + half), min(height, cy + half)
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    skin = cv2.inRange(hsv, np.array([0, 20, 40], dtype=np.uint8), np.array([25, 255, 255], dtype=np.uint8))
    skin2 = cv2.inRange(hsv, np.array([160, 20, 40], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8))
    skin = cv2.bitwise_or(skin, skin2)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (skin > 0).astype(np.uint8), connectivity=8
    )
    local_center = (float(cx - x0), float(cy - y0))
    candidates: list[tuple[int, float, int]] = []
    for label in range(1, count):
        x, y, component_width, component_height, area = (int(v) for v in stats[label])
        if area < 40:
            continue
        dx = max(float(x) - local_center[0], 0.0, local_center[0] - float(x + component_width - 1))
        dy = max(float(y) - local_center[1], 0.0, local_center[1] - float(y + component_height - 1))
        candidates.append((label, float(np.hypot(dx, dy)), area))
    if not candidates:
        return None

    nearest = min(candidate[1] for candidate in candidates)
    keep_distance = nearest + max(8.0, window_px * 0.08)
    largest = max(candidate[2] for candidate in candidates)
    chosen = {
        label
        for label, distance, area in candidates
        if distance <= keep_distance and area >= max(40, largest * 0.05)
    }
    ys, xs = np.where(np.isin(labels, list(chosen)))
    if xs.size < 80:
        return None
    points = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(points, rowvar=False))
    order = np.argsort(eigenvalues)
    minor = max(float(eigenvalues[order[0]]), 1e-9)
    ratio = float(eigenvalues[order[-1]]) / minor
    if not np.isfinite(ratio) or ratio < 1.2:
        return None
    vector = eigenvectors[:, order[-1]]
    return normalize_axis_angle_degrees(float(np.degrees(np.arctan2(vector[1], vector[0]))))


def locate_from_marker(
    image: np.ndarray,
    marker_corners: np.ndarray,
) -> ArmMidpointResult:
    """Fallback MUAC site: the ArUco marker is placed on the mid-upper arm.

    Used when MediaPipe Pose cannot see a full body (close-up arm photos).
    """
    if _is_empty_image(image) or marker_corners is None:
        return _empty_result()
    try:
        mid = _marker_center(marker_corners)
        points = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
        marker_side = float(
            np.mean(
                [
                    np.linalg.norm(points[(i + 1) % 4] - points[i])
                    for i in range(4)
                ]
            )
        )
    except Exception:
        return _empty_result()
    # A full-image skin PCA can mix the upper arm with the forearm or face.
    # Keep the fallback axis local to the marker neighbourhood.
    angle = _skin_axis_angle_degrees(
        image,
        mid,
        window_px=int(np.clip(marker_side * 4.0, 100, 220)),
    )
    return {
        "detected": True,
        "midpoint_px": mid,
        "shoulder_px": None,
        "elbow_px": None,
        "arm_angle_degrees": angle,
        "arm_axis_source": "marker_skin_component" if angle is not None else "marker_center_only",
        "confidence": 0.40,
    }
