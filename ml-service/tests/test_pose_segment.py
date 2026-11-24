"""Synthetic-image tests for pose localisation and arm segmentation."""

from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.pose_localize import (
    arm_axis_angle_degrees,
    axis_angle_difference_degrees,
    estimate_arm_axis_from_mask,
    locate_arm_midpoint,
    locate_from_marker,
    midpoint_px,
    normalize_axis_angle_degrees,
)
from app.pipeline.segment import (
    measure_width_at_row,
    measure_width_perpendicular,
    remove_polygon_from_mask,
    segment_arm,
)

# BGR skin tone inside the HSV inRange bands used by segment.py.
SKIN_BGR = (90, 150, 210)
BG_BGR = (30, 30, 30)
CANVAS = 400
ARM_X0, ARM_X1 = 150, 250  # 100 px wide vertical "arm"
ARM_Y0, ARM_Y1 = 50, 350
ARM_WIDTH_PX = ARM_X1 - ARM_X0
MIDPOINT = ((ARM_X0 + ARM_X1) / 2.0, (ARM_Y0 + ARM_Y1) / 2.0)
CROP_SIZE = 200


def _synthetic_arm_image() -> np.ndarray:
    image = np.full((CANVAS, CANVAS, 3), BG_BGR, dtype=np.uint8)
    image[ARM_Y0:ARM_Y1, ARM_X0:ARM_X1] = SKIN_BGR
    return image


def test_midpoint_and_arm_angle_helpers() -> None:
    shoulder = (100.0, 80.0)
    elbow = (100.0, 180.0)
    mid = midpoint_px(shoulder, elbow)
    assert mid == pytest.approx((100.0, 130.0))
    # Vertical arm pointing down → ~90° in image coordinates.
    assert arm_axis_angle_degrees(shoulder, elbow) == pytest.approx(90.0, abs=1e-6)


def test_locate_from_marker_uses_sticker_center() -> None:
    image = _synthetic_arm_image()
    corners = np.array(
        [[180.0, 180.0], [220.0, 180.0], [220.0, 220.0], [180.0, 220.0]],
        dtype=np.float32,
    )
    result = locate_from_marker(image, corners)
    assert result["detected"] is True
    assert result["midpoint_px"] == pytest.approx((200.0, 200.0))
    assert result["arm_angle_degrees"] is not None


def test_locate_arm_midpoint_handles_synthetic_image() -> None:
    image = _synthetic_arm_image()
    result = locate_arm_midpoint(image, side="left")

    assert set(result) >= {
        "detected",
        "midpoint_px",
        "shoulder_px",
        "elbow_px",
        "arm_angle_degrees",
        "confidence",
    }
    assert result["detected"] in (True, False)
    assert 0.0 <= result["confidence"] <= 1.0
    if result["detected"]:
        x, y = result["midpoint_px"]
        assert 0 <= x < CANVAS
        assert 0 <= y < CANVAS
        assert result["shoulder_px"] is not None
        assert result["elbow_px"] is not None
        assert result["arm_angle_degrees"] is not None
    else:
        assert result["midpoint_px"] is None


@pytest.mark.parametrize("image", [None, np.array([]), np.zeros((0, 0, 3), dtype=np.uint8)])
def test_locate_arm_midpoint_empty_image(image: np.ndarray | None) -> None:
    result = locate_arm_midpoint(image, side="left")
    assert result["detected"] is False
    assert result["midpoint_px"] is None


def test_segment_arm_on_skin_rectangle() -> None:
    image = _synthetic_arm_image()
    result = segment_arm(image, MIDPOINT, crop_size_px=CROP_SIZE)

    mask = result["mask"]
    assert mask.shape == (CROP_SIZE, CROP_SIZE)
    assert mask.dtype == np.uint8
    assert int(np.count_nonzero(mask)) > 1000
    assert result["centroid_px"] is not None
    cx, cy = result["centroid_px"]
    assert abs(cx - CROP_SIZE / 2) < 30
    assert 0 <= cy < CROP_SIZE


def test_measure_width_at_row_matches_synthetic_arm() -> None:
    image = _synthetic_arm_image()
    result = segment_arm(image, MIDPOINT, crop_size_px=CROP_SIZE)
    row = int(round(result["centroid_px"][1])) if result["centroid_px"] else CROP_SIZE // 2
    width = measure_width_at_row(result["mask"], row)
    assert width == pytest.approx(ARM_WIDTH_PX, rel=0.25, abs=15)


def test_measure_width_at_row_empty_or_oob() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    assert measure_width_at_row(mask, 10) == 0.0
    assert measure_width_at_row(mask, -1) == 0.0
    assert measure_width_at_row(mask, 99) == 0.0
    assert measure_width_at_row(None, 0) == 0.0


def test_measure_width_perpendicular_can_bridge_reference_gap() -> None:
    mask = np.zeros((300, 300), dtype=np.uint8)
    mask[20:280, 20:80] = 255
    mask[140:160, 45:55] = 0
    assert measure_width_perpendicular(mask, (50, 150), 90) == pytest.approx(25.0)
    assert measure_width_perpendicular(mask, (50, 150), 90, bridge_internal_gaps=True) == pytest.approx(60.0)


@pytest.mark.parametrize("arm_angle", [0.0, 90.0, 27.0, 135.0])
def test_mask_axis_and_direct_width_sampling_for_horizontal_vertical_and_diagonal_arms(arm_angle: float) -> None:
    import cv2

    size = 360
    center = np.array([180.0, 180.0])
    radians = np.deg2rad(arm_angle)
    axis = np.array([np.cos(radians), np.sin(radians)])
    normal = np.array([-axis[1], axis[0]])
    corners = np.array([
        center - 130 * axis - 22 * normal,
        center + 130 * axis - 22 * normal,
        center + 130 * axis + 22 * normal,
        center - 130 * axis + 22 * normal,
    ], dtype=np.int32)
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillConvexPoly(mask, corners, 255)

    estimate = estimate_arm_axis_from_mask(mask, tuple(center))
    assert estimate["angle_degrees"] is not None
    assert axis_angle_difference_degrees(float(estimate["angle_degrees"]), arm_angle) < 1.0
    assert measure_width_perpendicular(mask, tuple(center), float(estimate["angle_degrees"])) == pytest.approx(
        44.0, abs=2.0
    )


def test_rotated_mask_keeps_axis_and_cross_section_in_original_coordinates() -> None:
    import cv2

    mask = np.zeros((360, 360), dtype=np.uint8)
    mask[45:315, 158:202] = 255
    center = np.array([180.0, 180.0])
    rotation = cv2.getRotationMatrix2D(tuple(center), 31.0, 1.0)
    rotated = cv2.warpAffine(mask, rotation, (360, 360), flags=cv2.INTER_NEAREST)
    new_point = rotation @ np.array([180.0, 70.0, 1.0])
    expected_angle = arm_axis_angle_degrees(tuple(center), tuple(new_point))

    estimate = estimate_arm_axis_from_mask(rotated, tuple(center))
    assert estimate["angle_degrees"] is not None
    assert axis_angle_difference_degrees(float(estimate["angle_degrees"]), expected_angle) < 1.0
    assert measure_width_perpendicular(rotated, tuple(center), float(estimate["angle_degrees"])) == pytest.approx(
        44.0, abs=2.0
    )


def test_wrong_marker_axis_is_rejected_when_mask_shows_vertical_arm() -> None:
    mask = np.zeros((300, 300), dtype=np.uint8)
    mask[20:280, 110:190] = 255
    diagnostics: list[dict[str, object]] = []
    width = measure_width_perpendicular(mask, (150, 150), 11.06, diagnostics=diagnostics)

    assert width == 0.0
    assert diagnostics[0]["reason"] == "arm_axis_disagrees_with_segmented_mask"


def test_invalid_segmentation_has_no_mask_axis_or_measurable_width() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    estimate = estimate_arm_axis_from_mask(mask, (50, 50))
    assert estimate["angle_degrees"] is None
    assert measure_width_perpendicular(mask, (50, 50), 90.0) == 0.0


def test_width_sampler_rejects_a_cross_section_clipped_by_crop_boundary() -> None:
    mask = np.zeros((300, 300), dtype=np.uint8)
    mask[20:280, 0:80] = 255
    diagnostics: list[dict[str, object]] = []

    width = measure_width_perpendicular(mask, (40, 150), 90.0, diagnostics=diagnostics)

    assert width == 0.0
    assert diagnostics[0]["touches_mask_boundary"] is True
    assert diagnostics[0]["valid"] is False


def test_axis_angle_normalization_uses_unoriented_image_axes() -> None:
    assert normalize_axis_angle_degrees(-90.0) == pytest.approx(90.0)
    assert axis_angle_difference_degrees(11.0, 191.0) == pytest.approx(0.0)


def test_remove_polygon_from_mask_excludes_reference_marker() -> None:
    mask = np.full((100, 100), 255, dtype=np.uint8)
    cleaned = remove_polygon_from_mask(
        mask,
        np.array([[40, 40], [60, 40], [60, 60], [40, 60]], dtype=np.float32),
        (0, 0),
    )
    assert int(cleaned[50, 50]) == 0
    assert int(cleaned[20, 20]) == 255
