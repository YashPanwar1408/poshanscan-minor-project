"""Synthetic regression tests for pose/mask axis selection and sampling."""

from __future__ import annotations

import numpy as np
import pytest

from app.pipeline import run as run_module
from app.pipeline.estimate import estimate_muac
from app.pipeline.geometry_debug import draw_measurement_geometry_debug


def _marker_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    corners = np.array(
        [[180.0, 180.0], [220.0, 180.0], [220.0, 220.0], [180.0, 220.0]],
        dtype=np.float32,
    )
    mask = np.zeros((280, 280), dtype=np.uint8)
    # A vertical segmented arm cropped around the marker center.
    mask[10:270, 105:175] = 255
    return image, corners, mask


def _prepare_marker_fallback(monkeypatch, mask: np.ndarray, angle: float = 11.06) -> list[float]:
    image, corners, _ = _marker_fixture()
    seen_angles: list[float] = []
    monkeypatch.setattr(run_module, "detect_reference_marker", lambda *_args: {
        "detected": True,
        "scale_px_per_mm": 2.0,
        "confidence": 0.95,
        "marker_id": 0,
        "marker_corners": corners.copy(),
    })
    monkeypatch.setattr(run_module, "locate_either_arm", lambda _image: {
        "detected": False,
        "midpoint_px": None,
        "shoulder_px": None,
        "elbow_px": None,
        "arm_angle_degrees": None,
        "confidence": 0.0,
    })
    monkeypatch.setattr(run_module, "locate_from_marker", lambda *_args: {
        "detected": True,
        "midpoint_px": (200.0, 200.0),
        "shoulder_px": None,
        "elbow_px": None,
        "arm_angle_degrees": angle,
        "confidence": 0.40,
        "arm_axis_source": "marker_skin_component",
    })
    monkeypatch.setattr(run_module, "segment_arm", lambda *_args, **_kwargs: {
        "mask": mask.copy(),
        "centroid_px": (140.0, 140.0),
        "crop_origin_px": (60, 60),
        "crop_bgr": image[60:340, 60:340].copy(),
        "segmentation_method": "yolo",
        "segmentation_confidence": 0.95,
    })
    original_measure = run_module.measure_width_perpendicular

    def record_angle(mask_arg, center, arm_angle, *args, **kwargs):
        seen_angles.append(float(arm_angle))
        return original_measure(mask_arg, center, arm_angle, *args, **kwargs)

    monkeypatch.setattr(run_module, "measure_width_perpendicular", record_angle)
    return seen_angles


def test_marker_fallback_uses_segmented_axis_instead_of_11_degree_skin_axis(monkeypatch) -> None:
    image, _corners, mask = _marker_fixture()
    seen_angles = _prepare_marker_fallback(monkeypatch, mask, angle=11.06)

    result = run_module.run_muac_pipeline(image, 50.0, "child_6_59m")

    assert seen_angles
    assert all(angle == pytest.approx(90.0, abs=1.0) for angle in seen_angles)
    assert result["ok"] is True
    assert result["muac_estimate_mm"] == pytest.approx(estimate_muac(70.0, 2.0)["muac_estimate_mm"], abs=2.0)
    assert result["muac_estimate_mm"] != pytest.approx(306.016, abs=0.1)
    assert result["segmentation_ok"] is True


def test_306_016_geometry_regression_fails_closed_when_segmentation_cannot_support_axis(monkeypatch) -> None:
    image, _corners, _valid_mask = _marker_fixture()
    invalid_mask = np.zeros((280, 280), dtype=np.uint8)
    _prepare_marker_fallback(monkeypatch, invalid_mask, angle=11.06)
    classification_calls: list[float] = []
    monkeypatch.setattr(
        run_module,
        "classify_risk_band",
        lambda value, _category: classification_calls.append(float(value)),
    )

    # These were the recorded pre-fix geometry inputs; this formula check does
    # not reconstruct the unavailable source image.
    previous_estimate = estimate_muac(220.0, 2.890936)["muac_estimate_mm"]
    assert previous_estimate == pytest.approx(306.016, abs=0.001)

    result = run_module.run_muac_pipeline(image, 50.0, "child_6_59m")

    assert result["ok"] is False
    assert result["error"] == "arm_axis_unreliable"
    assert "risk_band" not in result
    assert classification_calls == []


def test_missing_pose_and_invalid_segmentation_do_not_reuse_skin_angle(monkeypatch) -> None:
    image, _corners, _valid_mask = _marker_fixture()
    blank = np.zeros((280, 280), dtype=np.uint8)
    seen_angles = _prepare_marker_fallback(monkeypatch, blank, angle=11.06)

    result = run_module.run_muac_pipeline(image, 50.0, "child_6_59m")

    assert seen_angles == []
    assert result["error"] == "arm_axis_unreliable"
    assert result["segmentation_ok"] is False
    assert "risk_band" not in result


def test_pose_axis_conflict_with_strong_segmented_axis_fails_closed(monkeypatch) -> None:
    image, corners, mask = _marker_fixture()
    monkeypatch.setattr(run_module, "detect_reference_marker", lambda *_args: {
        "detected": True,
        "scale_px_per_mm": 2.0,
        "confidence": 0.95,
        "marker_id": 0,
        "marker_corners": corners,
    })
    monkeypatch.setattr(run_module, "locate_either_arm", lambda _image: {
        "detected": True,
        "midpoint_px": (200.0, 200.0),
        "shoulder_px": (200.0, 100.0),
        "elbow_px": (200.0, 300.0),
        "arm_angle_degrees": 11.06,
        "confidence": 0.95,
    })
    monkeypatch.setattr(run_module, "segment_arm", lambda *_args, **_kwargs: {
        "mask": mask,
        "centroid_px": (140.0, 140.0),
        "crop_origin_px": (60, 60),
        "crop_bgr": image[60:340, 60:340].copy(),
        "segmentation_method": "yolo",
        "segmentation_confidence": 0.95,
    })

    result = run_module.run_muac_pipeline(image, 50.0, "child_6_59m")

    assert result["error"] == "arm_axis_unreliable"
    assert "risk_band" not in result


def test_debug_overlay_contains_original_frame_and_diagnostic_panel() -> None:
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[4:36, 15:25] = 255
    overlay = draw_measurement_geometry_debug(
        image=image,
        marker_corners=np.array([[40, 35], [60, 35], [60, 55], [40, 55]], dtype=np.float32),
        pose={"shoulder_px": (20.0, 10.0), "elbow_px": (20.0, 70.0)},
        crop_origin_px=(30, 30),
        mask=mask,
        measurement_center_px=(20.0, 20.0),
        arm_axis_degrees=90.0,
        axis_source="segmentation_mask",
        width_px=10.0,
        width_diagnostics=[{
            "start_px": (15.0, 20.0),
            "end_px": (24.0, 20.0),
            "width_px": 10.0,
            "valid": True,
        }],
        mask_axis_degrees=90.0,
        mask_axis_ratio=4.0,
    )

    assert overlay.shape == (80, 600, 3)
    assert np.any(overlay[:, 100:] != 0)
    assert tuple(overlay[10, 20]) == (0, 255, 0)
