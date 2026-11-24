"""Safety checks for final MUAC validity before assigning a screening band."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.pipeline import run as run_module
from app.pipeline.classify import classify_risk_band as real_classify_risk_band


def _run_pipeline_with_final_muac(
    monkeypatch,
    final_muac_mm: float,
    age_category: str,
    baseline_muac_mm: float = 120.0,
):
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    mask = np.full((200, 200), 255, dtype=np.uint8)
    classification_calls: list[tuple[float, str]] = []

    monkeypatch.setattr(run_module, "detect_reference_marker", lambda *_args: {
        "detected": True,
        "scale_px_per_mm": 1.0,
        "confidence": 0.95,
        "marker_id": 0,
        "marker_corners": None,
    })
    monkeypatch.setattr(run_module, "locate_either_arm", lambda _image: {
        "detected": True,
        "midpoint_px": (250.0, 250.0),
        "shoulder_px": (250.0, 100.0),
        "elbow_px": (250.0, 400.0),
        "arm_angle_degrees": 90.0,
        "confidence": 0.95,
    })
    monkeypatch.setattr(run_module, "segment_arm", lambda *_args, **_kwargs: {
        "mask": mask.copy(),
        "centroid_px": (100.0, 100.0),
        "crop_origin_px": (150, 150),
        "crop_bgr": image[:200, :200].copy(),
        "segmentation_method": "yolo",
        "segmentation_confidence": 0.95,
    })
    monkeypatch.setattr(run_module, "measure_width_perpendicular", lambda *_args, **_kwargs: 20.0)
    monkeypatch.setattr(run_module, "estimate_muac", lambda *_args: {
        "width_mm": 30.0,
        "muac_estimate_mm": baseline_muac_mm,
        "shape_factor_used": 1.28,
    })

    def calibration_spy(baseline: float) -> float:
        if not np.isfinite(baseline) or baseline <= 0:
            pytest.fail("invalid baseline reached calibration")
        return final_muac_mm

    monkeypatch.setattr(run_module, "_calibrator", SimpleNamespace(predict=calibration_spy))

    def classify_spy(muac_mm: float, category: str):
        classification_calls.append((muac_mm, category))
        return real_classify_risk_band(muac_mm, category)

    monkeypatch.setattr(run_module, "classify_risk_band", classify_spy)
    result = run_module.run_muac_pipeline(image, 50.0, age_category)
    return result, classification_calls


@pytest.mark.parametrize(
    ("muac_mm", "age_category", "reason"),
    [
        (306.016, "child_6_59m", "outside_expected_range"),
        (69.99, "child_6_59m", "outside_expected_range"),
        (180.01, "child_6_59m", "outside_expected_range"),
        (99.99, "adult", "outside_expected_range"),
        (450.01, "adult", "outside_expected_range"),
        (0.0, "child_6_59m", "non_positive"),
        (-1.0, "child_6_59m", "non_positive"),
        (float("nan"), "child_6_59m", "non_finite"),
        (float("inf"), "child_6_59m", "non_finite"),
        (float("-inf"), "child_6_59m", "non_finite"),
    ],
)
def test_invalid_final_muac_is_rejected_before_classification(monkeypatch, muac_mm, age_category, reason) -> None:
    result, classification_calls = _run_pipeline_with_final_muac(monkeypatch, muac_mm, age_category)

    assert result["ok"] is False
    assert result["error"] == "invalid_measurement"
    assert result["measurement_status"] == "invalid"
    assert result["measurement_valid"] is False
    assert result["measurement_failure_reason"] == reason
    assert result["segmentation_ok"] is True
    assert result["stage_status"]["segmentation"] == "ok"
    assert "risk_band" not in result
    assert classification_calls == []
    if reason == "outside_expected_range":
        assert "supported validity range" in result["message"]
        assert "No risk category was assigned" in result["message"]


@pytest.mark.parametrize(
    ("muac_mm", "age_category", "expected_band"),
    [
        (70.0, "child_6_59m", "SAM"),
        (125.0, "child_6_59m", "Normal"),
        (180.0, "child_6_59m", "Normal"),
        (100.0, "adult", "Severe"),
        (450.0, "adult", "Normal"),
    ],
)
def test_measurements_inside_inclusive_validity_range_are_classified(
    monkeypatch,
    muac_mm: float,
    age_category: str,
    expected_band: str,
) -> None:
    result, classification_calls = _run_pipeline_with_final_muac(monkeypatch, muac_mm, age_category)

    assert result["ok"] is True
    assert result["measurement_status"] == "valid"
    assert result["measurement_valid"] is True
    assert result["muac_estimate_mm"] == pytest.approx(muac_mm)
    assert result["risk_band"] == expected_band
    assert result["segmentation_ok"] is True
    assert len(classification_calls) == 1


def test_non_finite_geometric_estimate_is_not_passed_to_calibrator(monkeypatch) -> None:
    result, classification_calls = _run_pipeline_with_final_muac(
        monkeypatch,
        125.0,
        "child_6_59m",
        baseline_muac_mm=float("nan"),
    )
    assert result["error"] == "invalid_measurement"
    assert result["measurement_failure_reason"] == "non_finite"
    assert "risk_band" not in result
    assert classification_calls == []
