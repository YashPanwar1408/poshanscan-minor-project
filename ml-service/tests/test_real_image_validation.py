"""Unit tests for validation tooling; temporary pixels here are not real-image evidence."""

from __future__ import annotations

import csv
import json

import cv2
import numpy as np
import pytest

from scripts import validate_real_arm_photos as validation


def test_load_tape_references_requires_explicit_valid_manual_values(tmp_path) -> None:
    references = tmp_path / "references.csv"
    references.write_text("image_id,reference_muac_mm\nchild_01.jpg,123.5\n", encoding="utf-8")

    assert validation.load_tape_references(references) == {"child_01.jpg": 123.5}


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-5", "not-a-number"])
def test_load_tape_references_rejects_invalid_values(tmp_path, value) -> None:
    references = tmp_path / "references.csv"
    references.write_text(f"image_id,reference_muac_mm\narm.jpg,{value}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        validation.load_tape_references(references)


def test_paired_error_metrics_calculate_only_finite_pairs() -> None:
    metrics = validation.paired_error_metrics([101.0, 118.0, float("nan")], [100.0, 120.0, 140.0], (2.0, 5.0))

    assert metrics["paired_valid_count"] == 2
    assert metrics["mae_mm"] == pytest.approx(1.5)
    assert metrics["rmse_mm"] == pytest.approx(np.sqrt(2.5))
    assert metrics["bias_mm"] == pytest.approx(-0.5)
    assert metrics["within_tolerance_percent"] == {"2mm": 100.0, "5mm": 100.0}


def test_discover_images_skips_output_subtree_and_sorts_ids(tmp_path) -> None:
    inputs = tmp_path / "photos"
    (inputs / "nested").mkdir(parents=True)
    output = inputs / "validation_output"
    (output / "overlays").mkdir(parents=True)
    for path in [inputs / "b.JPG", inputs / "nested" / "a.png", output / "overlays" / "old.png"]:
        path.write_bytes(b"placeholder")

    found = validation.discover_images(inputs, output)

    assert [path.relative_to(inputs).as_posix() for path in found] == ["b.JPG", "nested/a.png"]


def test_folder_runner_writes_overlay_csv_and_sidecar_without_calling_synthetic_accuracy(tmp_path, monkeypatch) -> None:
    # This generated bitmap exists only to exercise file IO/overlay code. The
    # test deliberately does not confirm it as a real photograph, so no
    # accuracy metrics can be emitted from it.
    image_dir = tmp_path / "input"
    image_dir.mkdir()
    image = np.full((60, 80, 3), 80, dtype=np.uint8)
    image_path = image_dir / "arm.png"
    assert cv2.imwrite(str(image_path), image)
    references_path = tmp_path / "refs.csv"
    references_path.write_text("image_id,reference_muac_mm\narm.png,119.0\n", encoding="utf-8")

    monkeypatch.setattr(validation, "configure_segmentation_model", lambda *_args: True)
    monkeypatch.setattr(validation, "segmentation_mode", lambda: "yolo")
    monkeypatch.setattr(validation, "set_calibrator", lambda *_args: None)

    def fake_pipeline(_image, _marker_size, _age, diagnostics):
        diagnostics.update({
            "marker_detected": True,
            "marker_id": 0,
            "marker_corners_xy": [[30, 25], [42, 25], [42, 37], [30, 37]],
            "marker_center_px": [36, 31],
            "marker_scale_px_per_mm": 0.24,
            "segmentation_status": "ok",
            "segmentation_method": "yolo",
            "segmentation_confidence": 0.9,
            "pose_available": False,
            "pose_source": "marker_fallback",
            "pose": {"shoulder_px": None, "elbow_px": None},
            "segmentation_mask": np.ones((30, 30), dtype=np.uint8),
            "measurement_mask": np.ones((30, 30), dtype=np.uint8),
            "crop_origin_px": (20, 16),
            "measurement_center_crop_px": (16.0, 15.0),
            "mask_axis_degrees": 90.0,
            "mask_axis_eigenvalue_ratio": 8.0,
            "axis_method": "segmentation_mask",
            "axis_degrees": 90.0,
            "axis_confidence": 0.875,
            "axis_agreement_degrees": None,
            "accepted_samples": 1,
            "rejected_samples": 1,
            "width_samples": [
                {"start_px": [5, 15], "end_px": [24, 15], "width_px": 20.0, "valid": True},
                {"start_px": [4, 14], "end_px": [25, 14], "width_px": 22.0, "valid": False},
            ],
            "sampled_width_px": 20.0,
        })
        return {
            "ok": True,
            "measurement_valid": True,
            "muac_estimate_mm": 120.0,
            "risk_band": "MAM",
            "stage_status": {"segmentation": "ok", "muac_estimation": "ok"},
        }

    monkeypatch.setattr(validation, "run_muac_pipeline", fake_pipeline)
    out = tmp_path / "out"
    summary = validation.run_folder_validation(
        image_dir=image_dir,
        output_dir=out,
        references_path=references_path,
        real_photos_confirmed=False,
    )

    assert summary["image_count"] == 1
    assert summary["valid_measurement_count"] == 1
    assert summary["metrics"] is None
    rows = list(csv.DictReader((out / "measurements.csv").open(encoding="utf-8-sig")))
    assert rows[0]["measurement_valid"] == "True"
    assert rows[0]["estimated_muac_mm"] == "120.0"
    assert rows[0]["accepted_samples"] == "1"
    overlay_path = out / rows[0]["overlay_path"]
    sidecar_path = out / rows[0]["diagnostics_path"]
    assert cv2.imread(str(overlay_path)) is not None
    details = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert len(details["diagnostics"]["width_samples"]) == 2
    assert details["diagnostics"]["width_samples"][1]["valid"] is False
    report = (out / "validation_report.md").read_text(encoding="utf-8")
    assert "Not calculated" in report


def test_invalid_measurement_row_omits_estimate_and_error() -> None:
    row = validation._make_row(
        image_path=__import__("pathlib").Path("image.png"),
        relative_id="image.png",
        result={"ok": False, "error": "invalid_measurement"},
        diagnostics={"marker_detected": True, "rejection_reason": "outside_expected_range"},
        image_sha256=None,
        reference_muac_mm=306.0,
        overlay_path=None,
    )

    assert row["measurement_valid"] is False
    assert row["estimated_muac_mm"] is None
    assert row["error_mm"] is None
    assert row["rejection_reason"] == "outside_expected_range"
