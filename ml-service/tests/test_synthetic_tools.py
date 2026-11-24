"""Coverage for synthetic calibration tooling and image dataset reports."""

from __future__ import annotations

import csv

import cv2
import numpy as np
import pytest

from app.main import configure_synthetic_calibration
from app.main import app
from scripts.generate_synthetic_calibration_data import generate_rows, write_rows
from scripts.image_inventory import inspect_images
from scripts.process_dataset_images import process_images
from scripts.train_synthetic_calibration import main as train_main, read_dataset


def test_synthetic_generation_is_reproducible_and_marked() -> None:
    first = generate_rows(n_samples=25, seed=17)
    second = generate_rows(n_samples=25, seed=17)
    assert first == second
    assert all(row["provenance"] == "simulated_numerical_demo_only" for row in first)


def test_training_validates_provenance_and_roundtrips_model(tmp_path) -> None:
    data_path = tmp_path / "synthetic.csv"
    rows = generate_rows(n_samples=50, seed=9)
    write_rows(data_path, rows)
    sample_ids, baseline, target = read_dataset(data_path)
    assert len(sample_ids) == len(baseline) == len(target) == 50

    model_path = tmp_path / "calibration.joblib"
    report_path = tmp_path / "report.json"
    predictions_path = tmp_path / "predictions.csv"
    result = train_main(
        [
            "--data", str(data_path), "--model", str(model_path),
            "--report-json", str(report_path), "--report-csv", str(predictions_path),
            "--seed", "12",
        ]
    )
    assert result == 0
    from app.pipeline.estimate import CalibrationCorrector

    loaded = CalibrationCorrector.load(model_path)
    assert loaded.is_fitted
    assert np.isfinite(loaded.predict(120.0))
    import json

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["report_type"] == "synthetic_experimental_results_only"
    assert report["training_sample_count"] + report["testing_sample_count"] == 50
    assert set(report["baseline_test_metrics"]) == {"mae_mm", "rmse_mm", "mean_signed_bias_mm"}
    with predictions_path.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == report["testing_sample_count"]


def test_training_rejects_non_synthetic_provenance(tmp_path) -> None:
    path = tmp_path / "mixed.csv"
    rows = generate_rows(n_samples=10)
    rows[0]["provenance"] = "tape_measurement"
    write_rows(path, rows)
    with pytest.raises(ValueError, match="must not be mixed"):
        read_dataset(path)


def test_synthetic_model_loader_falls_back_when_missing_or_corrupt(tmp_path) -> None:
    assert configure_synthetic_calibration(False) is False


def test_app_lifespan_loads_opt_in_model_then_resets(monkeypatch, tmp_path) -> None:
    from fastapi.testclient import TestClient

    from app.pipeline.estimate import CalibrationCorrector
    import app.pipeline.run as pipeline_module

    model_path = tmp_path / "demo.joblib"
    CalibrationCorrector().fit([100, 120, 140], [105, 125, 145]).save(model_path)
    monkeypatch.setenv("POSHANSCAN_ENABLE_SYNTHETIC_CALIBRATION", "true")
    monkeypatch.setenv("POSHANSCAN_SYNTHETIC_CALIBRATION_MODEL", str(model_path))
    monkeypatch.setenv("POSHANSCAN_SEGMENTATION_MODEL", "legacy")
    with TestClient(app) as startup_client:
        assert startup_client.get("/health").json() == {"status": "ok"}
        assert pipeline_module._calibrator.is_fitted
    assert not pipeline_module._calibrator.is_fitted
    assert configure_synthetic_calibration(True, tmp_path / "missing.joblib") is False
    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_bytes(b"not a joblib artifact")
    assert configure_synthetic_calibration(True, corrupt) is False
    assert configure_synthetic_calibration(False) is False


def test_inventory_reports_duplicates_corruption_and_marker_misses(tmp_path) -> None:
    valid_path = tmp_path / "first.png"
    ok, encoded = cv2.imencode(".png", np.full((64, 96, 3), 245, dtype=np.uint8))
    assert ok
    payload = encoded.tobytes()
    valid_path.write_bytes(payload)
    (tmp_path / "duplicate.PNG").write_bytes(payload)
    (tmp_path / "corrupt.jpg").write_bytes(b"broken image")
    (tmp_path / "unsupported.bmp").write_bytes(b"not an allowed format")

    summary, rows = inspect_images(tmp_path)
    assert summary["total_images"] == 4
    assert summary["valid_images"] == 2
    assert summary["invalid_images"] == 2
    assert summary["duplicate_images"] == 1
    assert summary["unique_image_count_after_exact_deduplication"] == 1
    assert summary["exact_duplicate_pair_count"] == 1
    assert len(summary["unique_image_inventory"]) == 1
    assert sum(bool(row["is_unique_image"]) for row in rows) == 1
    assert sum(bool(row["exact_duplicate_group"]) for row in rows) == 2
    assert summary["images_without_detectable_aruco_markers"] == 2
    assert all(row["dataset_label"] == "synthetic_unverified" for row in rows)


def test_processing_report_does_not_invent_measurements_for_marker_failure(tmp_path) -> None:
    ok, encoded = cv2.imencode(".png", np.full((80, 100, 3), 255, dtype=np.uint8))
    assert ok
    (tmp_path / "blank.png").write_bytes(encoded.tobytes())
    (tmp_path / "note.txt").write_text("not an image", encoding="utf-8")

    rows = process_images(tmp_path)
    blank = next(row for row in rows if row["filename"] == "blank.png")
    invalid = next(row for row in rows if row["filename"] == "note.txt")
    assert blank["image_loading_status"] == "loaded"
    assert blank["aruco_detection_status"] == "not_detected"
    assert blank["corrected_muac_estimate_mm"] == ""
    assert "marker" in blank["processing_error"].lower()
    assert invalid["image_loading_status"] == "invalid_format"
    assert invalid["corrected_muac_estimate_mm"] == ""


def test_processing_report_omits_risk_band_for_out_of_range_result(monkeypatch, tmp_path) -> None:
    ok, encoded = cv2.imencode(".png", np.full((80, 100, 3), 255, dtype=np.uint8))
    assert ok
    (tmp_path / "numbered.png").write_bytes(encoded.tobytes())

    import scripts.process_dataset_images as processing_module

    monkeypatch.setattr(
        processing_module,
        "run_muac_pipeline",
        lambda *_args: {
            "ok": True,
            "baseline_muac_estimate_mm": 240.0,
            "muac_estimate_mm": 240.0,
            "risk_band": "Normal",
            "stage_status": {
                "reference": "detected",
                "pose": "marker_fallback",
                "segmentation": "ok",
                "muac_estimation": "outside_expected_range",
            },
        },
    )
    row = processing_module.process_images(tmp_path)[0]
    assert row["corrected_muac_estimate_mm"] == 240.0
    assert row["risk_band"] == ""
    assert "outside" in row["processing_error"].lower()
