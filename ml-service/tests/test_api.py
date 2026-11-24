"""API tests for /health and /infer error paths."""

from __future__ import annotations

import io

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
import app.main as main_module

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_redirects_to_docs() -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (307, 302)
    assert response.headers["location"] == "/docs"


def test_infer_missing_age_category_returns_422() -> None:
    """POST /infer without age_category must return 422 age_category_required."""
    blank = np.full((240, 320, 3), 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", blank)
    assert ok
    response = client.post(
        "/infer",
        files={"image": ("blank.png", io.BytesIO(encoded.tobytes()), "image/png")},
        # age_category deliberately omitted
        data={"reference_type": "aruco", "reference_size_mm": "50.0"},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "age_category_required"
    assert "message" in body


def test_infer_invalid_age_category_returns_422() -> None:
    """POST /infer with an unrecognised age_category must return 422 age_category_required."""
    blank = np.full((240, 320, 3), 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", blank)
    assert ok
    response = client.post(
        "/infer",
        files={"image": ("blank.png", io.BytesIO(encoded.tobytes()), "image/png")},
        data={"age_category": "teenager", "reference_type": "aruco", "reference_size_mm": "50.0"},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "age_category_required"


def test_infer_rejects_undecodable_file() -> None:
    response = client.post(
        "/infer",
        files={"image": ("note.txt", b"not-an-image", "text/plain")},
        data={"age_category": "child_6_59m"},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "invalid_image"
    assert "message" in body


def test_infer_no_marker_returns_422() -> None:
    blank = np.full((240, 320, 3), 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", blank)
    assert ok
    response = client.post(
        "/infer",
        files={"image": ("blank.png", io.BytesIO(encoded.tobytes()), "image/png")},
        data={"age_category": "child_6_59m", "reference_type": "aruco", "reference_size_mm": "50.0"},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "reference_object_not_detected"


def test_infer_response_contract_unchanged_when_pipeline_has_internal_diagnostics(monkeypatch) -> None:
    def fake_pipeline(image, reference_size_mm, age_category):
        return {
            "ok": True,
            "muac_estimate_mm": 121.5,
            "risk_band": "MAM",
            "age_category": age_category,
            "confidence_score": 0.75,
            "segmentation_ok": True,
            "stage_status": {"reference": "detected"},
            "baseline_muac_estimate_mm": 119.5,
        }

    monkeypatch.setattr(main_module, "run_muac_pipeline", fake_pipeline)
    ok, encoded = cv2.imencode(".png", np.full((64, 64, 3), 255, dtype=np.uint8))
    assert ok
    response = client.post(
        "/infer",
        files={"image": ("image.png", io.BytesIO(encoded.tobytes()), "image/png")},
        data={"age_category": "child_6_59m"},
    )
    assert response.status_code == 200
    assert set(response.json()) == {
        "muac_estimate_mm", "risk_band", "age_category", "confidence_score",
        "quality_flags", "pipeline_version",
    }


def test_infer_invalid_measurement_returns_existing_error_shape_without_risk_band(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "run_muac_pipeline", lambda *_args: {
        "ok": False,
        "error": "invalid_measurement",
        "measurement_status": "invalid",
        "measurement_valid": False,
        "segmentation_ok": True,
        "message": "The estimated MUAC is outside the supported validity range. No risk category was assigned.",
        "stage_status": {"segmentation": "ok", "muac_estimation": "outside_expected_range"},
    })
    ok, encoded = cv2.imencode(".png", np.full((64, 64, 3), 255, dtype=np.uint8))
    assert ok
    response = client.post(
        "/infer",
        files={"image": ("image.png", io.BytesIO(encoded.tobytes()), "image/png")},
        data={"age_category": "child_6_59m"},
    )
    body = response.json()
    assert response.status_code == 422
    assert body["error"] == "invalid_measurement"
    assert "No risk category was assigned" in body["message"]
    assert "risk_band" not in body
    assert "muac_estimate_mm" not in body
    assert set(body) == {"error", "message"}

