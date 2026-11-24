from __future__ import annotations

import hashlib
import logging

import pytest

from app.services import cv_client


@pytest.mark.asyncio
async def test_cv_client_forwards_exact_bytes_mime_and_opt_in_fingerprint(monkeypatch, caplog) -> None:
    image_bytes = b"original uploaded image bytes"
    captured: dict[str, object] = {}

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {
                "muac_estimate_mm": 125.0,
                "risk_band": "Normal",
                "confidence_score": 0.8,
                "quality_flags": {
                    "reference_detected": True,
                    "arm_detected": True,
                    "segmentation_ok": True,
                },
            }

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, *, files, data):
            captured.update(url=url, files=files, data=data)
            return FakeResponse()

    monkeypatch.setattr(cv_client.httpx, "AsyncClient", lambda **_kwargs: FakeClient())
    monkeypatch.setattr(cv_client.settings, "CV_SERVICE_URL", "http://ml.test")
    monkeypatch.setenv("POSHANSCAN_IMAGE_DIAGNOSTICS", "1")

    with caplog.at_level(logging.INFO, logger="poshanscan-backend.cv"):
        result = await cv_client.run_cv_inference(
            image_bytes=image_bytes,
            filename="capture.jpg",
            content_type="image/webp",
            age_category="adult",
        )

    assert result.muac_estimate_mm == 125.0
    assert captured["url"] == "http://ml.test/infer"
    upload = captured["files"]["image"]
    assert upload == ("capture.jpg", image_bytes, "image/webp")
    assert captured["data"]["age_category"] == "adult"
    assert hashlib.sha256(image_bytes).hexdigest() in caplog.text
    assert "content_type=image/webp" in caplog.text


@pytest.mark.asyncio
async def test_cv_client_does_not_log_image_fingerprint_by_default(monkeypatch, caplog) -> None:
    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {
                "muac_estimate_mm": 125.0,
                "risk_band": "Normal",
                "confidence_score": 0.8,
                "quality_flags": {
                    "reference_detected": True,
                    "arm_detected": True,
                    "segmentation_ok": True,
                },
            }

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr(cv_client.httpx, "AsyncClient", lambda **_kwargs: FakeClient())
    monkeypatch.setattr(cv_client.settings, "CV_SERVICE_URL", "http://ml.test")
    monkeypatch.delenv("POSHANSCAN_IMAGE_DIAGNOSTICS", raising=False)

    with caplog.at_level(logging.INFO, logger="poshanscan-backend.cv"):
        await cv_client.run_cv_inference(image_bytes=b"private payload")

    assert "image_diagnostic" not in caplog.text
