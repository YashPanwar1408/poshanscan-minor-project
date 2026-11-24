import logging
import asyncio
import hashlib
import mimetypes
import os
from typing import Optional
import httpx
from fastapi import HTTPException, status

from app.core.config import settings
from app.schemas.cv import CVInferResponse, CVQualityFlags

logger = logging.getLogger("poshanscan-backend.cv")


class CVInferenceError(Exception):
    def __init__(self, status_code: int, error: str, message: str):
        self.status_code = status_code
        self.error = error
        self.message = message
        super().__init__(message)


async def run_cv_inference(
    image_bytes: bytes,
    filename: str = "scan.jpg",
    reference_type: str = "aruco",
    reference_size_mm: float = 50.0,
    age_category: str = "child_6_59m",
    content_type: Optional[str] = None,
) -> CVInferResponse:
    """
    Call the production Computer Vision microservice (/infer).
    Raises CVInferenceError on processing failures (e.g., marker not found).
    """
    if not image_bytes or len(image_bytes) == 0:
        raise CVInferenceError(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error="invalid_image",
            message="No image data provided.",
        )

    # Normalize age category: "adult" or "child_6_59m"
    norm_age = "adult" if str(age_category).strip().lower() == "adult" else "child_6_59m"

    # Live HTTP Request to Yash's Service
    url = f"{settings.CV_SERVICE_URL.rstrip('/')}/infer"
    logger.info("Calling remote CV service at %s with age_category=%s", url, norm_age)

    upload_content_type = (
        content_type
        or mimetypes.guess_type(filename)[0]
        or "application/octet-stream"
    )
    if os.getenv("POSHANSCAN_IMAGE_DIAGNOSTICS", "").strip().lower() in {"1", "true", "yes", "on"}:
        logger.info(
            "image_diagnostic stage=backend_cv_forward sha256=%s bytes=%d content_type=%s",
            hashlib.sha256(image_bytes).hexdigest(),
            len(image_bytes),
            upload_content_type,
        )

    files = {
        "image": (filename, image_bytes, upload_content_type),
    }
    data = {
        "reference_type": reference_type,
        "reference_size_mm": str(reference_size_mm),
        "age_category": norm_age,
    }

    try:
        # Render's free ML instance can return a transient gateway response while
        # waking. Retry only gateway/service failures; validation errors are final.
        async with httpx.AsyncClient(timeout=httpx.Timeout(85.0, connect=20.0)) as client:
            response = None
            for attempt in range(3):
                response = await client.post(url, files=files, data=data)
                if response.status_code not in (502, 503, 504) or attempt == 2:
                    break
                await asyncio.sleep(2.0)
    except httpx.RequestError as exc:
        logger.error("Failed to connect to CV service: %s", exc)
        raise CVInferenceError(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error="cv_service_unavailable",
            message="Could not connect to the AI/CV estimation service.",
        )

    if response.status_code == 200:
        payload = response.json()
        payload.setdefault("age_category", norm_age)
        return CVInferResponse(**payload)

    # Error handling from Yash's API (422 / 500)
    try:
        err_payload = response.json()
        if isinstance(err_payload.get("detail"), list):
            message = "; ".join(
                str(item.get("msg", item)) if isinstance(item, dict) else str(item)
                for item in err_payload["detail"]
            )
        else:
            message = err_payload.get("message", "Error from CV service.")
        error_code = err_payload.get("error", "cv_error")
    except (ValueError, TypeError, AttributeError):
        error_code = "cv_error"
        message = response.text or "Error from CV service."

    raise CVInferenceError(
        status_code=response.status_code,
        error=error_code,
        message=message,
    )
