"""FastAPI entrypoint for the PoshanScan computer-vision microservice."""

from __future__ import annotations

import logging
import os
import base64
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from app.pipeline.estimate import CalibrationCorrector
from app.pipeline.run import run_muac_pipeline, set_calibrator
from app.pipeline.classify import AgeCategory
from app.pipeline.segment import (
    configure_segmentation_model,
    predict_yolo_image,
    segmentation_mode,
)
from app.schemas import (
    PIPELINE_VERSION,
    ErrorResponse,
    HealthResponse,
    InferResponse,
    QualityFlags,
    SegmentationResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("poshanscan-cv")
MAX_SEGMENT_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_SEGMENT_IMAGE_PIXELS = 20_000_000


@asynccontextmanager
async def _lifespan(_: FastAPI):
    enabled = _env_enabled(os.getenv("POSHANSCAN_ENABLE_SYNTHETIC_CALIBRATION"))
    model_path = os.getenv("POSHANSCAN_SYNTHETIC_CALIBRATION_MODEL")
    configure_synthetic_calibration(enabled, model_path)
    configure_segmentation_model(
        os.getenv("POSHANSCAN_SEGMENTATION_MODEL", "yolo"),
        os.getenv("POSHANSCAN_SEGMENTATION_WEIGHTS"),
    )
    try:
        yield
    finally:
        set_calibrator(None)
        configure_segmentation_model("legacy")


app = FastAPI(
    title="PoshanScan CV Service",
    description=(
        "Estimates mid-upper arm circumference (MUAC) from a photo of a child's "
        "upper arm taken next to a reference marker, for malnutrition screening."
    ),
    version=PIPELINE_VERSION,
    lifespan=_lifespan,
)


def _env_enabled(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def configure_synthetic_calibration(
    enabled: bool,
    model_path: Optional[str | Path] = None,
) -> bool:
    """Load the synthetic demo corrector only when explicitly enabled.

    Missing, incompatible, or unreadable artifacts leave inference on the
    geometric pass-through corrector instead of preventing service startup.
    """
    set_calibrator(None)
    if not enabled:
        logger.info("Synthetic calibration is disabled; using geometric baseline.")
        return False

    path = (
        Path(model_path)
        if model_path
        else Path(__file__).resolve().parent / "models" / "calibration_synthetic.joblib"
    )
    try:
        corrector = CalibrationCorrector.load(path)
        set_calibrator(corrector)
    except Exception as exc:
        logger.warning("Synthetic calibration unavailable; using geometric baseline: %s", exc)
        return False

    logger.warning(
        "Synthetic calibration demo enabled from %s. Do not use this synthetic model for real screening.",
        path,
    )
    return True


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)




def _error(status_code: int, error: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error=error, message=message).model_dump(),
    )


def _decode_image(payload: bytes) -> np.ndarray | None:
    if not payload:
        return None
    array = np.frombuffer(payload, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send browsers to the Swagger UI."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return a simple liveness check."""
    logger.info("health: service=liveness status=ok")
    return HealthResponse(status="ok")


@app.post(
    "/infer",
    response_model=InferResponse,
    responses={
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def infer(
    image: UploadFile = File(..., description="Photo of the upper arm with a reference marker."),
    age_category: Optional[str] = Form(None, description="Age category: 'child_6_59m' (WHO 6-59 month) or 'adult' (adult/maternal screening). REQUIRED."),
    reference_type: str = Form("aruco", description="Reference object type (currently only aruco)."),
    reference_size_mm: float = Form(50.0, description="Printed marker edge length in millimetres."),
) -> InferResponse | JSONResponse:
    """Run the MUAC pipeline on an uploaded image."""
    logger.info(
        "infer: started category=%s reference_mm=%.1f",
        age_category,
        reference_size_mm,
    )
    try:
        # Validate age_category before any expensive work — never infer or default.
        _age_category_err = _error(
            422,
            "age_category_required",
            "age_category must be specified as 'child_6_59m' or 'adult'.",
        )
        if not age_category:
            return _age_category_err
        try:
            validated_age_category = AgeCategory(age_category)
        except (ValueError, KeyError):
            return _age_category_err

        payload = await image.read()
        bgr = _decode_image(payload)
        if bgr is None:
            return _error(
                422,
                "invalid_image",
                "Could not decode the uploaded file as an image.",
            )

        if (reference_type or "aruco").strip().lower() != "aruco":
            logger.info(
                "reference_type=%s is not implemented; falling back to aruco",
                reference_type,
            )

        result = run_muac_pipeline(bgr, float(reference_size_mm), validated_age_category.value)
        logger.info(
            "infer: result=%s",
            {
                "ok": result.get("ok"),
                "error": result.get("error"),
                "muac_mm": result.get("muac_estimate_mm"),
                "risk_band": result.get("risk_band"),
            },
        )
        if not result.get("ok"):
            return _error(
                422,
                str(result.get("error") or "pipeline_failed"),
                str(result.get("message") or "Pipeline failed."),
            )

        return InferResponse(
            muac_estimate_mm=float(result["muac_estimate_mm"]),
            risk_band=str(result["risk_band"]),
            age_category=str(result["age_category"]),
            confidence_score=float(result["confidence_score"]),
            quality_flags=QualityFlags(
                reference_detected=True,
                arm_detected=True,
                segmentation_ok=bool(result.get("segmentation_ok")),
            ),
            pipeline_version=PIPELINE_VERSION,
        )
    except Exception:
        logger.exception("Unexpected error during /infer")
        return _error(
            500,
            "internal_error",
            "An unexpected error occurred while processing the image.",
        )


@app.post(
    "/segment",
    response_model=SegmentationResponse,
    responses={
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Preview upper-arm segmentation on a new image",
)
async def segment_image(
    image: UploadFile = File(..., description="A new image to segment."),
    confidence: float = Query(0.25, ge=0.0, le=1.0),
    imgsz: int = Query(640, ge=64, le=2048),
) -> SegmentationResponse | JSONResponse:
    """Return a predicted polygon, mask and overlay without estimating MUAC."""
    payload = await image.read(MAX_SEGMENT_UPLOAD_BYTES + 1)
    if len(payload) > MAX_SEGMENT_UPLOAD_BYTES:
        return _error(422, "image_too_large", "Image uploads must be 15 MB or smaller.")
    bgr = _decode_image(payload)
    if bgr is None:
        return _error(422, "invalid_image", "Could not decode the uploaded file as an image.")
    if bgr.shape[0] * bgr.shape[1] > MAX_SEGMENT_IMAGE_PIXELS:
        return _error(422, "image_dimensions_too_large", "Decoded images must contain 20 megapixels or fewer.")
    if segmentation_mode() != "yolo":
        return _error(
            503,
            "segmentation_model_unavailable",
            "The trained YOLO segmentation checkpoint is not loaded. Configure a trained model before using /segment.",
        )

    try:
        prediction = await run_in_threadpool(
            predict_yolo_image,
            bgr,
            imgsz=imgsz,
            confidence_threshold=confidence,
        )
        ok_mask, mask_png = cv2.imencode(".png", prediction["mask"])
        ok_preview, preview_png = cv2.imencode(".png", prediction["visualization"])
        if not ok_mask or not ok_preview:
            return _error(500, "preview_encoding_failed", "Could not encode the segmentation preview.")
        return SegmentationResponse(
            detected=bool(prediction["detected"]),
            width=int(prediction["width"]),
            height=int(prediction["height"]),
            instances=prediction["instances"],
            mask_png_base64=base64.b64encode(mask_png.tobytes()).decode("ascii"),
            visualization_png_base64=base64.b64encode(preview_png.tobytes()).decode("ascii"),
        )
    except ValueError as exc:
        return _error(422, "invalid_prediction_request", str(exc))
    except Exception:
        logger.exception("Unexpected error during /segment")
        return _error(500, "segmentation_failed", "The segmentation model could not process this image.")
