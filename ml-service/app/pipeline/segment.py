"""Segment the upper-arm region around the MUAC measurement site."""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypedDict

import cv2
import numpy as np

from app.pipeline.pose_localize import axis_angle_difference_degrees, estimate_arm_axis_from_mask

# HSV skin ranges in OpenCV (H: 0–180). Covers typical light/medium skin;
# a second band catches reddish hues that wrap around H=0.
_SKIN_HSV_LOWER_1 = np.array([0, 20, 40], dtype=np.uint8)
_SKIN_HSV_UPPER_1 = np.array([25, 255, 255], dtype=np.uint8)
_SKIN_HSV_LOWER_2 = np.array([160, 20, 40], dtype=np.uint8)
_SKIN_HSV_UPPER_2 = np.array([179, 255, 255], dtype=np.uint8)

DEFAULT_CROP_SIZE_PX = 200
YOLO_DEFAULT_CONFIDENCE = 0.25
YOLO_DEFAULT_MASK_THRESHOLD = 0.5
YOLO_DEFAULT_MIN_MASK_COVERAGE = 0.005
YOLO_DEFAULT_MAX_MASK_COVERAGE = 0.95
YOLO_DEFAULT_IMAGE_SIZE = 416
logger = logging.getLogger("poshanscan-cv.segmentation")

_SEGMENTATION_LOCK = threading.Lock()
_SEGMENTATION_MODE = "legacy"
_YOLO_MODEL: Any = None
_YOLO_DEVICE: str | int = "cpu"
_MASK_PREDICTOR_CLASSES: dict[float, type] = {}


@dataclass(frozen=True)
class YoloSegmentationConfig:
    """Runtime thresholds for the loaded upper-arm segmenter."""

    confidence: float = YOLO_DEFAULT_CONFIDENCE
    mask_threshold: float = YOLO_DEFAULT_MASK_THRESHOLD
    min_mask_coverage: float = YOLO_DEFAULT_MIN_MASK_COVERAGE
    max_mask_coverage: float = YOLO_DEFAULT_MAX_MASK_COVERAGE
    image_size: int = YOLO_DEFAULT_IMAGE_SIZE


_YOLO_CONFIG = YoloSegmentationConfig()


class SegmentArmResult(TypedDict):
    """Result of :func:`segment_arm`. Keep this shape when swapping in YOLO."""

    mask: np.ndarray
    centroid_px: Optional[tuple[float, float]]
    crop_origin_px: tuple[int, int]
    crop_bgr: np.ndarray
    segmentation_method: str
    segmentation_confidence: float


def _empty_mask(crop_size_px: int) -> np.ndarray:
    return np.zeros((crop_size_px, crop_size_px), dtype=np.uint8)


def _empty_result(crop_size_px: int = DEFAULT_CROP_SIZE_PX) -> SegmentArmResult:
    return {
        "mask": _empty_mask(crop_size_px),
        "centroid_px": None,
        "crop_origin_px": (0, 0),
        "crop_bgr": np.zeros((crop_size_px, crop_size_px, 3), dtype=np.uint8),
        "segmentation_method": "legacy",
        "segmentation_confidence": 0.0,
    }


def _load_yolo_model(weights_path: Path) -> Any:
    from ultralytics import YOLO

    return YOLO(str(weights_path))


def _model_names(model: Any) -> dict[int, str]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def _configured_yolo_values(
    confidence_threshold: Optional[float],
    mask_threshold: Optional[float],
    min_mask_coverage: Optional[float],
    max_mask_coverage: Optional[float],
    image_size: Optional[int],
) -> YoloSegmentationConfig:
    def float_value(argument: Optional[float], env_name: str, default: float) -> float:
        raw: Any = argument if argument is not None else os.getenv(env_name)
        try:
            value = default if raw is None else float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{env_name} must be a number.") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{env_name} must be finite and between 0 and 1.")
        return value

    def coverage_value(argument: Optional[float], env_name: str, default: float) -> float:
        raw: Any = argument if argument is not None else os.getenv(env_name)
        try:
            value = default if raw is None else float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{env_name} must be a number.") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{env_name} must be finite and between 0 and 1.")
        return value

    raw_size: Any = image_size if image_size is not None else os.getenv("POSHANSCAN_YOLO_IMAGE_SIZE")
    try:
        resolved_size = YOLO_DEFAULT_IMAGE_SIZE if raw_size is None else int(raw_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("POSHANSCAN_YOLO_IMAGE_SIZE must be an integer.") from exc
    if resolved_size < 64:
        raise ValueError("POSHANSCAN_YOLO_IMAGE_SIZE must be at least 64.")

    config = YoloSegmentationConfig(
        confidence=float_value(confidence_threshold, "POSHANSCAN_YOLO_CONFIDENCE", YOLO_DEFAULT_CONFIDENCE),
        mask_threshold=float_value(mask_threshold, "POSHANSCAN_YOLO_MASK_THRESHOLD", YOLO_DEFAULT_MASK_THRESHOLD),
        min_mask_coverage=coverage_value(
            min_mask_coverage, "POSHANSCAN_YOLO_MIN_MASK_COVERAGE", YOLO_DEFAULT_MIN_MASK_COVERAGE
        ),
        max_mask_coverage=coverage_value(
            max_mask_coverage, "POSHANSCAN_YOLO_MAX_MASK_COVERAGE", YOLO_DEFAULT_MAX_MASK_COVERAGE
        ),
        image_size=resolved_size,
    )
    if config.min_mask_coverage >= config.max_mask_coverage:
        raise ValueError("YOLO minimum mask coverage must be below maximum mask coverage.")
    return config


def configure_segmentation_model(
    mode: str = "legacy",
    weights_path: Optional[str | Path] = None,
    *,
    confidence_threshold: Optional[float] = None,
    mask_threshold: Optional[float] = None,
    min_mask_coverage: Optional[float] = None,
    max_mask_coverage: Optional[float] = None,
    image_size: Optional[int] = None,
) -> bool:
    """Load and validate a YOLO segmenter once at startup; failures select legacy.

    Thresholds may be passed directly or configured with ``POSHANSCAN_YOLO_*``
    environment variables. Ultralytics selects CUDA when available and CPU
    otherwise; this service does not assume a GPU exists.
    """
    global _SEGMENTATION_MODE, _YOLO_MODEL, _YOLO_CONFIG, _YOLO_DEVICE
    requested_mode = (mode or "legacy").strip().lower()
    with _SEGMENTATION_LOCK:
        _SEGMENTATION_MODE = "legacy"
        _YOLO_MODEL = None
        _YOLO_DEVICE = "cpu"
        if requested_mode == "legacy":
            _YOLO_CONFIG = YoloSegmentationConfig()
            logger.info("Arm segmentation mode: legacy HSV + GrabCut.")
            return True
        if requested_mode != "yolo":
            logger.warning("Unknown segmentation mode %r; using legacy HSV + GrabCut.", mode)
            return False

        path = (
            Path(weights_path)
            if weights_path
            else Path(__file__).resolve().parents[1] / "models" / "arm_segmentation" / "best.pt"
        )
        if not path.is_file():
            logger.warning("YOLO segmentation weights not found at %s; using legacy segmentation.", path)
            return False
        try:
            config = _configured_yolo_values(
                confidence_threshold,
                mask_threshold,
                min_mask_coverage,
                max_mask_coverage,
                image_size,
            )
            model = _load_yolo_model(path)
            task = getattr(model, "task", None)
            if str(task or "").lower() != "segment":
                raise ValueError(f"Expected a YOLO segmentation checkpoint, got task={task!r}.")
            names = _model_names(model)
            if names != {0: "upper_arm"}:
                raise ValueError(f"Expected class mapping {{0: 'upper_arm'}}, got {names!r}.")
            _mask_predictor_class(config.mask_threshold)
        except Exception as exc:
            logger.warning("Could not load YOLO segmentation model; using legacy segmentation: %s", exc)
            return False
        _YOLO_MODEL = model
        _YOLO_CONFIG = config
        _YOLO_DEVICE = _select_yolo_device()
        _SEGMENTATION_MODE = "yolo"
        logger.info(
            "Arm segmentation mode: YOLO using %s on %s (conf=%.3f mask=%.3f coverage=%.4f..%.4f imgsz=%d).",
            path,
            _YOLO_DEVICE,
            config.confidence,
            config.mask_threshold,
            config.min_mask_coverage,
            config.max_mask_coverage,
            config.image_size,
        )
        return True


def _select_yolo_device() -> str | int:
    """Prefer CUDA when installed and available; otherwise explicitly use CPU."""
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        logger.info("PyTorch CUDA check unavailable; using CPU for YOLO inference.")
        return "cpu"


def _mask_predictor_class(mask_threshold: float) -> type:
    """Build an Ultralytics predictor that applies the configured logit threshold.

    Ultralytics' stock segmentation predictor binarizes masks internally before
    returning ``Results``. Its ``masks.data`` therefore cannot support a real
    configurable mask threshold. This small adapter keeps Ultralytics' model
    preprocessing, NMS, boxes, and Results object, while applying the configured
    probability threshold to the post-NMS prototype logits before binarization.
    """
    threshold = float(mask_threshold)
    if threshold in _MASK_PREDICTOR_CLASSES:
        return _MASK_PREDICTOR_CLASSES[threshold]

    import torch
    from torch.nn import functional as torch_functional
    from ultralytics.engine.results import Results
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor
    from ultralytics.utils import ops

    def probability_logit(value: float) -> float:
        return (
            -math.inf if value <= 0.0
            else math.inf if value >= 1.0
            else math.log(value / (1.0 - value))
        )

    class ConfigurableMaskPredictor(SegmentationPredictor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._poshanscan_mask_threshold = threshold

        def construct_result(self, pred, img, orig_img, img_path, proto):
            if pred.shape[0] == 0:
                return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])

            pred = pred.clone()
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            height, width = orig_img.shape[:2]
            channels, mask_height, mask_width = proto.shape
            coefficients = pred[:, 6:] @ proto.float().view(channels, -1)
            threshold_logit = probability_logit(float(self._poshanscan_mask_threshold))
            gain = min(4 * mask_height / height, 4 * mask_width / width)
            padding = (4 * mask_height - max(round(height * gain), 1)) | (
                4 * mask_width - max(round(width * gain), 1)
            )
            upsample = 4 // math.gcd(4, padding | padding // 2)
            chunk_size = max(
                1,
                16_000_000 // (height * width + upsample * upsample * mask_height * mask_width),
            )
            binary_masks = coefficients.new_empty((len(coefficients), height, width), dtype=torch.uint8)
            for offset, chunk in zip(
                range(0, len(binary_masks), chunk_size),
                coefficients.view(1, -1, mask_height, mask_width).split(chunk_size, 1),
            ):
                if upsample > 1:
                    chunk = torch_functional.interpolate(chunk, scale_factor=upsample, mode="bilinear")
                chunk = ops.scale_masks(chunk, (height, width))[0]
                binary_masks[offset:offset + len(chunk)] = chunk > threshold_logit

            binary_masks = ops.crop_mask(binary_masks, pred[:, :4])
            keep = binary_masks.amax((-2, -1)) > 0
            results = Results(
                orig_img,
                path=img_path,
                names=self.model.names,
                boxes=pred[:, :6],
                masks=binary_masks,
            )
            if getattr(self, "_feats", None) is None and not keep.all():
                results = results[keep]
            return results

    ConfigurableMaskPredictor.__name__ = f"PoshanScanMaskThreshold{threshold:g}Predictor"
    _MASK_PREDICTOR_CLASSES[threshold] = ConfigurableMaskPredictor
    return ConfigurableMaskPredictor


def _update_predictor_mask_threshold(model: Any, threshold: float) -> None:
    """Apply a per-call threshold override to the already cached predictor."""
    predictor = getattr(model, "predictor", None)
    if predictor is not None and hasattr(predictor, "_poshanscan_mask_threshold"):
        predictor._poshanscan_mask_threshold = float(threshold)


def segmentation_mode() -> str:
    return _SEGMENTATION_MODE


def predict_yolo_image(
    image_bgr: np.ndarray,
    *,
    imgsz: int = YOLO_DEFAULT_IMAGE_SIZE,
    confidence_threshold: Optional[float] = None,
    mask_threshold: Optional[float] = None,
) -> dict[str, Any]:
    """Predict full-image upper-arm polygons using the model loaded at startup.

    This is the inspectable prediction path used by ``POST /segment`` and the
    CLI helper. It does not run the MUAC measurement pipeline.
    """
    if _SEGMENTATION_MODE != "yolo" or _YOLO_MODEL is None:
        raise RuntimeError("The YOLO segmentation model is not loaded.")
    image_bgr = _validated_bgr_image(image_bgr)
    if imgsz < 64:
        raise ValueError("imgsz must be at least 64.")
    config = _YOLO_CONFIG
    confidence = config.confidence if confidence_threshold is None else float(confidence_threshold)
    threshold = config.mask_threshold if mask_threshold is None else float(mask_threshold)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1.")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("mask_threshold must be between 0 and 1.")

    with _SEGMENTATION_LOCK:
        _update_predictor_mask_threshold(_YOLO_MODEL, threshold)
        predictions = _YOLO_MODEL.predict(
            source=image_bgr,
            imgsz=imgsz,
            conf=confidence,
            classes=[0],
            retina_masks=True,
            device=_YOLO_DEVICE,
            predictor=_mask_predictor_class(threshold),
            verbose=False,
        )

    height, width = image_bgr.shape[:2]
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    instances: list[dict[str, Any]] = []
    prediction = predictions[0] if predictions else None
    masks = getattr(prediction, "masks", None) if prediction is not None else None
    boxes = getattr(prediction, "boxes", None) if prediction is not None else None
    polygons = getattr(masks, "xy", None) if masks is not None else None
    if boxes is not None and polygons is not None:
        classes = _to_numpy(getattr(boxes, "cls", np.zeros(len(polygons)))).reshape(-1)
        confidences = _to_numpy(getattr(boxes, "conf", np.zeros(len(polygons)))).reshape(-1)
        box_values = _to_numpy(getattr(boxes, "xyxy", np.zeros((len(polygons), 4))))
        for index in range(min(len(polygons), len(classes), len(confidences), len(box_values))):
            if int(classes[index]) != 0:
                continue
            score = float(confidences[index])
            if not math.isfinite(score) or score < confidence:
                continue
            polygon = np.asarray(polygons[index], dtype=np.float64).reshape(-1, 2)
            if polygon.shape[0] < 3 or not np.isfinite(polygon).all():
                continue
            polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
            instance_mask = _instance_mask_on_image(masks, index, polygon, width, height, threshold)
            if instance_mask is None:
                logger.warning("YOLO returned an invalid upper_arm mask at instance %d.", index)
                continue
            coverage = float(np.count_nonzero(instance_mask) / instance_mask.size)
            if not config.min_mask_coverage <= coverage <= config.max_mask_coverage:
                logger.warning("YOLO mask coverage %.4f is implausible; ignoring instance %d.", coverage, index)
                continue
            bbox = np.asarray(box_values[index], dtype=np.float64).reshape(-1)
            if bbox.size < 4 or not np.isfinite(bbox[:4]).all():
                continue
            combined_mask = cv2.bitwise_or(combined_mask, instance_mask)
            instances.append({
                "class_id": 0,
                "class_name": "upper_arm",
                "confidence": score,
                "bbox_xyxy": [
                    float(np.clip(bbox[0], 0, width - 1)),
                    float(np.clip(bbox[1], 0, height - 1)),
                    float(np.clip(bbox[2], 0, width - 1)),
                    float(np.clip(bbox[3], 0, height - 1)),
                ],
                "polygon": [[float(x), float(y)] for x, y in polygon],
            })

    combined_coverage = float(np.count_nonzero(combined_mask) / combined_mask.size)
    if instances and not config.min_mask_coverage <= combined_coverage <= config.max_mask_coverage:
        logger.warning("Combined YOLO mask coverage %.4f is implausible; returning no detection.", combined_coverage)
        combined_mask.fill(0)
        instances.clear()

    visualization = image_bgr.copy()
    if np.any(combined_mask):
        tint = np.zeros_like(visualization)
        tint[:] = (40, 200, 40)
        foreground = combined_mask > 0
        visualization[foreground] = cv2.addWeighted(
            visualization[foreground], 0.45, tint[foreground], 0.55, 0
        )
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(visualization, contours, -1, (0, 255, 0), 2)
        for instance in instances:
            x1, y1, x2, y2 = [int(round(value)) for value in instance["bbox_xyxy"]]
            cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 180, 255), 2)
            cv2.putText(
                visualization,
                f"upper_arm {instance['confidence']:.2f}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 180, 255),
                2,
                cv2.LINE_AA,
            )
    return {
        "detected": bool(instances),
        "width": width,
        "height": height,
        "instances": instances,
        "mask": combined_mask,
        "visualization": visualization,
    }


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _validated_bgr_image(image: Any) -> np.ndarray:
    """Normalize a valid grayscale/BGR image to finite uint8 BGR pixels."""
    if image is None:
        raise ValueError("Expected a non-empty image.")
    try:
        array = np.asarray(image)
    except Exception as exc:
        raise ValueError("Expected an image array.") from exc
    if array.size == 0 or array.ndim not in {2, 3}:
        raise ValueError("Expected a non-empty grayscale or BGR image.")
    if array.ndim == 3 and array.shape[2] not in {1, 3}:
        raise ValueError("Expected a grayscale or three-channel BGR image.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("Expected a non-empty grayscale or BGR image.")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("Image pixels must be finite numeric values.")
    if array.dtype != np.uint8:
        if np.any(array < 0) or np.any(array > 255):
            raise ValueError("Image pixels must be in the range 0..255.")
        array = np.rint(array).astype(np.uint8)
    return _to_bgr(array)


def _instance_mask_on_image(
    masks: Any,
    index: int,
    polygon_xy: Optional[np.ndarray],
    width: int,
    height: int,
    threshold: float,
) -> Optional[np.ndarray]:
    """Threshold one predicted mask and return it on the original image grid."""
    raw_masks = getattr(masks, "data", None)
    if raw_masks is not None:
        try:
            data = _to_numpy(raw_masks)
            if data.ndim == 3:
                if index >= data.shape[0]:
                    return None
                values = np.asarray(data[index])
            elif data.ndim == 2 and index == 0:
                values = np.asarray(data)
            else:
                return None
            values = values.astype(np.float32, copy=False)
            if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
                return None
            maximum = float(values.max(initial=0.0))
            if maximum > 1.0:
                if maximum > 255.0:
                    return None
                values = values / 255.0
            if values.shape != (height, width):
                values = cv2.resize(values, (width, height), interpolation=cv2.INTER_LINEAR)
            return np.where(values >= threshold, 255, 0).astype(np.uint8)
        except (IndexError, TypeError, ValueError, cv2.error):
            return None

    if polygon_xy is None or polygon_xy.shape[0] < 3 or not np.isfinite(polygon_xy).all():
        return None
    polygon = np.asarray(polygon_xy, dtype=np.float64).copy()
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    output = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(output, [np.round(polygon).astype(np.int32)], 255)
    return output


def _crop_mask(mask: np.ndarray, origin: tuple[int, int], crop_size_px: int) -> np.ndarray:
    """Crop a full-image binary mask with the same zero padding as the image crop."""
    height, width = mask.shape[:2]
    x0, y0 = origin
    x1, y1 = x0 + crop_size_px, y0 + crop_size_px
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(width, x1), min(height, y1)
    output = np.zeros((crop_size_px, crop_size_px), dtype=np.uint8)
    if xb <= xa or yb <= ya:
        return output
    dx, dy = max(0, -x0), max(0, -y0)
    output[dy:dy + (yb - ya), dx:dx + (xb - xa)] = mask[ya:yb, xa:xb]
    return output


def _yolo_mask(
    image_bgr: np.ndarray,
    crop_origin_px: tuple[int, int],
    crop_size_px: int,
) -> Optional[tuple[np.ndarray, float]]:
    model = _YOLO_MODEL
    if model is None:
        return None
    try:
        config = _YOLO_CONFIG
        with _SEGMENTATION_LOCK:
            _update_predictor_mask_threshold(model, config.mask_threshold)
            predictions = model.predict(
                source=image_bgr,
                imgsz=config.image_size,
                conf=config.confidence,
                classes=[0],
                retina_masks=True,
                device=_YOLO_DEVICE,
                predictor=_mask_predictor_class(config.mask_threshold),
                verbose=False,
            )
        if not predictions:
            return None
        prediction = predictions[0]
        masks = getattr(prediction, "masks", None)
        boxes = getattr(prediction, "boxes", None)
        polygons = getattr(masks, "xy", None) if masks is not None else None
        if masks is None or boxes is None:
            return None
        raw_classes = getattr(boxes, "cls", None)
        raw_confidences = getattr(boxes, "conf", None)
        if raw_classes is None or raw_confidences is None:
            return None
        classes = _to_numpy(raw_classes).reshape(-1)
        confidences = _to_numpy(raw_confidences).reshape(-1)
        if polygons is None:
            polygons = [None] * min(len(classes), len(confidences))
        height, width = image_bgr.shape[:2]
        merged_mask = np.zeros((crop_size_px, crop_size_px), dtype=np.uint8)
        candidate_confidences: list[float] = []
        target_half_side = max(16, min(50, crop_size_px // 4))
        crop_center = crop_size_px // 2
        target_y0 = max(0, crop_center - target_half_side)
        target_y1 = min(crop_size_px, crop_center + target_half_side + 1)
        target_x0 = max(0, crop_center - target_half_side)
        target_x1 = min(crop_size_px, crop_center + target_half_side + 1)
        raw_mask_data = getattr(masks, "data", None)
        if raw_mask_data is not None:
            raw_mask_data = _to_numpy(raw_mask_data)
            mask_count = raw_mask_data.shape[0] if raw_mask_data.ndim == 3 else 1
        else:
            mask_count = len(polygons)
        for index in range(min(len(polygons), len(classes), len(confidences), mask_count)):
            confidence = float(confidences[index])
            if (
                int(classes[index]) != 0
                or not math.isfinite(confidence)
                or confidence < config.confidence
            ):
                continue
            polygon_value = polygons[index]
            polygon = None
            if polygon_value is not None:
                polygon = np.asarray(polygon_value, dtype=np.float64).reshape(-1, 2)
                if polygon.shape[0] < 3 or not np.isfinite(polygon).all():
                    polygon = None
            full_mask = _instance_mask_on_image(
                masks, index, polygon, width, height, config.mask_threshold
            )
            if full_mask is None or not np.any(full_mask):
                logger.warning("YOLO returned an empty or invalid upper_arm mask at instance %d.", index)
                continue
            binary = _crop_mask(full_mask, crop_origin_px, crop_size_px)
            # The marker can occlude the MUAC midpoint, so accept visible upper-arm
            # fragments close to the pose point and merge them before marker removal.
            if np.any(binary[target_y0:target_y1, target_x0:target_x1]):
                merged_mask = cv2.bitwise_or(merged_mask, binary)
                candidate_confidences.append(confidence)
        coverage = float(np.count_nonzero(merged_mask) / merged_mask.size)
        if not candidate_confidences or not config.min_mask_coverage <= coverage <= config.max_mask_coverage:
            if candidate_confidences:
                logger.warning("YOLO mask coverage %.4f is implausible; using legacy segmentation.", coverage)
            return None
        return merged_mask, max(candidate_confidences)
    except Exception as exc:
        logger.warning("YOLO segmentation inference failed; using legacy segmentation: %s", exc)
        return None


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


def _extract_crop(
    image: np.ndarray,
    midpoint_px: tuple[float, float],
    crop_size_px: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop ``crop_size_px`` square around midpoint; pad if it hits the border."""
    height, width = image.shape[:2]
    cx = int(round(float(midpoint_px[0])))
    cy = int(round(float(midpoint_px[1])))
    half = crop_size_px // 2
    x0 = cx - half
    y0 = cy - half
    x1 = x0 + crop_size_px
    y1 = y0 + crop_size_px

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - width)
    pad_bottom = max(0, y1 - height)

    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(width, x1), min(height, y1)
    crop = image[ya:yb, xa:xb]
    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
    if crop.shape[0] != crop_size_px or crop.shape[1] != crop_size_px:
        crop = cv2.resize(crop, (crop_size_px, crop_size_px), interpolation=cv2.INTER_NEAREST)
    return crop, (x0, y0)


def _skin_mask(crop_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    band_a = cv2.inRange(hsv, _SKIN_HSV_LOWER_1, _SKIN_HSV_UPPER_1)
    band_b = cv2.inRange(hsv, _SKIN_HSV_LOWER_2, _SKIN_HSV_UPPER_2)
    skin = cv2.bitwise_or(band_a, band_b)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel)
    return skin


def _grabcut_refine(crop_bgr: np.ndarray, skin: np.ndarray) -> np.ndarray:
    """Refine a skin-colour seed mask with GrabCut. Falls back to ``skin``."""
    if int(cv2.countNonZero(skin)) < 50:
        return skin

    gc_mask = np.full(skin.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[skin > 0] = cv2.GC_PR_FGD
    kernel = np.ones((5, 5), dtype=np.uint8)
    sure_fg = cv2.erode(skin, kernel, iterations=2)
    sure_bg = cv2.bitwise_not(cv2.dilate(skin, kernel, iterations=2))
    gc_mask[sure_bg > 0] = cv2.GC_BGD
    gc_mask[sure_fg > 0] = cv2.GC_FGD

    if not np.any(gc_mask == cv2.GC_FGD) and not np.any(gc_mask == cv2.GC_PR_FGD):
        return skin

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            crop_bgr,
            gc_mask,
            None,
            bgd_model,
            fgd_model,
            5,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return skin

    refined = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    if int(cv2.countNonZero(refined)) == 0:
        return skin
    return refined


def _mask_centroid(mask: np.ndarray) -> Optional[tuple[float, float]]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def segment_arm(
    image: np.ndarray,
    midpoint_px: tuple[float, float],
    crop_size_px: int = DEFAULT_CROP_SIZE_PX,
) -> SegmentArmResult:
    """Segment the arm in a crop centred on the MUAC midpoint.

    YOLO mode runs a startup-loaded model on the full image and crops its
    polygon mask around the MUAC midpoint. Missing or invalid predictions use
    the HSV skin-colour threshold + GrabCut fallback. Keep this result shape so
    estimate/classify do not need to know which segmenter produced the mask::

        segment_arm(image: np.ndarray, midpoint_px: tuple, crop_size_px: int = 200) -> dict
        # dict keys: mask (H=W=crop_size_px, uint8 0/255), centroid_px, crop_origin_px, crop_bgr

    Args:
        image: Full BGR (or grayscale) photo.
        midpoint_px: ``(x, y)`` MUAC site in full-image pixel coordinates.
        crop_size_px: Square crop side length.

    Returns:
        Segmentation dict. ``mask`` is always ``crop_size_px x crop_size_px``.
        ``centroid_px`` is in crop coordinates, or None if nothing was found.
    """
    if _is_empty_image(image) or midpoint_px is None or crop_size_px <= 0:
        return _empty_result(max(int(crop_size_px), 1) if crop_size_px else DEFAULT_CROP_SIZE_PX)

    try:
        bgr = _validated_bgr_image(image)
    except ValueError as exc:
        logger.warning("Invalid image supplied to arm segmentation: %s", exc)
        return _empty_result(max(int(crop_size_px), 1))
    crop_bgr, origin = _extract_crop(bgr, midpoint_px, crop_size_px)

    method = "legacy"
    confidence = 1.0
    yolo_result = (
        _yolo_mask(bgr, origin, crop_size_px)
        if _SEGMENTATION_MODE == "yolo"
        else None
    )
    if yolo_result is not None:
        mask, confidence = yolo_result
        method = "yolo"
    else:
        if _SEGMENTATION_MODE == "yolo":
            logger.warning("YOLO returned no usable arm mask; using legacy HSV + GrabCut for this image.")
            method = "legacy_fallback"
            confidence = 0.0
        skin = _skin_mask(crop_bgr)
        mask = _grabcut_refine(crop_bgr, skin)

    return {
        "mask": mask,
        "centroid_px": _mask_centroid(mask),
        "crop_origin_px": origin,
        "crop_bgr": crop_bgr,
        "segmentation_method": method,
        "segmentation_confidence": confidence,
    }


def measure_width_at_row(mask: np.ndarray, row: int) -> float:
    """Pixel width of the arm on a horizontal slice of the binary mask.

    Uses the longest contiguous foreground run so small holes/speckles do
    not inflate the width. Returns 0.0 if ``row`` is out of bounds or empty.
    """
    if mask is None or mask.size == 0:
        return 0.0
    if row < 0 or row >= mask.shape[0]:
        return 0.0
    line = (mask[row] > 0).astype(np.uint8)
    if int(line.sum()) == 0:
        return 0.0

    padded = np.concatenate(([0], line, [0]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    run_lengths = ends - starts
    return float(run_lengths.max()) if run_lengths.size else 0.0


def measure_width_perpendicular(
    mask: np.ndarray,
    center_px: tuple[float, float],
    arm_angle_degrees: float,
    bridge_internal_gaps: bool = False,
    search_radius_px: int = 0,
    min_width_px: float = 0.0,
    max_width_px: Optional[float] = None,
    diagnostics: Optional[list[dict[str, Any]]] = None,
    max_bridge_gap_px: Optional[float] = None,
) -> float:
    """Measure local arm cross-sections along lines normal to the arm axis.

    Sampling occurs in the original mask coordinates, avoiding canvas clipping
    and interpolation from a rotated mask. If a reliable elongated mask axis
    conflicts strongly with the requested angle, the measurement fails closed.
    Marker gaps may be bridged only between the closest foreground runs on
    opposite sides of the sample center.
    """
    if diagnostics is not None:
        diagnostics.clear()
    try:
        requested_angle = float(arm_angle_degrees)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if (
        mask is None
        or not isinstance(mask, np.ndarray)
        or mask.ndim != 2
        or mask.size == 0
        or not np.isfinite(requested_angle)
    ):
        return 0.0
    height, width = mask.shape[:2]
    try:
        cx, cy = float(center_px[0]), float(center_px[1])
    except (TypeError, ValueError, IndexError):
        return 0.0
    if not np.isfinite(cx) or not np.isfinite(cy):
        return 0.0
    cx = float(np.clip(cx, 0, width - 1))
    cy = float(np.clip(cy, 0, height - 1))
    angle = requested_angle % 180.0

    mask_axis = estimate_arm_axis_from_mask(
        mask,
        center_px=(cx, cy),
        min_foreground_px=80,
        min_eigenvalue_ratio=4.0,
    )
    mask_angle = mask_axis.get("angle_degrees")
    if mask_angle is not None:
        difference = axis_angle_difference_degrees(angle, float(mask_angle))
        if difference > 45.0:
            if diagnostics is not None:
                diagnostics.append({
                    "rejected": True,
                    "reason": "arm_axis_disagrees_with_segmented_mask",
                    "requested_axis_degrees": angle,
                    "mask_axis_degrees": float(mask_angle),
                    "axis_difference_degrees": difference,
                })
            return 0.0

    radians = math.radians(angle)
    axis = np.array([math.cos(radians), math.sin(radians)], dtype=np.float64)
    normal = np.array([-axis[1], axis[0]], dtype=np.float64)
    half_diagonal = float(math.hypot(width, height))
    ts = np.arange(-half_diagonal, half_diagonal + 0.25, 0.5, dtype=np.float64)
    step = 0.5
    bridge_limit = (
        max(0.0, float(max_bridge_gap_px))
        if max_bridge_gap_px is not None
        else max(12.0, float(min_width_px) * 2.0)
    )

    def sample_line(
        line_center: np.ndarray,
    ) -> tuple[float, Optional[tuple[float, float]], Optional[tuple[float, float]], bool, bool]:
        points = np.rint(line_center[None, :] + ts[:, None] * normal[None, :]).astype(np.int32)
        inside = (
            (points[:, 0] >= 0)
            & (points[:, 0] < width)
            & (points[:, 1] >= 0)
            & (points[:, 1] < height)
        )
        foreground = np.zeros(len(points), dtype=bool)
        foreground[inside] = mask[points[inside, 1], points[inside, 0]] > 0
        transitions = np.diff(np.pad(foreground.astype(np.int8), (1, 1)))
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        if starts.size == 0:
            return 0.0, None, None, False, False
        runs = [
            (float(ts[start]), float(ts[end - 1]), int(end - start))
            for start, end in zip(starts, ends)
        ]
        selected: Optional[tuple[float, float, int]] = None
        bridged = False
        containing = [run for run in runs if run[0] - step <= 0.0 <= run[1] + step]
        if containing:
            selected = max(containing, key=lambda run: run[2])
        elif bridge_internal_gaps:
            negative = [run for run in runs if run[1] < 0.0]
            positive = [run for run in runs if run[0] > 0.0]
            if negative and positive:
                left = max(negative, key=lambda run: run[1])
                right = min(positive, key=lambda run: run[0])
                gap = right[0] - left[1] - step
                if gap <= bridge_limit:
                    selected = (left[0], right[1], left[2] + right[2])
                    bridged = True
        if selected is None:
            selected = max(runs, key=lambda run: run[2])

        start_t, end_t, _run_pixels = selected
        start = line_center + start_t * normal
        end = line_center + end_t * normal
        start_pixel = np.rint(start).astype(int)
        end_pixel = np.rint(end).astype(int)
        measured_width = float(np.linalg.norm(end_pixel - start_pixel) + 1.0)
        touches_boundary = bool(
            start_pixel[0] <= 0 or start_pixel[0] >= width - 1
            or end_pixel[0] <= 0 or end_pixel[0] >= width - 1
            or start_pixel[1] <= 0 or start_pixel[1] >= height - 1
            or end_pixel[1] <= 0 or end_pixel[1] >= height - 1
        )
        return (
            measured_width,
            (float(start[0]), float(start[1])),
            (float(end[0]), float(end[1])),
            bridged,
            touches_boundary,
        )

    search = max(0, int(search_radius_px))
    offsets = range(-search, search + 1) if search else (0,)
    accepted: list[float] = []
    for offset in offsets:
        line_center = np.array([cx, cy], dtype=np.float64) + float(offset) * axis
        measured_width, start, end, bridged, touches_boundary = sample_line(line_center)
        valid = (
            measured_width >= float(min_width_px)
            and measured_width > 0.0
            and not touches_boundary
            and (max_width_px is None or measured_width <= float(max_width_px))
        )
        if diagnostics is not None:
            diagnostics.append({
                "offset_along_axis_px": float(offset),
                "center_px": (float(line_center[0]), float(line_center[1])),
                "start_px": start,
                "end_px": end,
                "width_px": float(measured_width),
                "bridged_gap": bridged,
                "touches_mask_boundary": touches_boundary,
                "reason": "cross_section_touches_mask_boundary" if touches_boundary else None,
                "valid": valid,
            })
        if valid:
            accepted.append(float(measured_width))
    valid_widths = accepted
    return float(np.median(valid_widths)) if valid_widths else 0.0


def fill_polygon_on_mask(
    mask: np.ndarray,
    polygon_xy: np.ndarray,
    origin_px: tuple[int, int],
) -> np.ndarray:
    """Paint a full-image polygon onto a crop mask (e.g. the ArUco sticker)."""
    if mask is None or polygon_xy is None:
        return mask
    pts = np.asarray(polygon_xy, dtype=np.float32).reshape(-1, 2)
    ox, oy = origin_px
    local = np.round(pts - np.array([ox, oy], dtype=np.float32)).astype(np.int32)
    filled = mask.copy()
    cv2.fillConvexPoly(filled, local, 255)
    return filled


def remove_polygon_from_mask(
    mask: np.ndarray,
    polygon_xy: np.ndarray,
    origin_px: tuple[int, int],
) -> np.ndarray:
    """Remove a reference marker polygon so it cannot inflate arm width."""
    if mask is None or polygon_xy is None:
        return mask
    pts = np.asarray(polygon_xy, dtype=np.float32).reshape(-1, 2)
    ox, oy = origin_px
    local = np.round(pts - np.array([ox, oy], dtype=np.float32)).astype(np.int32)
    cleaned = mask.copy()
    cv2.fillConvexPoly(cleaned, local, 0)
    return cleaned
