"""Run reproducible per-image MUAC geometry diagnostics on a photo folder.

Accuracy metrics are calculated only when tape references are supplied and
the operator explicitly confirms the inputs are real photographs with manual
measurements. This command does not train the model or edit source images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))

from app.pipeline.geometry_debug import draw_measurement_geometry_debug  # noqa: E402
from app.pipeline.run import MUAC_RANGES_MM, run_muac_pipeline, set_calibrator  # noqa: E402
from app.pipeline.segment import configure_segmentation_model, segmentation_mode  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_ERROR_TOLERANCES_MM = (5.0, 10.0)
CSV_FIELDS = [
    "image_id",
    "image_path",
    "image_sha256",
    "marker_detected",
    "marker_id",
    "marker_scale_px_per_mm",
    "segmentation_status",
    "segmentation_method",
    "segmentation_confidence",
    "pose_available",
    "pose_source",
    "axis_method",
    "axis_degrees",
    "axis_confidence",
    "axis_agreement_degrees",
    "accepted_samples",
    "rejected_samples",
    "sampled_width_px",
    "measurement_valid",
    "estimated_muac_mm",
    "reference_muac_mm",
    "error_mm",
    "rejection_reason",
    "processing_error",
    "overlay_path",
    "diagnostics_path",
]


def discover_images(image_dir: Path, output_dir: Optional[Path] = None, recursive: bool = True) -> list[Path]:
    """Return supported image files in stable relative-path order."""
    root = image_dir.resolve()
    if not root.is_dir():
        raise ValueError(f"Image directory does not exist: {root}")
    excluded = output_dir.resolve() if output_dir is not None else None
    iterator: Iterable[Path] = root.rglob("*") if recursive else root.glob("*")
    images: list[Path] = []
    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        resolved = path.resolve()
        if excluded is not None:
            try:
                resolved.relative_to(excluded)
                continue
            except ValueError:
                pass
        images.append(resolved)
    return sorted(images, key=lambda path: path.relative_to(root).as_posix().casefold())


def _parse_reference_measurement(value: str, row_number: int) -> float:
    try:
        measurement = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Reference CSV row {row_number}: reference_muac_mm must be numeric.") from exc
    if not math.isfinite(measurement) or measurement <= 0:
        raise ValueError(f"Reference CSV row {row_number}: reference_muac_mm must be finite and positive.")
    return measurement


def load_tape_references(path: Optional[Path]) -> dict[str, float]:
    """Load CSV columns image_id,reference_muac_mm; no values are inferred."""
    if path is None:
        return {}
    if not path.is_file():
        raise ValueError(f"Reference CSV does not exist: {path}")
    references: dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"image_id", "reference_muac_mm"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Reference CSV must include image_id and reference_muac_mm headers.")
        for row_number, row in enumerate(reader, start=2):
            image_id = str(row.get("image_id") or "").strip().replace("\\", "/")
            if not image_id:
                raise ValueError(f"Reference CSV row {row_number}: image_id cannot be empty.")
            if image_id in references:
                raise ValueError(f"Reference CSV contains duplicate image_id {image_id!r}.")
            references[image_id] = _parse_reference_measurement(row.get("reference_muac_mm", ""), row_number)
    return references


def _match_references(
    image_paths: list[Path],
    image_root: Path,
    references: dict[str, float],
) -> dict[Path, float]:
    """Match reference IDs to relative paths, allowing unique basename IDs."""
    basename_counts: dict[str, int] = {}
    for path in image_paths:
        basename_counts[path.name] = basename_counts.get(path.name, 0) + 1
    matched: dict[Path, float] = {}
    for image_path in image_paths:
        relative_id = image_path.relative_to(image_root.resolve()).as_posix()
        if relative_id in references:
            matched[image_path] = references[relative_id]
        elif basename_counts[image_path.name] == 1 and image_path.name in references:
            matched[image_path] = references[image_path.name]
    return matched


def paired_error_metrics(
    estimates: list[float],
    references: list[float],
    tolerances_mm: tuple[float, ...] = DEFAULT_ERROR_TOLERANCES_MM,
) -> dict[str, Any]:
    """Calculate descriptive errors only for finite paired measurements."""
    if len(estimates) != len(references):
        raise ValueError("Estimates and references must have equal lengths.")
    if not tolerances_mm or any(not math.isfinite(value) or value <= 0 for value in tolerances_mm):
        raise ValueError("Every error tolerance must be a finite positive number of millimetres.")
    pairs = [
        (float(estimate), float(reference))
        for estimate, reference in zip(estimates, references)
        if math.isfinite(float(estimate)) and math.isfinite(float(reference))
    ]
    if not pairs:
        return {"paired_valid_count": 0, "mae_mm": None, "rmse_mm": None, "bias_mm": None, "within_tolerance_percent": {}}
    errors = np.asarray([estimate - reference for estimate, reference in pairs], dtype=np.float64)
    return {
        "paired_valid_count": int(errors.size),
        "mae_mm": float(np.mean(np.abs(errors))),
        "rmse_mm": float(np.sqrt(np.mean(np.square(errors)))),
        "bias_mm": float(np.mean(errors)),
        "within_tolerance_percent": {
            f"{tolerance:g}mm": float(np.mean(np.abs(errors) <= tolerance) * 100.0)
            for tolerance in tolerances_mm
        },
    }


def _axis_confidence_description() -> str:
    return (
        "Pose axis confidence is the MediaPipe shoulder/elbow visibility proxy. "
        "Mask axis confidence is the uncalibrated anisotropy proxy 1 - 1/eigenvalue_ratio. "
        "Neither value is a probability of correctness."
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _make_row(
    image_path: Path,
    relative_id: str,
    result: dict[str, Any],
    diagnostics: dict[str, Any],
    image_sha256: Optional[str],
    reference_muac_mm: Optional[float],
    overlay_path: Optional[str],
    processing_error: Optional[str] = None,
    diagnostics_path: Optional[str] = None,
) -> dict[str, Any]:
    measurement_valid = bool(result.get("ok") and result.get("measurement_valid", True))
    estimated_muac = result.get("muac_estimate_mm") if measurement_valid else None
    if estimated_muac is not None:
        estimated_muac = float(estimated_muac)
    error = (
        float(estimated_muac) - float(reference_muac_mm)
        if estimated_muac is not None and reference_muac_mm is not None
        else None
    )
    return {
        "image_id": relative_id,
        "image_path": str(image_path),
        "image_sha256": image_sha256,
        "marker_detected": bool(diagnostics.get("marker_detected", False)),
        "marker_id": diagnostics.get("marker_id"),
        "marker_scale_px_per_mm": diagnostics.get("marker_scale_px_per_mm"),
        "segmentation_status": diagnostics.get("segmentation_status", "not_run"),
        "segmentation_method": diagnostics.get("segmentation_method"),
        "segmentation_confidence": diagnostics.get("segmentation_confidence"),
        "pose_available": bool(diagnostics.get("pose_available", False)),
        "pose_source": diagnostics.get("pose_source"),
        "axis_method": diagnostics.get("axis_method"),
        "axis_degrees": diagnostics.get("axis_degrees"),
        "axis_confidence": diagnostics.get("axis_confidence"),
        "axis_agreement_degrees": diagnostics.get("axis_agreement_degrees"),
        "accepted_samples": int(diagnostics.get("accepted_samples", 0)),
        "rejected_samples": int(diagnostics.get("rejected_samples", 0)),
        "sampled_width_px": diagnostics.get("sampled_width_px"),
        "measurement_valid": measurement_valid,
        "estimated_muac_mm": estimated_muac,
        "reference_muac_mm": reference_muac_mm,
        "error_mm": error,
        "rejection_reason": diagnostics.get("rejection_reason") or result.get("error"),
        "processing_error": processing_error,
        "overlay_path": overlay_path,
        "diagnostics_path": diagnostics_path,
    }


def _write_overlay(
    image: np.ndarray,
    diagnostics: dict[str, Any],
    result: dict[str, Any],
    output_path: Path,
) -> None:
    mask = diagnostics.get("segmentation_mask")
    origin = diagnostics.get("crop_origin_px", (0, 0))
    center = diagnostics.get("measurement_center_crop_px")
    overlay = draw_measurement_geometry_debug(
        image=image,
        marker_corners=(
            np.asarray(diagnostics["marker_corners_xy"], dtype=np.float32)
            if diagnostics.get("marker_corners_xy") is not None
            else None
        ),
        pose=diagnostics.get("pose"),
        crop_origin_px=origin,
        mask=mask,
        measurement_center_px=center,
        arm_axis_degrees=diagnostics.get("axis_degrees"),
        axis_source=str(diagnostics.get("axis_method") or "unavailable"),
        width_px=float(diagnostics.get("sampled_width_px") or 0.0),
        width_diagnostics=diagnostics.get("width_samples", []),
        mask_axis_degrees=diagnostics.get("mask_axis_degrees"),
        mask_axis_ratio=diagnostics.get("mask_axis_eigenvalue_ratio"),
        axis_agreement_degrees=diagnostics.get("axis_agreement_degrees"),
        segmentation_status=str(diagnostics.get("segmentation_status") or "not_run"),
        measurement_valid=bool(result.get("ok") and result.get("measurement_valid", True)),
        rejection_reason=diagnostics.get("rejection_reason") or result.get("error"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        raise OSError(f"Could not write overlay: {output_path}")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(value.tolist() if isinstance(value, np.ndarray) else value, ensure_ascii=False)
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_report(
    output_dir: Path,
    rows: list[dict[str, Any]],
    metrics: Optional[dict[str, Any]],
    reference_csv: Optional[Path],
    real_photos_confirmed: bool,
    model_status: str,
    tolerances: tuple[float, ...],
    run_configuration: dict[str, Any],
    reference_count: int,
    matched_reference_count: int,
    unmatched_reference_ids: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "measurements.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})

    metrics_payload = {
        "image_count": len(rows),
        "valid_measurement_count": sum(bool(row["measurement_valid"]) for row in rows),
        "invalid_measurement_count": sum(not bool(row["measurement_valid"]) for row in rows),
        "manual_reference_csv": str(reference_csv.resolve()) if reference_csv else None,
        "real_photographs_confirmed_by_operator": real_photos_confirmed,
        "segmentation_model_status": model_status,
        "error_tolerances_mm": list(tolerances),
        "reference_row_count": reference_count,
        "reference_ids_matched_to_images": matched_reference_count,
        "unmatched_reference_ids": unmatched_reference_ids,
        "axis_confidence_definition": _axis_confidence_description(),
        "run_configuration": run_configuration,
        "paired_error_metrics": metrics,
        "accuracy_metrics_calculated": metrics is not None and metrics.get("paired_valid_count", 0) > 0,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, allow_nan=False), encoding="utf-8")

    if not real_photos_confirmed:
        metrics_note = "Not calculated: run again with `--confirm-real-photographs` after confirming the folder contains real photographs and any tape references are manual measurements."
    elif reference_csv is None:
        metrics_note = "Not calculated: no tape-reference CSV was supplied."
    elif metrics is None or metrics.get("paired_valid_count", 0) == 0:
        metrics_note = "Not calculated: no valid paired estimate/reference measurements were available."
    else:
        tolerance_text = ", ".join(f"within ±{value:g} mm: {metrics['within_tolerance_percent'][f'{value:g}mm']:.1f}%" for value in tolerances)
        metrics_note = (
            f"Paired valid images: {metrics['paired_valid_count']}; MAE {metrics['mae_mm']:.3f} mm; "
            f"RMSE {metrics['rmse_mm']:.3f} mm; bias {metrics['bias_mm']:+.3f} mm; {tolerance_text}."
        )
    report = [
        "# Real-image MUAC geometry validation report",
        "",
        "This report contains per-image diagnostics. It is not a clinical validation or reliability claim.",
        "",
        f"- Images processed: {len(rows)}",
        f"- Valid measurements: {metrics_payload['valid_measurement_count']}",
        f"- Invalid/rejected measurements: {metrics_payload['invalid_measurement_count']}",
        f"- YOLO model state: {model_status}",
        f"- Operator confirmed real photographs: {real_photos_confirmed}",
        f"- Manual reference rows: {reference_count}; matched to input images: {matched_reference_count}; unmatched IDs: {len(unmatched_reference_ids)}",
        f"- Measurement validity bounds: child {MUAC_RANGES_MM['child_6_59m'][0]:g}–{MUAC_RANGES_MM['child_6_59m'][1]:g} mm; adult {MUAC_RANGES_MM['adult'][0]:g}–{MUAC_RANGES_MM['adult'][1]:g} mm.",
        f"- Error metrics: {metrics_note}",
        "",
        "## Outputs",
        "",
        "- `measurements.csv`: one row per image, including quality states, axis and sampling diagnostics, and manual reference comparison when paired.",
        "- `metrics.json`: counts, confidence definition, run setup, and paired-error metrics when permitted and available.",
        "- `overlays/`: original-resolution photos with segmentation, marker, pose, axis, sampling center, accepted/rejected lines, final width, and validity state.",
        "",
        "## Interpretation limits",
        "",
        "Axis confidence is an uncalibrated visibility or mask-shape proxy, not a probability of correctness. Accuracy metrics include only images with a valid estimate and a manually supplied tape reference. Invalid estimates remain visible in the CSV and are not silently converted or included in the paired accuracy metrics.",
        "",
        "The script preserves the existing MUAC formula, ArUco px/mm calibration, clinical cutoffs, and validity bounds. No synthetic images or generated tape values should be used to claim real-world validation.",
    ]
    (output_dir / "validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def run_folder_validation(
    image_dir: Path,
    output_dir: Path,
    references_path: Optional[Path] = None,
    age_category: str = "child_6_59m",
    reference_size_mm: float = 50.0,
    recursive: bool = True,
    real_photos_confirmed: bool = False,
    tolerances_mm: tuple[float, ...] = DEFAULT_ERROR_TOLERANCES_MM,
    require_yolo: bool = True,
    weights_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Process a folder and write overlays, CSV, JSON metrics, and Markdown report."""
    root = image_dir.resolve()
    out = output_dir.resolve()
    images = discover_images(root, out, recursive)
    references = load_tape_references(references_path)
    matched_references = _match_references(images, root, references)
    image_ids = {image.relative_to(root).as_posix() for image in images}
    basename_counts: dict[str, int] = {}
    for image in images:
        basename_counts[image.name] = basename_counts.get(image.name, 0) + 1
    matched_reference_ids = {
        reference_id
        for reference_id in references
        if reference_id in image_ids
        or (basename_counts.get(reference_id, 0) == 1 and any(image.name == reference_id for image in images))
    }
    unmatched_reference_ids = sorted(set(references) - matched_reference_ids)

    configured = configure_segmentation_model("yolo", str(weights_path) if weights_path else None)
    mode = segmentation_mode()
    model_status = "YOLO loaded" if configured and mode == "yolo" else f"YOLO unavailable; active mode is {mode}"
    if require_yolo and mode != "yolo":
        raise RuntimeError(
            f"{model_status}. Install ml-service/requirements-segmentation.txt and verify the checkpoint, "
            "or pass --allow-legacy-fallback to run diagnostics without YOLO."
        )
    # Keep local real-image validation on the same geometric baseline; the
    # packaged synthetic calibration demo is intentionally not loaded here.
    set_calibrator(None)
    checkpoint_path = weights_path.resolve() if weights_path else SERVICE_ROOT / "app" / "models" / "arm_segmentation" / "best.pt"
    checkpoint_sha256 = _sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    run_configuration = {
        "input_directory": str(root),
        "recursive": recursive,
        "age_category": age_category,
        "reference_marker_size_mm": reference_size_mm,
        "real_photographs_confirmed_by_operator": real_photos_confirmed,
        "segmenter_mode": mode,
        "model_status": model_status,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "ultralytics": _package_version("ultralytics"),
        "torch": _package_version("torch"),
        "segmentation_environment": {
            key: os.environ.get(key)
            for key in (
                "POSHANSCAN_YOLO_CONFIDENCE",
                "POSHANSCAN_YOLO_MASK_THRESHOLD",
                "POSHANSCAN_YOLO_MIN_MASK_COVERAGE",
                "POSHANSCAN_YOLO_MAX_MASK_COVERAGE",
                "POSHANSCAN_YOLO_IMAGE_SIZE",
            )
        },
    }

    rows: list[dict[str, Any]] = []
    for image_path in images:
        relative_id = image_path.relative_to(root).as_posix()
        try:
            image_sha256 = _sha256_file(image_path)
        except OSError:
            image_sha256 = None
        reference_value = matched_references.get(image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        diagnostics: dict[str, Any] = {}
        result: dict[str, Any]
        processing_error = None
        if image is None:
            result = {"ok": False, "error": "invalid_image"}
            diagnostics.update({
                "image_dimensions_wh": None,
                "marker_detected": False,
                "segmentation_status": "not_run",
                "pose_available": False,
                "accepted_samples": 0,
                "rejected_samples": 0,
                "rejection_reason": "invalid_image",
            })
            overlay_relative = None
            diagnostics_relative = None
        else:
            try:
                result = run_muac_pipeline(
                    image,
                    reference_size_mm,
                    age_category,
                    diagnostics=diagnostics,
                )
            except Exception as exc:  # Preserve remaining images and make this failure auditable.
                result = {"ok": False, "error": "processing_error"}
                processing_error = f"{type(exc).__name__}: {exc}"
                diagnostics.setdefault("rejection_reason", "processing_error")

            overlay_relative = Path("overlays") / Path(relative_id).parent / (
                f"{Path(relative_id).stem}__{Path(relative_id).suffix.lstrip('.').lower()}_geometry.png"
            )
            try:
                _write_overlay(image, diagnostics, result, out / overlay_relative)
                overlay_path = overlay_relative.as_posix()
                diagnostics_relative = overlay_relative.with_suffix(".json")
                diagnostic_payload = {
                    "image_id": relative_id,
                    "pipeline_result": {
                        "ok": bool(result.get("ok")),
                        "error": result.get("error"),
                        "measurement_status": result.get("measurement_status"),
                        "measurement_valid": bool(result.get("ok") and result.get("measurement_valid", True)),
                        "muac_estimate_mm": result.get("muac_estimate_mm") if result.get("ok") else None,
                        "stage_status": result.get("stage_status"),
                    },
                    "diagnostics": {
                        key: value
                        for key, value in diagnostics.items()
                        if key not in {"segmentation_mask", "measurement_mask"}
                    },
                }
                (out / diagnostics_relative).write_text(
                    json.dumps(_json_safe(diagnostic_payload), indent=2, allow_nan=False),
                    encoding="utf-8",
                )
            except Exception as exc:
                overlay_path = None
                diagnostics_relative = None
                processing_error = processing_error or f"Overlay {type(exc).__name__}: {exc}"
        row = _make_row(
            image_path,
            relative_id,
            result,
            diagnostics,
            image_sha256,
            reference_value,
            overlay_path,
            processing_error,
            diagnostics_relative.as_posix() if diagnostics_relative is not None else None,
        )
        rows.append(row)

    paired_rows = [row for row in rows if row["measurement_valid"] and row["reference_muac_mm"] is not None]
    metrics = None
    if real_photos_confirmed and references_path is not None and paired_rows:
        metrics = paired_error_metrics(
            [float(row["estimated_muac_mm"]) for row in paired_rows],
            [float(row["reference_muac_mm"]) for row in paired_rows],
            tolerances_mm,
        )
    _write_report(
        out,
        rows,
        metrics,
        references_path,
        real_photos_confirmed,
        model_status,
        tolerances_mm,
        run_configuration,
        len(references),
        len(matched_reference_ids),
        unmatched_reference_ids,
    )
    return {
        "image_count": len(rows),
        "valid_measurement_count": sum(bool(row["measurement_valid"]) for row in rows),
        "invalid_measurement_count": sum(not bool(row["measurement_valid"]) for row in rows),
        "paired_valid_count": int(metrics["paired_valid_count"]) if metrics else 0,
        "model_status": model_status,
        "reference_row_count": len(references),
        "matched_reference_count": len(matched_reference_ids),
        "unmatched_reference_ids": unmatched_reference_ids,
        "output_dir": str(out),
        "metrics": metrics,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path, help="Folder containing arm photographs")
    parser.add_argument("--output-dir", type=Path, default=SERVICE_ROOT / "results" / "real_image_validation")
    parser.add_argument("--references", type=Path, help="CSV with image_id,reference_muac_mm from manual tape readings")
    parser.add_argument("--age-category", choices=tuple(MUAC_RANGES_MM), default="child_6_59m")
    parser.add_argument("--reference-size-mm", type=float, default=50.0)
    parser.add_argument("--weights", type=Path, help="Optional YOLO segmentation checkpoint")
    parser.add_argument("--tolerances-mm", type=float, nargs="+", default=list(DEFAULT_ERROR_TOLERANCES_MM))
    parser.add_argument("--non-recursive", action="store_true", help="Only inspect files directly in image_dir")
    parser.add_argument(
        "--confirm-real-photographs",
        action="store_true",
        help="Confirm images are real photographs and any references are manually tape-measured; required to calculate accuracy metrics.",
    )
    parser.add_argument(
        "--allow-legacy-fallback",
        action="store_true",
        help="Allow diagnostic-only runs if YOLO cannot load; CSV will identify the actual segmenter.",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.reference_size_mm) or args.reference_size_mm <= 0:
        parser.error("--reference-size-mm must be finite and positive")
    if any(not math.isfinite(value) or value <= 0 for value in args.tolerances_mm):
        parser.error("--tolerances-mm values must be finite and positive")

    synthetic_roots = [
        SERVICE_ROOT / "dataset",
        SERVICE_ROOT / "sample_images",
        SERVICE_ROOT / "results",
    ]
    resolved_images = args.image_dir.resolve()
    if args.confirm_real_photographs and any(
        resolved_images == root.resolve() or root.resolve() in resolved_images.parents
        for root in synthetic_roots
    ):
        parser.error(
            "This project folder contains documented synthetic/unverified or generated review images. Choose a separate external folder of real photographs; do not mark project-generated inputs as real validation."
        )
    try:
        summary = run_folder_validation(
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            references_path=args.references,
            age_category=args.age_category,
            reference_size_mm=args.reference_size_mm,
            recursive=not args.non_recursive,
            real_photos_confirmed=args.confirm_real_photographs,
            tolerances_mm=tuple(args.tolerances_mm),
            require_yolo=not args.allow_legacy_fallback,
            weights_path=args.weights,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
