"""Run production ArUco detection and write private candidate diagnostics.

This script does not modify its input image or call the MUAC pipeline. The
JSON and visualizations contain derivatives of the supplied photograph; store
them in a private, git-ignored output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.pipeline import reference_detect as production_detector  # noqa: E402


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _detector_parameters() -> tuple[Any, dict[str, Any]]:
    if hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
    else:  # OpenCV 4.x legacy API
        params = cv2.aruco.DetectorParameters_create()
    values: dict[str, Any] = {}
    for name in dir(params):
        if name.startswith("_"):
            continue
        try:
            value = getattr(params, name)
        except Exception:
            continue
        if isinstance(value, (bool, int, float, str)):
            values[name] = value
    return params, values


def _detect_with_rejections(gray: np.ndarray) -> tuple[list[np.ndarray], np.ndarray | None, list[np.ndarray]]:
    dictionary = production_detector._get_dictionary()
    if hasattr(cv2.aruco, "ArucoDetector"):
        params, _ = _detector_parameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    return list(corners or []), ids, list(rejected or [])


def _label(image: np.ndarray, message: str, color: tuple[int, int, int]) -> None:
    cv2.putText(image, message, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)


def _resize_panel(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return panel


def inspect(input_path: Path, output_dir: Path, marker_size_mm: float) -> dict[str, Any]:
    raw = input_path.read_bytes()
    pil = Image.open(input_path)
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError(f"OpenCV could not decode {input_path.name}")

    production = production_detector.detect_reference_marker(decoded, marker_size_mm)
    gray = production_detector._to_grayscale(decoded)
    candidates, candidate_ids, rejected = _detect_with_rejections(gray)
    params, params_json = _detector_parameters()

    candidate_ids_list = (
        np.asarray(candidate_ids).reshape(-1).astype(int).tolist()
        if candidate_ids is not None
        else []
    )
    production_corners = production.get("marker_corners")
    production_points = (
        np.asarray(production_corners, dtype=np.float32).reshape(4, 2)
        if production_corners is not None
        else None
    )
    side_lengths: list[float] = []
    if production_points is not None:
        side_lengths = [
            float(np.linalg.norm(production_points[(i + 1) % 4] - production_points[i]))
            for i in range(4)
        ]

    original_overlay = decoded.copy()
    rejected_overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for index, quad in enumerate(rejected):
        points = np.asarray(quad, dtype=np.int32).reshape(-1, 2)
        cv2.polylines(original_overlay, [points], True, (0, 80, 255), 2, cv2.LINE_AA)
        cv2.polylines(rejected_overlay, [points], True, (0, 80, 255), 2, cv2.LINE_AA)
        x, y = points[0]
        cv2.putText(original_overlay, f"R{index + 1}", (int(x), max(18, int(y) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1, cv2.LINE_AA)
        cv2.putText(rejected_overlay, f"R{index + 1}", (int(x), max(18, int(y) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1, cv2.LINE_AA)

    if production_points is not None:
        pts = production_points.astype(np.int32)
        cv2.polylines(original_overlay, [pts], True, (0, 230, 0), 3, cv2.LINE_AA)
        for corner_index, point in enumerate(pts):
            cv2.circle(original_overlay, tuple(point), 5, (255, 80, 0), -1, cv2.LINE_AA)
            cv2.putText(original_overlay, str(corner_index), tuple(point + (7, -7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 0), 2, cv2.LINE_AA)
        cv2.putText(original_overlay, f"DETECTED ID {production['marker_id']}", (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(original_overlay, "NO DECODED MARKER", (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    _label(original_overlay, f"Production result; rejected={len(rejected)}", (0, 230, 0) if production["detected"] else (0, 0, 255))
    _label(rejected_overlay, f"Rejected candidate quadrilaterals: {len(rejected)}", (0, 80, 255))

    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    _label(gray_bgr, "Grayscale input to production detector", (255, 180, 0))
    original_panel = decoded.copy()
    _label(original_panel, "Original decoded pixels", (255, 180, 0))
    panel_size = (720, 960)
    contact_sheet = np.vstack((
        np.hstack((_resize_panel(original_panel, panel_size), _resize_panel(gray_bgr, panel_size))),
        np.hstack((_resize_panel(original_overlay, panel_size), _resize_panel(rejected_overlay, panel_size))),
    ))

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    cv2.imwrite(str(output_dir / f"{stem}_grayscale.png"), gray)
    cv2.imwrite(str(output_dir / f"{stem}_detected_and_rejected.jpg"), original_overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(output_dir / f"{stem}_rejected_candidates.jpg"), rejected_overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(output_dir / f"{stem}_diagnostic_contact_sheet.jpg"), contact_sheet, [cv2.IMWRITE_JPEG_QUALITY, 93])

    report = {
        "filename": input_path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "file_bytes": len(raw),
        "pillow_format": pil.format,
        "pillow_dimensions_wh": list(pil.size),
        "exif_orientation": pil.getexif().get(274),
        "opencv_version": cv2.__version__,
        "opencv_module_path": cv2.__file__,
        "opencv_python_distribution": _distribution_version("opencv-python"),
        "opencv_contrib_distribution": _distribution_version("opencv-contrib-python"),
        "dictionary": "DICT_4X4_50",
        "dictionary_constant": int(production_detector.ARUCO_DICT_ID),
        "detector_api": "ArucoDetector" if hasattr(cv2.aruco, "ArucoDetector") else "detectMarkers legacy API",
        "detector_parameters": params_json,
        "decoded_dimensions_wh": [int(decoded.shape[1]), int(decoded.shape[0])],
        "production_detection": {
            "detected": bool(production["detected"]),
            "marker_id": production["marker_id"],
            "confidence": float(production["confidence"]),
            "reference_size_mm": float(marker_size_mm),
            "scale_px_per_mm": production["scale_px_per_mm"],
            "corners_xy": production_points.tolist() if production_points is not None else None,
            "side_lengths_px": side_lengths,
            "mean_black_square_edge_px": float(np.mean(side_lengths)) if side_lengths else None,
        },
        "same_default_parameter_detector": {
            "detected_ids": candidate_ids_list,
            "rejected_candidate_count": len(rejected),
            "rejected_candidates_xy": [np.asarray(item).reshape(-1, 2).astype(float).tolist() for item in rejected],
        },
        "diagnostic_images": [
            f"{stem}_grayscale.png",
            f"{stem}_detected_and_rejected.jpg",
            f"{stem}_rejected_candidates.jpg",
            f"{stem}_diagnostic_contact_sheet.jpg",
        ],
        "input_was_modified": False,
    }
    (output_dir / f"{stem}_diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="Original image files; inputs are read-only")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--marker-size-mm", default=50.0, type=float)
    args = parser.parse_args()
    for image in args.images:
        print(json.dumps(inspect(image, args.output_dir, args.marker_size_mm), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
