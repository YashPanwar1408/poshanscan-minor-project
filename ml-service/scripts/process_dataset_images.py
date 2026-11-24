"""Run uploaded images through the existing MUAC pipeline and report stage outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipeline.estimate import CalibrationCorrector  # noqa: E402
from app.pipeline.run import run_muac_pipeline, set_calibrator  # noqa: E402

DEFAULT_IMAGES = ROOT / "dataset" / "images"
DEFAULT_OUTPUT = ROOT / "results" / "dataset_image_processing.json"
SUPPORTED = {".jpg", ".jpeg", ".png"}


def process_images(
    images_dir: Path,
    reference_size_mm: float = 50.0,
    age_category: str = "child_6_59m",
) -> list[dict[str, Any]]:
    files = sorted(
        (path for path in images_dir.rglob("*") if path.is_file() and path.name not in {"README.md", ".gitkeep"}),
        key=lambda path: str(path).lower(),
    ) if images_dir.is_dir() else []
    rows: list[dict[str, Any]] = []
    for path in files:
        row: dict[str, Any] = {
            "filename": str(path.relative_to(images_dir)),
            "dataset_label": "synthetic_unverified",
            "image_loading_status": "not_run",
            "aruco_detection_status": "not_run",
            "pose_detection_status": "not_run",
            "segmentation_status": "not_run",
            "muac_estimation_status": "not_run",
            "baseline_geometric_muac_mm": "",
            "corrected_muac_estimate_mm": "",
            "risk_band": "",
            "processing_error": "",
        }
        if path.suffix.lower() not in SUPPORTED:
            row["image_loading_status"] = "invalid_format"
            row["processing_error"] = f"Unsupported extension {path.suffix or '(none)'}; expected JPG, JPEG, or PNG."
            rows.append(row)
            continue

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            row["image_loading_status"] = "failed"
            row["processing_error"] = "OpenCV could not decode the image."
            rows.append(row)
            continue
        row["image_loading_status"] = "loaded"
        try:
            result = run_muac_pipeline(image, reference_size_mm, age_category)
            stages = result.get("stage_status", {})
            row["aruco_detection_status"] = stages.get("reference", "unknown")
            row["pose_detection_status"] = stages.get("pose", "unknown")
            row["segmentation_status"] = stages.get("segmentation", "unknown")
            row["muac_estimation_status"] = stages.get("muac_estimation", "unknown")
            if result.get("ok"):
                row["baseline_geometric_muac_mm"] = result.get("baseline_muac_estimate_mm", "")
                row["corrected_muac_estimate_mm"] = result.get("muac_estimate_mm", "")
                if row["muac_estimation_status"] == "ok":
                    row["risk_band"] = result.get("risk_band", "")
                elif row["muac_estimation_status"] == "outside_expected_range":
                    row["processing_error"] = (
                        "Estimate is outside the expected age-category range; risk band omitted. "
                        "The numeric value is retained for pipeline diagnostics only."
                    )
            else:
                row["processing_error"] = result.get("message") or result.get("error") or "Pipeline did not return a measurement."
        except Exception as exc:
            row["processing_error"] = f"Pipeline error: {exc}"
        rows.append(row)
    return rows


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--reference-size-mm", type=float, default=50.0)
    parser.add_argument("--age-category", choices=("child_6_59m", "adult"), default="child_6_59m")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--enable-synthetic-calibration", action="store_true")
    parser.add_argument("--model", type=Path, default=ROOT / "app" / "models" / "calibration_synthetic.joblib")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    set_calibrator(None)
    if args.enable_synthetic_calibration:
        model_path = _resolve(args.model)
        try:
            set_calibrator(CalibrationCorrector.load(model_path))
            print("Synthetic calibration demo enabled for this offline run; not for real screening.")
        except Exception as exc:
            print(f"Synthetic model unavailable; using geometric baseline: {exc}", file=sys.stderr)

    rows = process_images(_resolve(args.images), args.reference_size_mm, args.age_category)
    output = _resolve(args.output_json)
    csv_path = output.with_suffix(".csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "dataset_label": "synthetic_unverified",
                "total_images": len(rows),
                "successful_pipeline_images": sum(row["muac_estimation_status"] == "ok" for row in rows),
                "outside_expected_range_images": sum(
                    row["muac_estimation_status"] == "outside_expected_range" for row in rows
                ),
                "stage_status_counts": {
                    key: {
                        status: sum(row.get(key) == status for row in rows)
                        for status in sorted({str(row.get(key, "unknown")) for row in rows})
                    }
                    for key in (
                        "image_loading_status",
                        "aruco_detection_status",
                        "pose_detection_status",
                        "segmentation_status",
                        "muac_estimation_status",
                    )
                },
                "rows": rows,
                "limitation": (
                    "Diagnostic outputs on synthetic or unverified images are not tape validation or clinical evidence. "
                    "An estimate outside the age-category range is flagged and should not be interpreted as a MUAC result."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    fields = list(rows[0].keys()) if rows else [
        "filename", "dataset_label", "image_loading_status", "aruco_detection_status", "pose_detection_status",
        "segmentation_status", "muac_estimation_status", "baseline_geometric_muac_mm",
        "corrected_muac_estimate_mm", "risk_band", "processing_error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Processed {len(rows)} images; wrote {output} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
