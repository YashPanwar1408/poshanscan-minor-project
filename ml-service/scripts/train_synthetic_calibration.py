"""Train and evaluate the optional synthetic Ridge demonstration corrector."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipeline.estimate import CalibrationCorrector  # noqa: E402

DEFAULT_DATA = ROOT / "dataset" / "synthetic" / "synthetic_calibration.csv"
DEFAULT_MODEL = ROOT / "app" / "models" / "calibration_synthetic.joblib"
DEFAULT_JSON = ROOT / "results" / "synthetic_calibration_report.json"
DEFAULT_CSV = ROOT / "results" / "synthetic_calibration_test_predictions.csv"
REQUIRED_COLUMNS = {
    "sample_id",
    "baseline_muac_mm",
    "simulated_reference_muac_mm",
    "provenance",
}
EXPECTED_PROVENANCE = "simulated_numerical_demo_only"


def read_dataset(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Dataset CSV has no header.")
        missing = sorted(REQUIRED_COLUMNS - {name.strip() for name in reader.fieldnames})
        if missing:
            raise ValueError(f"Dataset CSV is missing required columns: {', '.join(missing)}")
        rows = list(reader)

    if len(rows) < 10:
        raise ValueError("At least 10 rows are required for this demonstration train/test split.")
    sample_ids: list[str] = []
    baseline: list[float] = []
    reference: list[float] = []
    for row_number, row in enumerate(rows, start=2):
        if row.get("provenance", "").strip() != EXPECTED_PROVENANCE:
            raise ValueError(
                f"Row {row_number} is not labelled {EXPECTED_PROVENANCE!r}; "
                "real or unverified measurements must not be mixed into this synthetic trainer."
            )
        sample_id = row.get("sample_id", "").strip()
        if not sample_id:
            raise ValueError(f"Row {row_number} has no sample_id.")
        try:
            x = float(row["baseline_muac_mm"])
            y = float(row["simulated_reference_muac_mm"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Row {row_number} has a non-numeric calibration value.") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"Row {row_number} contains a non-finite calibration value.")
        sample_ids.append(sample_id)
        baseline.append(x)
        reference.append(y)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_id values must be unique.")
    return sample_ids, np.asarray(baseline, dtype=np.float64), np.asarray(reference, dtype=np.float64)


def _metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    errors = np.asarray(predicted, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {
        "mae_mm": float(np.mean(np.abs(errors))),
        "rmse_mm": float(np.sqrt(np.mean(np.square(errors)))),
        "mean_signed_bias_mm": float(np.mean(errors)),
    }


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not 0.1 <= args.test_size <= 0.5:
        raise SystemExit("--test-size must be between 0.1 and 0.5.")
    if args.alpha < 0 or not math.isfinite(args.alpha):
        raise SystemExit("--alpha must be a finite non-negative number.")

    data_path = _resolve(args.data)
    if not data_path.is_file():
        raise SystemExit(f"Synthetic calibration CSV not found: {data_path}")
    sample_ids, baseline, simulated_reference = read_dataset(data_path)
    indices = np.arange(len(sample_ids))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
    )

    corrector = CalibrationCorrector(alpha=args.alpha)
    corrector.fit(baseline[train_indices].tolist(), simulated_reference[train_indices].tolist())
    ridge_predictions = np.asarray(
        [corrector.predict(float(value)) for value in baseline[test_indices]], dtype=np.float64
    )
    test_baseline = baseline[test_indices]
    test_target = simulated_reference[test_indices]
    report = {
        "report_type": "synthetic_experimental_results_only",
        "dataset_provenance": EXPECTED_PROVENANCE,
        "dataset_file": str(data_path.relative_to(ROOT)) if data_path.is_relative_to(ROOT) else str(data_path),
        "training_sample_count": int(len(train_indices)),
        "testing_sample_count": int(len(test_indices)),
        "random_seed": int(args.seed),
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "split_method": "random row split; no participant identities exist in numeric simulation",
        "baseline_test_metrics": _metrics(test_baseline, test_target),
        "ridge_test_metrics": _metrics(ridge_predictions, test_target),
        "model_parameters": {
            "estimator": "sklearn.linear_model.Ridge",
            "alpha": float(args.alpha),
            "coefficient": float(corrector.model.coef_[0]),
            "intercept": float(corrector.model.intercept_),
            "feature": "baseline_muac_mm",
            "target": "simulated_reference_muac_mm",
        },
        "limitation": (
            "Simulated numeric data only. These metrics do not validate image processing, "
            "physical scale, tape agreement, or clinical performance."
        ),
    }

    model_path = _resolve(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    corrector.save(model_path)

    json_path = _resolve(args.report_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    csv_path = _resolve(args.report_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "provenance",
                "baseline_muac_mm",
                "simulated_reference_muac_mm",
                "ridge_muac_mm",
                "baseline_error_mm",
                "ridge_error_mm",
            ],
        )
        writer.writeheader()
        for index, ridge_value in zip(test_indices, ridge_predictions):
            writer.writerow(
                {
                    "sample_id": sample_ids[int(index)],
                    "provenance": EXPECTED_PROVENANCE,
                    "baseline_muac_mm": f"{baseline[index]:.4f}",
                    "simulated_reference_muac_mm": f"{simulated_reference[index]:.4f}",
                    "ridge_muac_mm": f"{ridge_value:.4f}",
                    "baseline_error_mm": f"{baseline[index] - simulated_reference[index]:.4f}",
                    "ridge_error_mm": f"{ridge_value - simulated_reference[index]:.4f}",
                }
            )

    print(f"Trained simulated Ridge model: {model_path}")
    print(f"Training/testing rows: {len(train_indices)}/{len(test_indices)}")
    print(f"Synthetic report: {json_path}")
    print(f"Synthetic test predictions: {csv_path}")
    print(json.dumps({"baseline_test_metrics": report["baseline_test_metrics"], "ridge_test_metrics": report["ridge_test_metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
