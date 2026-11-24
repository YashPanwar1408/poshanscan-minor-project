"""Generate simulated numeric pairs for the Ridge integration demonstration."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "dataset" / "synthetic" / "synthetic_calibration.csv"
PROVENANCE = "simulated_numerical_demo_only"


def generate_rows(n_samples: int = 500, seed: int = 2026) -> list[dict[str, object]]:
    """Create reproducible simulated baseline/target values, never human data."""
    if n_samples < 10:
        raise ValueError("n_samples must be at least 10 to support a useful train/test split.")
    rng = np.random.default_rng(seed)
    baseline = rng.uniform(75.0, 340.0, size=n_samples)
    simulated_reference = 0.94 * baseline + 8.0 + rng.normal(0.0, 4.0, size=n_samples)
    return [
        {
            "sample_id": f"SIM-{index:05d}",
            "baseline_muac_mm": round(float(baseline[index]), 4),
            "simulated_reference_muac_mm": round(float(simulated_reference[index]), 4),
            "provenance": PROVENANCE,
        }
        for index in range(n_samples)
    ]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    rows = generate_rows(args.samples, args.seed)
    write_rows(output, rows)
    print(f"Wrote {len(rows)} simulated numerical rows to {output}")
    print("These values are simulated and are not tape measurements or clinical evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
