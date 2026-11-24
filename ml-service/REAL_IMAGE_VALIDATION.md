# Real-image MUAC validation

This workflow evaluates the current inference pipeline on a folder of original arm photographs. It does not train or modify the model. It reports the result of the existing ArUco calibration, segmentation, pose/mask axis selection, width sampler, MUAC formula, and measurement-validity checks.

## Windows setup

Run from PowerShell in `ml-service`. Use an isolated virtual environment so the base Python installation is unchanged:

```powershell
$python312 = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
& $python312 -m venv .venv-validation
.\.venv-validation\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-segmentation.txt
```

The requirements file pins the YOLO integration dependency. The validator requires YOLO by default and stops if it cannot load. `--allow-legacy-fallback` is available for diagnostic-only runs; the CSV and report identify if HSV/GrabCut was used. Do not treat that fallback as YOLO evaluation.

## One-command run

Put real photographs in an external folder, separate from the project `dataset`, `sample_images`, and generated `results` folders. Then run:

```powershell
python scripts\validate_real_arm_photos.py "D:\PoshanScanValidation\photos" `
  --output-dir "D:\PoshanScanValidation\results" `
  --references "D:\PoshanScanValidation\manual_tape_muac.csv" `
  --age-category child_6_59m `
  --confirm-real-photographs
```

The tape-reference file must be filled from manual measurements, with one row per image. Start from [`examples/manual_tape_references_template.csv`](examples/manual_tape_references_template.csv), which contains only the required headers:

```csv
image_id,reference_muac_mm
```

Use the path relative to the photo folder when filenames repeat in subfolders (for example `session_1/child_001.jpg`). A unique basename also works. Only measurements with both a valid pipeline estimate and a matched manual reference enter MAE, RMSE, bias, and tolerance percentages. The default tolerances are ±5 mm and ±10 mm; set others with `--tolerances-mm 3 5 10`.

`--confirm-real-photographs` is required before any paired accuracy statistics are calculated. The flag is an operator assertion; it cannot authenticate photo provenance or whether a tape value was actually measured. The CLI also blocks that assertion for the project’s documented synthetic `dataset`, `sample_images`, and generated `results` roots.

## Outputs

The output folder contains:

- `measurements.csv`: relative image identifier and SHA-256, marker/scale, segmentation method and status, pose availability, selected axis method and angle, axis confidence/agreement, accepted and rejected sample counts, sampled width, validity, estimated MUAC only when valid, optional tape reference/error, and rejection reason.
- `metrics.json`: run counts, model checkpoint hash, software versions, relevant runtime settings, confidence definition, unmatched reference IDs, and paired metrics only when real-photo confirmation and valid manual pairs are present.
- `validation_report.md`: run summary and limitations.
- `overlays/`: one original-resolution diagnostic image per readable input, plus a JSON sidecar containing every sampling-line candidate. The image displays the marker, YOLO mask, pose landmarks when available, selected axis, sample center, representative accepted/rejected lines, and measurement status.

Axis confidence is a diagnostic proxy, not a calibrated probability: pose uses the existing landmark visibility score; mask-derived axis uses `1 - 1/eigenvalue_ratio`. The JSON sidecar retains all accepted/rejected sample coordinates even when the overlay limits drawn lines for readability.

## Safety and interpretation

The workflow does not relax measurement validity bounds, change clinical thresholds, alter ArUco marker scale, modify the MUAC formula, or accept invalid estimates. Invalid or failed images remain in the CSV and are omitted from paired accuracy metrics. Do not use generated images or generated tape values as empirical validation. A valid in-range estimate is not proof of accuracy or clinical reliability.
