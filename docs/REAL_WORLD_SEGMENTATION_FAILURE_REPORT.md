# PoshanScan Real-World Segmentation Failure Investigation

**Status:** Local investigation and configuration correction complete. Not pushed or deployed.  
**Primary photograph:** `adult2.jpeg` (the supplied real photograph).  
**Synthetic comparison image:** `child4.webp` was not used as evidence for real-world performance.

## Findings

The failure reproduces on the exact bytes of `adult2.jpeg`. At the service's former inference size of 640 and production confidence threshold 0.25, the integrated model returned no instance. The same unchanged checkpoint, image, class filter, mask threshold and confidence at image size 416 returned one `upper_arm` instance at confidence 0.5917. The checkpoint metadata records training at `imgsz=416`. The existing five-image annotated test split also performed substantially better at 416 than at 640. This establishes an inference-size mismatch as the primary reproducible cause of the reported no-mask result.

At 416, the adult photograph produces a plausible arm-shaped mask, but it also contains a small disconnected component and some boundary spill near clothing/marker. There is no human-reviewed mask for this photograph, so its visual plausibility is not a measured segmentation score and is not proof of boundary accuracy. The small synthetic training set also leaves a real domain-generalization risk.

## Exact input and model

| Item | Evidence |
|---|---|
| Source bytes | 77,867 bytes; SHA-256 `5b4dc0fecac7c9a06e4ca6a6e19c38642d6421517c1a3b0ac912fe5567f37ac6` |
| Decoded image | 960 × 1280 pixels, OpenCV `IMREAD_COLOR` (BGR) |
| Local diagnostic runtime | Python 3.12.3, OpenCV 5.1.0, Ultralytics 8.4.216, CPU |
| Checkpoint | `ml-service/app/models/arm_segmentation/best.pt`; SHA-256 `dc42cf4f893051abeb6faca2bef0f0c56f44f8182b12f2ec904eec71a2a026b1` |
| Checkpoint metadata | task `segment`; class 0 `upper_arm`; training `imgsz=416`, 100 epochs, batch 4 |
| Model handling | Existing checkpoint loaded read-only; not retrained, modified or replaced |

The diagnostic used the byte-identical source copy stored under the ignored local results folder. Its hash matched the supplied original. No image pixels or image hashes were added to application logs by this work. The local FastAPI `/segment` smoke request was made with those same original bytes. A hash of the request received by the current remote Render deployment was not available, so this report does not claim to have traced a live deployed request byte-for-byte.

## Direct prediction results

Direct calls used class filter `[0]`, mask threshold 0.5, `retina_masks=True`, and raw-prediction NMS IoU 0.7. All returned masks were mapped to the original 960 × 1280 dimensions. The service adapter does not explicitly set NMS IoU; its active local predictor reported 0.5. This difference is recorded because the direct raw prediction and adapter are not identical NMS configurations. For the production threshold cases below, the 416 raw and adapter masks were pixel-identical; at 640/confidence 0.15 they were also identical.

| Image size | Confidence | Raw instances | Scores (class 0) | Combined raw mask | Components | Raw inference time* |
|---:|---:|---:|---|---:|---|---:|
| 640 | 0.25 | 0 | — | 0 px (0.00%) | — | 283.0 ms |
| 640 | 0.15 | 1 | 0.1645 | 130,925 px (10.65%) | 130,600; 325 px | 173.2 ms |
| 640 | 0.10 | 1 | 0.1645 | 130,925 px (10.65%) | 130,600; 325 px | 149.2 ms |
| 640 | 0.05 | 1 | 0.1645 | 130,925 px (10.65%) | 130,600; 325 px | 145.1 ms |
| 640 | 0.01 | 4 | 0.1645, 0.0410, 0.0201, 0.0151 | 350,386 px (28.51%) | 349,636; 750 px | 186.3 ms |
| 416 | 0.25 | 1 | 0.5917 | 202,690 px (16.49%) | 202,084; 606 px | 44.1 ms |
| 416 | 0.15 | 2 | 0.5917, 0.1830 | 251,383 px (20.46%) | 249,858; 919; 606 px | 122.7 ms |
| 416 | 0.10 | 2 | 0.5917, 0.1830 | 251,383 px (20.46%) | 249,858; 919; 606 px | 119.8 ms |
| 416 | 0.05 | 3 | 0.5917, 0.1830, 0.0696 | 251,383 px (20.46%) | 249,858; 919; 606 px | 131.2 ms |
| 416 | 0.01 | 7 | 0.5917, 0.1830, 0.0696, 0.0194, 0.0107, 0.0107, 0.0105 | 415,813 px (33.84%) | 415,069; 606; 138 px | 136.8 ms |

At 416/confidence 0.25 the top prediction has bounding box `[252.91, 52.24, 559.45, 941.90]` in original-image XY coordinates. At 640/confidence 0.15 the sole prediction has box `[299.72, 39.76, 551.93, 679.89]`. The latter is visibly incomplete compared with the arm in the photograph. At confidence 0.01, both sizes admit multiple low-score masks and substantially larger foreground coverage; those diagnostic thresholds are not proposed for production.

\* One inference call per row after a warm-up; CPU timing varies with image size and machine load and is not a benchmark.

The production crop path on the 640/confidence 0.25 miss entered the legacy fallback, which is not accepted as a YOLO result. The marker-based crop context was available (marker ID 0, center approximately `(387.25, 542.75)`, crop 397 × 397, origin `(189, 345)`). At 416/confidence 0.25 the model path returned a YOLO mask. The crop does not explain the original absence: the full-image model had already returned zero candidates at 640/confidence 0.25.

No ground-truth polygon exists for `adult2.jpeg`; therefore expected-region overlap, IoU, Dice, precision and recall were **not computed**. The percentages above are image coverage by a prediction, not accuracy.

## Training-data comparison

The locally available checkpoint-matched quarantine dataset has 40 annotated images: 28 train, 7 validation and 5 test. The project dataset provenance describes these as AI-generated examples. The existing split was inspected and not changed.

| Property | Existing subset | `adult2.jpeg` / observation |
|---|---|---|
| Arm orientation | Principal-axis median 90.78°; range 81.16–100.76° | Upright portrait arm; orientation is not an obvious outlier |
| Relative annotated mask size | Median width 26.6% and height 45.6% of image | 416 predicted box is about 31.9% wide and 69.5% high; larger/taller than the training median |
| Mask coverage | Median 8.89%; range 2.17–27.30% | 16.49% for the 416/confidence 0.25 prediction; not directly comparable as a quality metric |
| Image/background | Varied generated scenes; background edge-density proxy median 3.53%, range 1.03–9.27% | Real adult subject, dark graphic shirt, a hand near the shoulder, marker beside clothing and a busy background; visual domain differs |
| Subject and arm | Predominantly generated child examples | Adult arm and larger apparent arm extent; no locally available real adult training example identified |
| Ground truth | Human-reviewed polygons in the selected training subset | No reviewed annotation for this real photograph |

These distribution observations support a domain-generalization concern, but one photograph cannot establish broad real-world failure or success. The 40-image set is small; generated appearance may not represent real skin, clothing boundaries, marker printing, occlusion and camera variation.

## Preprocessing and post-processing investigation

1. The local `/segment` decoder uses `cv2.imdecode(..., cv2.IMREAD_COLOR)` and passes the resulting BGR NumPy image to Ultralytics. No application resize, crop, JPEG recompression or color conversion occurs before the model call in this path. Ultralytics performs its normal aspect-preserving letterbox for the requested image size.
2. The diagnostic passed the exact original BGR array to the checkpoint, with `retina_masks=True`. Ultralytics reported `orig_shape=[1280, 960]`; raw masks were returned at 960 × 1280. The adapter mask had the same full-image dimensions.
3. Class filtering is `[0]` / `upper_arm`. The default confidence remains 0.25 and the mask threshold remains 0.5. Existing coverage checks are 0.5%–95%; they did not reject the 416/confidence 0.25 result.
4. At 416/confidence 0.25, raw and adapter masks had IoU 1.0 with each other. At 640/confidence 0.15, raw and adapter masks also had IoU 1.0. Thus no evidence identifies mask resizing, letterbox removal, class filtering, or service mask post-processing as the reason for the 640/confidence 0.25 miss.
5. At lower thresholds the adapter can discard/alter extra detections as designed. This does not justify accepting a low-confidence spill. The threshold-0.01 visualizations show why lowering production confidence would be unsafe.
6. The independent earlier ArUco check reported ID 0 on the supplied real photo. This investigation confirms the model result with the marker-derived crop context; ArUco calibration and MUAC code were not changed.
7. The offline diagnostic environment reported OpenCV 5.1.0 and Ultralytics 8.4.216. The deployed Render image was not rebuilt or interrogated during this task; a remote request hash and runtime package versions were not collected. The local evidence establishes the image-size effect in the integrated local runtime, not parity of every remote dependency.

## Visual and quantitative artifacts

Artifacts are stored in `ml-service/results/real_world_segmentation_failure/rerun/`. This directory is ignored by Git because it contains a supplied person photograph and derived visualizations; it remains local.

- `original.jpeg` — byte-identical source copy (hash recorded above).
- `predictions.csv` — class, confidence, box, dimensions, pixel area, coverage, components and explicit no-ground-truth note for each returned instance.
- `diagnostic_summary.json` — runtime, input/checkpoint hashes, preprocessing, crop context, direct-vs-adapter IoU settings and per-threshold summaries.
- `imgsz_416/conf_0p250/` and `imgsz_640/conf_0p250/` — original, raw boxes, raw mask, mask overlay, contours, connected components, post-processed full-image mask/overlay, crop mask/overlay/contours, and raw-vs-postprocessed comparison.
- `imgsz_416/conf_0p150/`, `imgsz_416/conf_0p010/`, `imgsz_640/conf_0p150/`, and `imgsz_640/conf_0p010/` — representative lower-confidence outputs showing added detections and spill.
- `training_subset_contact_sheet.jpg` — visual review of all 40 annotated training/validation/test examples.
- `training_subset_image_audit.csv` and `training_subset_summary.json` — image/mask size, orientation, coverage and descriptive image statistics.
- `annotated_test_resolution_comparison.csv` and `annotated_test_resolution_summary.json` — comparison against the existing five annotated test images, without split changes.
- `annotated_test_resolution/imgsz_416/` and `annotated_test_resolution/imgsz_640/` — prediction/reference overlays for that same five-image test split.

The pre-existing investigation outputs in `ml-service/results/real_world_segmentation_failure/` were retained. The rerun folder was created after correcting diagnostic metadata so raw NMS IoU and the adapter's effective predictor IoU are reported separately.

## Existing annotated test-split comparison

The five existing annotated test images were evaluated using the current adapter at confidence 0.25 and mask threshold 0.5. No labels or split files were edited.

| Inference size | Detections | Mean per-image IoU | Mean per-image Dice |
|---:|---:|---:|---:|
| 640 | 4/5 | 0.7417 | 0.7697 |
| 416 | 5/5 | 0.9270 | 0.9616 |

This is evidence that 416 better matches the existing checkpoint and its five annotated synthetic test images. It is **not** real-image or clinical validation. The sample is only five images from a small, generated dataset.

## Classification and local change

### A. Model inference/configuration — confirmed issue

The former default inference size was 640 while the checkpoint was trained at 416. On the exact real photograph, 640/confidence 0.25 yielded no prediction, while 416 at the same confidence produced one stronger prediction. On the unchanged five-image annotated test split, 416 also had higher mean IoU/Dice and detected all five images. No confidence threshold was lowered.

**Local correction:** default YOLO image size is now 416 in `ml-service/app/pipeline/segment.py`, the `/segment` route, and the prediction/evaluation helpers. `POSHANSCAN_YOLO_IMAGE_SIZE` can still override the default. The adapter's image size is set from this configuration; no API response fields changed. A local FastAPI `/segment` request with no `imgsz` query returned HTTP 200, `detected=true`, one class-0 `upper_arm` result at 0.5917 confidence, and a 960 × 1280 mask. The route did not use a legacy fallback.

### B. Image preprocessing — not indicated as the cause locally

The original bytes decoded to the expected dimensions and color order. The model output and adapter retained the original spatial dimensions, and raw/adapter masks matched in the relevant single-prediction cases. No evidence of a crop or coordinate remapping bug explains the 640 miss.

### C. Mask post-processing — not indicated as the cause of the threshold miss

Coverage and centered-crop checks accepted the 416 result; they do not manufacture its stronger confidence. The 640/confidence 0.25 raw model result was empty before crop post-processing. No post-processing change was made.

### D. Training domain — meaningful remaining risk

Only 40 generated images are available for this checkpoint-matched subset, and the real adult photograph has no counterpart in the training annotations. Improving real-world domain coverage is warranted before claiming robust deployment behavior. This is a limitation and a next-step need, not evidence that this one photo alone proves the model cannot segment real arms.

### E. Real-image annotation/ground truth — unresolved limitation

There is no independently reviewed arm mask for `adult2.jpeg`. We cannot quantify whether the 416 mask leaks outside the arm, omits anatomy, or follows the intended amodal policy correctly. A qualified human must annotate and review real-photo polygons before real-image segmentation accuracy can be reported.

## Recommendation for the next training decision

Do not retrain solely from the visual appearance of this one prediction. First, curate a pilot real-photo set and annotate it under the existing upper-arm policy: shoulder-to-elbow anatomical region, arm boundary only, excluding torso/clothing/background/forearm; maintain the established amodal convention under the ArUco marker. Include cropped arms clipped at image edges only under the approved visible-region policy. Have annotations reviewed by a person familiar with the target anatomy.

A useful first pilot is roughly 50–100 diverse, real photographs from at least 10–20 distinct people, with varied ages/body sizes, skin tones, arm orientation, clothing edges, lighting, phones, distance, marker placement, partial occlusion and backgrounds. This is a practical starting range, not a statistical guarantee or claim that it is sufficient for production. Group every image from the same person/session into one partition to prevent subject leakage. Keep a subject-disjoint validation group and a locked subject-disjoint test group; do not use test images for tuning. Add the real examples to training only after annotations are reviewed, preserving an explicit manifest and recording the split seed.

Evaluate segmentation separately from MUAC: report per-image/pixel IoU and Dice against reviewed real masks, mask-detection/empty-mask rates, boundary overlays, and subgroup/error slices by capture condition. Evaluate MUAC independently using standardized tape references and a separate protocol; segmentation overlap does not demonstrate measurement agreement or clinical reliability.

## Files changed and verification

Local implementation changes for this investigation:

- `ml-service/app/pipeline/segment.py` — default inference size 416.
- `ml-service/app/main.py` — `/segment` default derives from the segmentation configuration.
- `ml-service/scripts/predict_arm_segmentation.py` and `ml-service/scripts/evaluate_poshanscan_colab.py` — defaults aligned to 416.
- `ml-service/app/models/arm_segmentation/README.md` and `ml-service/README.md` — document checkpoint-aligned inference size.
- `ml-service/tests/test_segmentation_workflow.py` — regression checks for the 416 default.
- `ml-service/scripts/diagnose_real_world_segmentation.py` — reproducible diagnostic; reports raw NMS IoU separately from the adapter predictor's effective IoU and defaults to checking 416 before 640.

The integrated checkpoint hash was rechecked and matches the expected digest. The complete ML-service test suite passed after the production default change: **118 passed**, with one upstream Starlette/httpx deprecation warning. No MUAC formula, ArUco calibration, clinical threshold, dataset split, annotation, frontend contract, checkpoint or deployed service was changed.

`git diff --check` reported no whitespace errors; Git emitted only its standard LF-to-CRLF working-copy notices. Existing unrelated local edits from the earlier API logging and ArUco investigation remain in the worktree and are not attributed to this report.

## Conclusion

The no-mask failure at the configured confidence was reproducibly caused by running this 416-trained checkpoint at inference size 640. The local default is corrected to 416, and direct inference plus the local `/segment` route were verified. This change resolves the reproducible configuration mismatch, but it does not establish real-world segmentation accuracy. The 416 output still needs a human-reviewed real-photo ground-truth mask, followed by subject-disjoint evaluation across a diverse real-photo set, before deployment claims or further retraining decisions can be justified.
