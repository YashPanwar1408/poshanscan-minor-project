# PoshanScan Real Photo ArUco Debug Report

**Status:** local investigation complete; no push or deployment performed.  
**Priority image:** `D:\Downloads\adult2.jpeg` (original file).  
**Control image:** `D:\Downloads\child4.webp` (AI-generated; not real-world evidence).

## Findings

The real photograph's marker is detected by the production ArUco function both locally and in the deployed ML pipeline. The deployed `/infer` request did not fail at marker detection: it reached arm segmentation and returned `segmentation_failed`. Thus the failure reproduced with this exact photograph is upstream of MUAC measurement but downstream of ArUco calibration. It is not evidence that the ArUco dictionary or detector needs to be weakened or changed.

The deployed `/segment` request also returned HTTP 200 but no usable arm instance at the configured confidence threshold (`0.25`). For diagnosis only, inference at confidence `0.10` returned a mask at confidence `0.1645`; its overlay spills onto the marker/background/clothing and is not a valid arm mask. Lowering the production threshold would accept a visibly poor mask, so no threshold change was made. The observed real-photo blocker is the segmentation model's failure to produce a usable mask on this photograph; synthetic-to-real generalization is a plausible explanation, but cannot be proven from these two images alone.

The AI-generated `child4.webp` yields no decoded marker and 25 rejected quadrilaterals under local OpenCV 5.1.0. The user independently observed no decoded ID and 21 rejected candidates with OpenCV 4.13.0. This is consistent across versions and is not real-world marker validation.

## Original-image evidence

| Image | SHA-256 | Bytes | Format / dimensions | EXIF orientation | Local production result |
|---|---|---:|---|---|---|
| `adult2.jpeg` | `5b4dc0fecac7c9a06e4ca6a6e19c38642d6421517c1a3b0ac912fe5567f37ac6` | 77,867 | JPEG, 960 × 1280 | absent | ID 0 detected; confidence 0.9752 |
| `child4.webp` | `6fc050d38e6cf01127665a458a5fc294c0c8ef2af6535991e25d661689ac0ed7` | 1,378,958 | WebP, 1086 × 1448 | absent | no ID; 25 rejected candidates |

For `adult2.jpeg`, the local production code used OpenCV **5.1.0**, dictionary **`DICT_4X4_50`** (constant 0), and the `ArucoDetector` API. It decoded the image to 960 × 1280. Detected ID: **0**. Detected corners (x,y): `[(307,464), (467,464), (468,621), (307,622)]`. Edge lengths were approximately 160.00, 157.00, 161.00, and 158.00 pixels (mean **159.00 px**). With the configured 50 mm reference edge, the detector computed **3.1800 px/mm**. The same-parameter detector also reported **10 rejected candidates**, while still decoding ID 0.

The default detector parameters recorded from this local runtime include adaptive threshold window 3–23 (step 10), threshold constant 7, minimum/maximum perimeter rates 0.03/4.0, polygon approximation rate 0.03, minimum corner distance rate 0.05, minimum border distance 3, corner refinement disabled (0), and error-correction rate 0.6. The full parameter snapshot and all rejected quadrilaterals are in `ml-service/results/real_photo_aruco_debug/adult2_diagnostics.json`.

The user's independent OpenCV **4.13.0** test also decoded ID 0 from the same original adult photo. This agrees with the local production result. The local environment has both `opencv-python` and `opencv-contrib-python` distributions installed at **5.1.0.95**; the latter provides the ArUco module. Their simultaneous installation is unnecessary/ambiguous package ownership, but it did not prevent detection in this test and is not established as the cause of the reported failure. The deployed runtime's exact OpenCV version is **unknown**: its prior logs/health response did not expose it. Startup logging for version, dictionary, and API has been added locally, but was not deployed.

## Deployed endpoint reproduction

The exact original `adult2.jpeg` bytes were sent directly as a multipart upload to the deployed ML endpoints; the file itself was not recompressed or replaced.

| Endpoint | Result | Interpretation |
|---|---|---|
| `/segment` with configured confidence 0.25 | HTTP 200; 960 × 1280; zero usable instances | Arm segmenter did not return an accepted mask. |
| `/infer` (`reference_type=aruco`, 50 mm, adult category) | HTTP 422, `segmentation_failed` | The pipeline passed the marker stage and failed at segmentation; no MUAC value was accepted or fabricated. |
| `/segment` diagnostic confidence 0.10 | One `upper_arm` result at confidence 0.1645 | Visual mask is poor and spills outside the arm; diagnostic only, not accepted as a fix. |

The deployed `/infer` response is direct runtime evidence that ArUco detection did not cause this request's failure: the pipeline emits this segmentation error only after its marker/localization stage. A full authenticated browser → backend `/scans` run was not performed because no worker session was available. Therefore, the report does not claim a deployed end-to-end browser test or a remote byte-hash comparison.

## Image flow and transformations

Static trace of the upload path:

1. In `frontend/src/routes/app/capture.$childId.tsx`, file upload stores the selected `File` as a `Blob` directly. Preview uses an object URL. Submission wraps the same blob in a `File` named `capture.jpg`; that changes multipart filename/type metadata, not the blob's encoded bytes. The separate camera-capture path draws a video frame to canvas and encodes JPEG at quality 0.9; it was not used for this original-file upload.
2. `backend/app/routers/scans.py` reads the complete upload bytes and forwards them. It does not resize, crop, rotate, or re-encode the image.
3. `backend/app/services/cv_client.py` forwards those bytes as multipart. Before this local change it always labeled every file `image/jpeg`, even when the original was WebP. The HTTP content type does not change bytes, and OpenCV decodes from the byte stream, but wrong metadata is avoidable. The client now forwards the incoming MIME type (or filename-derived type) while preserving the bytes.
4. `ml-service/app/main.py` decodes the byte stream with `cv2.imdecode(..., cv2.IMREAD_COLOR)`. The tested JPEG has no EXIF orientation; decoded size remains 960 × 1280.

Opt-in privacy-preserving diagnostics now log only SHA-256, byte count, content type, and decoded dimensions at backend-forward and ML-receive stages when `POSHANSCAN_IMAGE_DIAGNOSTICS=1` (or `true`). They are off by default and never log pixels or save the uploaded image. Local logs confirmed the source hash and byte count at both ML `/segment` and `/infer` receive stages: `5b4dc0...37ac6`, 77,867 bytes, 960 × 1280. Since this instrumentation is local-only, it cannot establish a hash from the already-running deployed backend. The remote reproduction instead sent the original file directly to the ML endpoint.

## Calibration dimension

`ml-service/scripts/generate_aruco_marker.py` explicitly instructs users to measure the **black ArUco square edge** and pass that measurement as `reference_size_mm`. The detector corners bound that black code square; the surrounding white quiet zone is not part of the measured edge. The generated image defaults to an 800 px black square plus a 120 px white border on each side, so the full white-edged print is larger than the calibration square.

Therefore the code's configured 50 mm means **black-square edge = 50 mm**, not the outer white paper/card edge. The actual physical print was not measured with a ruler as part of this investigation. Do not use 50 mm if that number describes only the outer white boundary; verify the printed black edge first. No calibration or UI value was changed.

## Local changes

- `ml-service/scripts/debug_aruco_image.py`: reproducible read-only tool using the production detector; records hashes, dimensions, version, parameters, corners, candidate quads, and derived diagnostic images. It supports both current and legacy OpenCV detector APIs.
- `backend/app/services/cv_client.py`, `backend/app/routers/scans.py`, `backend/app/routers/sync.py`: preserve bytes while forwarding the correct multipart MIME type; optional hash/byte-count logging.
- `ml-service/app/main.py`: startup OpenCV/dictionary/API log and opt-in image fingerprint/dimension logs for `/segment` and `/infer`.
- `backend/tests/test_cv_client.py`, `backend/tests/conftest.py`, `ml-service/tests/test_api.py`: coverage for MIME/byte forwarding and privacy-safe opt-in diagnostics.

No change was made to the dictionary, marker ID behavior, detector thresholds, calibration formula, MUAC formula, clinical thresholds, YOLO checkpoint, or public API response contract. No model retraining, dataset edit, push, or deployment was performed.

## Tests

- ML service: **118 passed**, one upstream Starlette/httpx deprecation warning.
- Backend, run in the existing `backend-api:latest` Docker image against a temporary staging copy excluding `.env`: **41 passed**, one upstream Starlette/httpx deprecation warning.
- Local production marker detection: **adult2 ID 0 detected**, expected edge scale computed.
- Deployed ML smoke: marker stage passed; `/segment` returned no accepted mask and `/infer` correctly rejected the request as `segmentation_failed`.
- No frontend test/build or authenticated browser `/scans` test was run in this pass.

## Diagnostic artifacts and limits

Private, ignored artifacts are stored under `ml-service/results/real_photo_aruco_debug/`, including original/grayscale panels, detected corners, rejected quadrilaterals, JSON evidence, and the low-confidence YOLO overlay. The supplied original files were read-only. Keep this directory private because visual derivatives still contain the photographs.

To rerun the local production-detector diagnostics from the repository's `ml-service` directory:

```powershell
.\.venv\Scripts\python.exe scripts\debug_aruco_image.py D:\Downloads\adult2.jpeg D:\Downloads\child4.webp --output-dir results\real_photo_aruco_debug
```

The two photographs establish behavior only for these specific files. They do not establish broad marker robustness or clinical/model accuracy. The deployed service's OpenCV version and the full authenticated browser/backend byte path remain unverified. The concrete blocker for the tested adult photo is its unusable predicted arm mask, not ArUco decoding.
