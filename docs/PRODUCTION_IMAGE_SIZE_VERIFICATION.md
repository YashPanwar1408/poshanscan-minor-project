# PoshanScan Production Image-Size Verification

**Status: prepared locally; production is not yet verified.** No Git push or Render deployment was performed. The deployed Render service's dashboard environment settings are not accessible from the tools available in this session, so its current `POSHANSCAN_YOLO_IMAGE_SIZE` value is unknown.

## Configuration audit

The service resolves inference size in `ml-service/app/pipeline/segment.py` as follows: an explicit function argument wins; otherwise `POSHANSCAN_YOLO_IMAGE_SIZE` wins; otherwise the default is 416. The `/infer` pipeline uses the loaded segmentation configuration. `/segment` defaults its optional `imgsz` query parameter to 416, but callers can explicitly request another size.

The project-wide search found the environment variable in:

- `ml-service/app/pipeline/segment.py` — environment parsing and validation.
- `ml-service/tests/test_segmentation_workflow.py` — configuration regression test.
- `ml-service/scripts/validate_real_arm_photos.py` — validation report metadata.
- `ml-service/README.md` and `ml-service/app/models/arm_segmentation/README.md` — setting documentation.
- `docs/REAL_WORLD_SEGMENTATION_FAILURE_REPORT.md` — investigation documentation.
- `render.yaml` — now explicitly declares `POSHANSCAN_YOLO_IMAGE_SIZE: "416"` for the `poshanscan-minor-project-ml` Docker service.

No other checked-in deployment file or environment template declares a competing value. The variable is unset in the local shell used for verification. Render's live dashboard value cannot be confirmed here; a manually configured dashboard value or a service that has not synced the updated Blueprint may still be 640.

There is also a service-identity mismatch to resolve before deployment. The checked-in `render.yaml` names its ML service `poshanscan-minor-project-ml`, while earlier user-provided Render screenshots showed the backend calling `https://poshanscan-minor-project-ml-model.onrender.com` and showed logs for a service named `poshanscan-minor-project-ml-model`. One earlier instance logged `imgsz=640`. That is evidence about the instance visible in that screenshot, not proof of the current live dashboard setting. The repository's backend reads `CV_SERVICE_URL` from the backend environment; its default is local `http://localhost:8001`. I cannot verify the current backend `CV_SERVICE_URL`, current Render services, or which service currently handles production traffic.

The checked-in Render Blueprint parses as valid YAML and sets the declared ML service to 416. That service is named `poshanscan-minor-project-ml`, uses `rootDir: ml-service`, and builds `./Dockerfile`. This does not establish that the differently named service in the earlier screenshots is Blueprint-managed or will receive this setting.

## Actual local runtime verification

The service was started through FastAPI's application lifespan, which loaded the integrated checkpoint and initialized the actual predictor. This verifies the runtime call path rather than only reading the default constant.

| Check | Observed result |
|---|---|
| Local `POSHANSCAN_YOLO_IMAGE_SIZE` | Unset |
| Startup segmentation log | `conf=0.250 mask=0.500 ... imgsz=416` |
| Loaded runtime configuration | image size 416; confidence 0.25; mask threshold 0.5 |
| Ultralytics predictor arguments after `/segment` | `imgsz=416`, `conf=0.25`, `iou=0.5`, `classes=[0]`, `device=cpu` |
| Ultralytics predictor arguments after `/infer` | `imgsz=416`, `conf=0.25`, `iou=0.5`, `classes=[0]`, `device=cpu` |
| Checkpoint | `best.pt`; task `segment`; class 0 `upper_arm` |
| Checkpoint SHA-256 | `dc42cf4f893051abeb6faca2bef0f0c56f44f8182b12f2ec904eec71a2a026b1` — matches expected value |

With the byte-identical local copy of `adult2.jpeg`, `/segment` without an `imgsz` query returned HTTP 200, `detected=true`, one class-0 `upper_arm` prediction at confidence 0.5917425, and a mask of 960 × 1280 pixels. The request used the original 960 × 1280 image. This confirms the local default route actually executed inference at 416.

The local `/infer` route also completed with predictor `imgsz=416`. For `age_category=adult`, it returned HTTP 200 and 331.425 mm / `Normal`; for `age_category=child_6_59m`, it returned HTTP 200 and 137.733 mm / `Normal`. These are unreferenced outputs from an adult photograph with no tape-measured MUAC. They are recorded to distinguish pipeline execution from measurement validation and must not be treated as accuracy or clinical evidence. The adult photograph is not an appropriate pediatric evaluation sample.

The expected source image hash was `5b4dc0fecac7c9a06e4ca6a6e19c38642d6421517c1a3b0ac912fe5567f37ac6`; the exact source copy and all diagnostic images remain under the ignored, local-only `ml-service/results/real_world_segmentation_failure/` folder.

## Preserved behavior and artifacts

This verification changes no model weights, confidence threshold, mask threshold, mask coverage limits, MUAC formula, ArUco dictionary/calibration, clinical threshold, API schema or frontend behavior. The checkpoint file was not modified. The only deployment setting added in this task is the explicit 416 value in `render.yaml`.

The prior local inference-size correction remains in the ML service defaults and inference helpers. Its evidence and diagnostic overlays are documented in [REAL_WORLD_SEGMENTATION_FAILURE_REPORT.md](REAL_WORLD_SEGMENTATION_FAILURE_REPORT.md).

## Tests and build

| Suite | Result |
|---|---|
| ML service | 118 passed; one Starlette/httpx deprecation warning |
| Backend | 41 passed; one Starlette/httpx deprecation warning and pytest-asyncio configuration deprecation warning |
| Frontend | 44 passed across 3 test files |
| Frontend production build | Passed (`vite build`) |
| Render Blueprint parse | Passed; ML service environment value is `416` |
| Checkpoint SHA-256 | Passed; matches expected hash |

The backend suite ran in a new ignored `backend/.venv` so the existing Anaconda and ML-service Python environments were not replaced. Its initial collection required setting the inherited test-process `DEBUG` value to `false`; that change was limited to the test process and is not a deployment edit.

## Render redeployment and verification steps

1. In Render, inspect the backend service's `CV_SERVICE_URL` and identify the exact ML service that receives production `/infer` calls. Earlier screenshots point to `poshanscan-minor-project-ml-model`, but verify the current value in the dashboard.
2. Open that exact ML service's **Environment** settings. Inspect `POSHANSCAN_YOLO_IMAGE_SIZE`; set it to `416`, or remove it so the code default applies. Do not leave it at `640`. The `render.yaml` value applies to the service named `poshanscan-minor-project-ml` only unless you confirm/sync the Blueprint to the live service.
3. Ensure the reviewed code containing the 416 default and updated `render.yaml` is pushed to the repository. This worktree has other pending changes, so review and stage specific intended files; do not use a blanket `git add .` without reviewing them.
4. After the authorized push, use Render's Blueprint sync only after confirming which service it manages, then deploy the ML service that `CV_SERVICE_URL` points to. Wait for that service to report **Live**.
5. In that service's logs, confirm the startup line reports `imgsz=416`, `conf=0.250`, and `mask=0.500`. A 640 startup line means the live setting or deployed commit is still wrong; stop and correct it before accepting the redeployment.
6. From PowerShell, run `/segment` with the original file and no explicit `imgsz` parameter:

   ```powershell
   curl.exe -sS -X POST "https://<ML-SERVICE-HOST>/segment?confidence=0.25" `
     -F "image=@D:\Downloads\adult2.jpeg;type=image/jpeg" `
     -o segment-response.json
   ```

   Inspect `segment-response.json`: require HTTP 200, `detected: true`, an instance with `class_id: 0`, `class_name: "upper_arm"`, and `width: 960`, `height: 1280`. Decode/review the returned mask and visualization; a successful response alone does not prove boundary accuracy.

7. Run the full `/infer` request separately, matching the photograph's adult category:

   ```powershell
   curl.exe -sS -X POST "https://<ML-SERVICE-HOST>/infer" `
     -F "image=@D:\Downloads\adult2.jpeg;type=image/jpeg" `
     -F "age_category=adult" `
     -F "reference_type=aruco" `
     -F "reference_size_mm=50" `
     -o infer-response.json
   ```

   Confirm in logs that marker detection and YOLO inference ran, and that the startup/effective predictor size is 416. Inspect any measurement under existing validity rules. Do not treat a successful status or a `Normal` band as a verified MUAC value without an independent tape reference. A genuine invalid-measurement 422 is preferable to fabricating or forcing a valid result.

8. Record the actual ML service name/URL, backend `CV_SERVICE_URL`, deployment ID, timestamp, startup log, HTTP statuses, response metadata, image/checkpoint hashes and reviewed overlays in this report after the remote checks. Keep the original image private; do not publish it in logs or public artifacts.

## Production verification status

The code and checked-in Blueprint are prepared to use 416, and the local model's actual `/segment` and `/infer` calls were observed using predictor `imgsz=416`. **Production is not yet verified.** The current Render dashboard setting, active ML service identity, backend `CV_SERVICE_URL`, deployed commit, live startup configuration, and production `/segment`/`/infer` responses remain uninspected because no Render connector or credentials are available here and no deployment was authorized. Do not describe the deployed service as fixed until the above remote checks pass.
