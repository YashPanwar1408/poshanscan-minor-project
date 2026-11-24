# PoshanScan ML Runtime Verification

**Date:** 2026-11-24
**Scope:** Dependency configuration, checkpoint loading, and real local HTTP smoke requests. This is integration evidence, not model-accuracy or clinical validation.

## Root cause

The production ML service in `render.yaml` uses the Docker runtime with `rootDir: ml-service` and `dockerfilePath: ./Dockerfile`. The Dockerfile previously defaulted `INSTALL_SEGMENTATION_DEPS=false`, and installed `requirements-segmentation.txt` only when that argument was explicitly enabled. The production configuration supplied no build argument, so the image omitted Ultralytics and PyTorch and could not load the integrated checkpoint.

The local shell resolved `python` to `D:\anaconda3\python.exe`, which failed to start with Windows `0xC0000022` (access denied). A per-user Python 3.12.3 installation was available and executable. The prior local service runtime therefore had no installed Ultralytics package, matching the earlier `No module named 'ultralytics'` failure.

## Changes made

- The Dockerfile now installs the pinned segmentation requirement unconditionally.
- The Docker build verifies the checkpoint SHA-256, loads it with Ultralytics, and checks task `segment` plus class map `{0: upper_arm}`. An invalid or missing checkpoint now fails the image build.
- `requirements-segmentation.txt` contains only the pinned Ultralytics package; base requirements are installed separately, avoiding a second unpinned resolution of the base packages.
- Local setup instructions create and activate an isolated `.venv` and install both requirements files. Global Anaconda packages are not modified.
- The 6.69 MiB checkpoint was not modified.

## Checkpoint and runtime verification

| Check | Result |
|---|---|
| File | `ml-service/app/models/arm_segmentation/best.pt` exists |
| SHA-256 | `dc42cf4f893051abeb6faca2bef0f0c56f44f8182b12f2ec904eec71a2a026b1` (matches expected) |
| Ultralytics | `8.4.216` |
| PyTorch / torchvision | `2.15.0+cpu` / `0.29.2+cpu` |
| CUDA | Unavailable; CPU inference was used |
| Model metadata | task `segment`; class `{0: upper_arm}` |
| OpenCV ArUco | Available |

The working runtime was `ml-service/.venv` created with Python 3.12.3. Model loading and real prediction ran through the service adapter on CPU.

## Real HTTP smoke requests

The existing labeled project image `ml-service/dataset/images/10.png` was used as an integration sample. No labels or images were changed.

- `GET /health`: HTTP 200.
- `POST /segment?imgsz=416&confidence=0.25`: HTTP 200; one detected `upper_arm` instance at confidence `0.8550`. The source image was 1086 by 1448 pixels; the decoded mask and overlay retained the exact source dimensions. The mask contained 159,114 foreground pixels. `/segment` only runs in YOLO mode, and the startup log confirmed it loaded the specified checkpoint on CPU.
- `POST /infer`: HTTP 422 `invalid_measurement`. The request genuinely ran the pipeline: ArUco marker ID 0 was detected; MediaPipe did not return pose landmarks, so the existing marker fallback was used; the segmentation mask supplied the arm axis; and the unchanged measurement path produced 182.213 mm. The configured 70 to 180 mm validity gate rejected that estimate, and no risk category was assigned.

This smoke request did not produce a valid MUAC measurement. It confirms the model and endpoint path execute and that the validity guard rejects the unsuitable estimate. It does not establish measurement accuracy or clinical reliability.

## Tests and limits

- ML-service suite in the isolated runtime: **117 passed**, with one upstream Starlette/httpx deprecation warning.
- The test that checks `/segment`'s unavailable-model response now explicitly selects legacy mode, so its result no longer depends on whether a real checkpoint happens to be installed on the test machine.
- A Docker daemon was unavailable locally (`com.docker.service` was stopped), so the production image itself could not be built in this environment. The Dockerfile now includes a build-time checksum and model-load check; a Render/Docker build remains the final container-level verification.
- No training, dataset split, annotation, MUAC formula, threshold, API schema, or frontend behavior was changed.
