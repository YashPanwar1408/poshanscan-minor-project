# PoshanScan ML Runtime Verification

**Verified:** 2026-11-24
**Scope:** production Docker image build, checkpoint load, and real HTTP smoke requests. This is integration evidence, not model-accuracy or clinical validation.

## Runtime issue and correction

The earlier `/infer` smoke request returned HTTP 503 because the active ML runtime could not import Ultralytics. Installing a package into the Windows Python environment would not change the dependencies inside the production container. The production service in `render.yaml` uses `ml-service/Dockerfile`.

The Dockerfile already installed `requirements-segmentation.txt`, but that file only pinned Ultralytics, leaving PyTorch and torchvision to be resolved indirectly. I pinned the verified framework versions and made the Docker build install the CPU wheels directly from the official PyTorch CPU index before installing the segmentation requirements. The build now imports the real checkpoint and asserts its SHA-256, task and class mapping; a bad or incomplete image fails at build time. `ml-service/.dockerignore` also keeps datasets, local virtual environments and generated results out of the Docker context.

The currently available artifacts do not include the exact image ID or build log for the container that returned 503, so it is not possible to prove whether it was an old image or another runtime configuration. The corrected production image below was built and tested from the current repository source.

## Checkpoint and container

| Check | Result |
|---|---|
| Checkpoint | `ml-service/app/models/arm_segmentation/best.pt` exists (7,012,916 bytes) |
| SHA-256 | `dc42cf4f893051abeb6faca2bef0f0c56f44f8182b12f2ec904eec71a2a026b1` (matches expected) |
| Model metadata | YOLO task `segment`; class map `{0: upper_arm}` |
| Container runtime | Ultralytics `8.4.216`; torch `2.15.0+cpu`; torchvision `0.29.2+cpu` |
| Device | CPU; `torch.cuda.is_available()` returned `False` |
| Image built | `poshanscan-ml:runtime-0264589`, image size 765,973,494 bytes |
| Startup | YOLO checkpoint loaded once on CPU; `/health` returned HTTP 200 |

The checkpoint bytes were not changed. The Docker image itself validates the expected digest, task, and class mapping during build.

## Real HTTP smoke requests

The existing project image `ml-service/dataset/images/10.png` was uploaded to both endpoints. It was not changed.

- `POST /segment?imgsz=416&confidence=0.25`: HTTP 200; `detected=true`; one upper-arm instance. Source, response metadata and decoded PNG mask were all 1086 by 1448 pixels. The startup log explicitly says `YOLO ... on cpu`, so this was not the legacy fallback.
- `POST /infer` with `age_category=child_6_59m`, ArUco reference type and a 50 mm marker: HTTP 422 `invalid_measurement`. The pipeline detected ArUco ID 0, did not find MediaPipe landmarks, used its marker fallback, and logged a segmentation axis of 97.41 degrees with a 131 px sample width. The existing formula produced 182.213 mm, above the configured 70–180 mm validity range. The response assigned no MUAC value or risk band. This confirms the pipeline and safety gate ran; it does not establish a valid measurement.

Container logs showed the model loaded at startup and both actual HTTP requests reached the FastAPI app. This sample remains unsuitable for a valid measurement smoke result.

## Tests and limitations

- ML-service suite: **117 passed**, with one upstream Starlette/httpx deprecation warning. Pytest required `--basetemp .pytest-temp -p no:cacheprovider` because the default Windows user Temp directory denied access.
- Container dependency, checkpoint-load, startup, `/health`, `/segment`, and `/infer` checks: passed as described above.
- The original 503 was caused by a runtime missing Ultralytics. Its exact old image provenance is unavailable, so the report does not claim which old build setting produced it.
- No retraining, dataset split, annotation, MUAC formula, clinical threshold, API schema or frontend behavior was changed.
- These checks do not demonstrate real-world segmentation quality, tape agreement, MUAC accuracy, or clinical reliability.
