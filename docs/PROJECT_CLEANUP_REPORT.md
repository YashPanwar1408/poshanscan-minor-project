# PoshanScan project cleanup report

**Audit and cleanup date:** 24 November 2026

**Project:** `D:\minor project complete\poshanscan`

## Audit inventory and disposition

| Classification | Audited paths / examples | Disposition |
|---|---|---|
| Essential application source | `backend/app/`, `frontend/src/`, `ml-service/app/`, `render.yaml`, root and service configuration | Retained. Existing paths and imports were not moved. |
| Trained model and runtime | `ml-service/app/models/arm_segmentation/best.pt`, its `.sha256` record, `ml-service/models/upper_arm/best.pt`, segmentation pipeline and requirements | Retained. Both checkpoint copies were hashed and are byte-identical by SHA-256. No checkpoint bytes were changed. |
| Research and reproducibility | `ml-service/dataset/`, `results/`, `sample_images/`, annotation ZIPs, notebooks, reports, manual tape-reference template, validation/evaluation scripts | Retained. No split, annotation, original image, report, or research artifact was regenerated or removed. |
| Tests and developer tooling | Backend, frontend, and ML tests; ML audit, conversion, validation, and training scripts; `frontend/node_modules/` | Retained. Dependency folders remain because removing them would require a separately verified restoration. |
| Configuration and deployment | `.git/`, `render.yaml`, Dockerfiles, package manifests/lockfile, requirements files, local environment-file paths | Retained. `.env` contents were not opened. No credentials were changed or copied into these reports. |
| User/application data | Root and backend `uploads/`, `backend/poshanscan.db`, service logs | Retained without cleanup. These may contain user data or useful diagnostic evidence. |
| Generated cache/build output | `frontend/dist/`, 11 Python `__pycache__/` directories containing only `.pyc` files, temporary pytest output | Removed only after exact-path and contents checks; details below. |
| Uncertain/inaccessible cache | `backend/.pytest_cache/`, `ml-service/.pytest_cache/` | Preserved. Windows returned `Access is denied` while enumerating them, so their contents could not be verified for safe deletion. |
| Dependencies | `frontend/package.json`, `package-lock.json`, backend and ML requirements | No dependencies or versions were removed or changed. |

The top-level structure remains `backend/`, `frontend/`, `ml-service/`,
`docs/`, `uploads/`, `.git/`, and the root documentation/configuration. No code or
research directory was renamed or relocated. The working tree already contained
uncommitted backend and ML changes from earlier PoshanScan work; those changes
were left intact.

The final filesystem inventory counted 381 files in `backend/`, 40,903 in
`frontend/` (including retained `node_modules/`), 1,558 in `ml-service/`, and
2 reports in `docs/`. `uploads/` at the repository root was empty.
These metadata counts do not include the contents of protected `.pytest_cache/`
directories, which could not be enumerated; `.git/` is preserved separately.

## Cleanup actions

- Removed the freshly generated, Git-ignored `frontend/dist/` after a
  successful production build: 52 files, 5,201,930 bytes. It can be recreated
  with `npm run build`.
- Removed 154 `.pyc` files from 11 verified `__pycache__/` directories. Each
  directory contained only bytecode cache files.
- Removed the pytest scratch trees created for this verification run:
  `ml-service/.pytest-run-tmp/` and
  `D:\minor project complete\_poshanscan_backend_test_tmp_20261124/`. The
  backend test scratch tree was created empty for this run and held only
  temporary test uploads and pytest files.
- Removed the historical pytest fixture-only trees
  `ml-service/results/cvat_import_audit_execution_check/pytest_tmp/` and
  `ml-service/results/cvat_import_audit_execution_check/pytest_mlservice/`:
  66 generated files (about 405 KB) in each. Their
  `*current` symlinks all resolved inside the same tree; contents were only
  synthetic test fixtures/results; no source or report outside those trees
  referenced them. The enclosing audit report, staging exports, review
  materials, and other results were retained.
- No application source, checkpoint, database, upload, annotation, dataset,
  staging export, or other research report was deleted. No files were moved.
- Kept the two identical `best.pt` copies because the runtime path and source
  checkpoint path have separate documented roles. Their SHA-256 values both
  equal `dc42cf4f893051abeb6faca2bef0f0c56f44f8182b12f2ec904eec71a2a026b1`.

## Model integrity

The required runtime checkpoint was hashed before and after cleanup at
`ml-service/app/models/arm_segmentation/best.pt`.

```text
Expected: dc42cf4f893051abeb6faca2bef0f0c56f44f8182b12f2ec904eec71a2a026b1
Before:   dc42cf4f893051abeb6faca2bef0f0c56f44f8182b12f2ec904eec71a2a026b1
After:    dc42cf4f893051abeb6faca2bef0f0c56f44f8182b12f2ec904eec71a2a026b1
Sidecar:  matches the expected hash and filename
```

The source copy at `ml-service/models/upper_arm/best.pt` has the same SHA-256
and size (7,012,916 bytes). Neither copy was renamed, replaced, or edited.

## Duplicate audit

A SHA-256 scan examined 1,729 readable project files outside `node_modules/`,
`.git/`, cache directories, local environment files, databases, logs, and
uploads. It found 283 exact-duplicate groups after the two pytest fixture trees
were removed. The identified patterns include 120 annotation-to-`yolo_working`
label mirrors, 31 pairs across CVAT staging/execution-check results, and the
two required checkpoint copies. Other exact copies are in generated results
or lack enough provenance to prove deletion safe. They were retained. This
scan did not enumerate the protected `.pytest_cache/` directories.

## Verification results

- Frontend: 44 tests passed; TypeScript check passed; production build passed;
  lint passed with 0 errors and 17 Fast Refresh warnings.
- Backend: 39 tests passed. The tests ran against isolated SQLite and upload
  paths outside the project. A working Python 3.12 interpreter was used with
  existing pure-Python packages available in the Anaconda package directory;
  no packages were installed or changed.
- ML service: 117 tests passed with an isolated temporary directory. These
  tests verify application logic and mocked model loading/inference paths; they
  do not prove that the real checkpoint can be loaded in this machine's active
  runtime.
- A live local FastAPI process started in legacy fallback mode. `/health`
  returned HTTP 200; an uploaded image to `/segment` returned HTTP 503 because
  the YOLO model did not load. A request without an image returned HTTP 422.
  The service process was stopped after these checks.
- `git diff --check` exited successfully. Git emitted only LF/CRLF normalization
  notices for existing modified files.
- No real-photo accuracy metrics were generated. The real-image validation
  workflow is documented in `ml-service/REAL_IMAGE_VALIDATION.md` and remains
  pending photographs and manual tape references.

## Runtime verification limits

The configured Anaconda environment is Python 3.13.5, but its
`D:\anaconda3\python.exe` exits before producing Python output with
NTSTATUS `0xC0000012` (`STATUS_INVALID_DEVICE_REQUEST`). Its file ACL grants
read/execute access, so this is not established as a file-permission denial.
The project-level `venv` executable terminates with the same status, and
`py -0p` reports no registered Python installations. A separate Python 3.12.3
works, but its environment lacks the Ultralytics and torchvision packages
needed for a real YOLO checkpoint load. The ML tests pass without those
packages because model-loading cases use controlled test doubles.

Docker configuration was inspected, but the Docker engine named pipe was not
available, so a container build could not be performed. The local service
check verified startup and health only in its legacy fallback mode. The
checkpoint checksum and mocked integration tests pass, while real model
loading, successful YOLO `/segment`/`/infer` requests, and backend-to-ML network
connectivity remain unverified in this environment.

## Remaining items

1. Repair or select a working Python environment for the ML service and install
   the pinned optional segmentation requirements in that environment.
2. Start the ML service and verify a real checkpoint load, `/health`, `/segment`,
   and `/infer` with an approved test photograph; then verify the backend's
   configured connection to that service.
3. Run the documented real-image workflow with consented photographs and
   independently measured tape references before making any empirical or
   clinical-performance claim.
