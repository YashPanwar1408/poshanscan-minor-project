# PoshanScan upper-arm segmentation checkpoint

`best.pt` is the YOLOv8n-seg checkpoint installed for FastAPI inference.

- Task: `segment`
- Class mapping: `0: upper_arm`
- SHA-256: `dc42cf4f893051abeb6faca2bef0f0c56f44f8182b12f2ec904eec71a2a026b1`
- Source: `ml-service/models/upper_arm/best.pt` (copied byte-for-byte after hash verification)
- Runtime verified: Ultralytics `8.4.216`, PyTorch `2.15.0+cpu`, torchvision `0.29.2+cpu` on Python 3.12.3; CPU inference.

The model is loaded once during FastAPI startup. Install the base service requirements and the pinned `requirements-segmentation.txt` package to enable YOLO locally. The production Dockerfile installs these dependencies and validates the checkpoint at image build time. Set `POSHANSCAN_SEGMENTATION_MODEL=legacy` only to select the HSV + GrabCut fallback explicitly.

Inference settings are controlled by `POSHANSCAN_YOLO_CONFIDENCE` (default `0.25`), `POSHANSCAN_YOLO_MASK_THRESHOLD` (default `0.5`), `POSHANSCAN_YOLO_MIN_MASK_COVERAGE` (default `0.005`), `POSHANSCAN_YOLO_MAX_MASK_COVERAGE` (default `0.95`), and `POSHANSCAN_YOLO_IMAGE_SIZE` (default `640`). CUDA is used when PyTorch reports it available; otherwise inference runs on CPU.

The mask feeds the existing ArUco removal, pose/geometric width measurement, and MUAC calculation. The checkpoint predicts an arm mask, not MUAC. The inference smoke tests are not clinical validation. See [`docs/ML_RUNTIME_VERIFICATION_REPORT.md`](../../../../docs/ML_RUNTIME_VERIFICATION_REPORT.md) for the current runtime checks and limitations.
