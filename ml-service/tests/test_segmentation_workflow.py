"""Tests for annotation gating, group-safe splits, and optional YOLO fallback."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.pipeline import segment
from app.pipeline import run as run_module
from scripts.evaluate_arm_segmentation import (
    binary_mask_metrics,
    binary_metrics_from_counts,
    main as evaluate_main,
)
from scripts.export_cvat_unique_images import export_unique_images
from scripts.import_cvat_annotations import import_cvat_archive, parse_class_names
from scripts.split_dataset import create_splits, discover_duplicate_groups
from scripts.train_arm_segmentation import main as train_main
from scripts.validate_annotations import validate_annotations, validate_polygon_line


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    assert ok
    path.write_bytes(encoded.tobytes())


def test_yolo_polygon_line_validation() -> None:
    assert validate_polygon_line("0 0.1 0.1 0.9 0.1 0.5 0.9", 1) == (True, "")
    valid, error = validate_polygon_line("0 0 0 1.2 0 0.5 1", 2)
    assert not valid and "[0, 1]" in error
    valid, error = validate_polygon_line("0 0.2 0.2 0.2 0.2 0.2 0.2", 3)
    assert not valid and "distinct points" in error
    valid, error = validate_polygon_line("0 0.2 0.2 0.4", 4)
    assert not valid and "pairs" in error


def test_annotation_validation_reports_missing_and_valid_labels(tmp_path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    _write_image(images / "1.png", np.zeros((32, 32, 3), dtype=np.uint8))
    _write_image(images / "2.jpg", np.zeros((32, 32, 3), dtype=np.uint8))
    labels.mkdir(parents=True)
    (labels / "1.txt").write_text("0 0.1 0.1 0.8 0.1 0.5 0.9\n", encoding="utf-8")

    summary, rows = validate_annotations(images, labels)
    assert summary["image_count"] == 2
    assert summary["valid_annotated_images"] == 1
    assert summary["missing_labels"] == 1
    assert not summary["training_ready"]
    assert {row["status"] for row in rows} == {"valid", "missing_label"}


def test_split_annotation_validation_ignores_preserved_exact_duplicate_alias(tmp_path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    splits = tmp_path / "splits"
    for name in ("1.png", "2.png", "3.png", "4.png"):
        _write_image(images / name, np.zeros((32, 32, 3), dtype=np.uint8))
    labels.mkdir(parents=True)
    for name in ("1.txt", "2.txt", "3.txt"):
        (labels / name).write_text("0 0.1 0.1 0.8 0.1 0.5 0.9\n", encoding="utf-8")
    splits.mkdir()
    (splits / "train.txt").write_text("images/1.png\n", encoding="utf-8")
    (splits / "val.txt").write_text("images/2.png\n", encoding="utf-8")
    (splits / "test.txt").write_text("images/3.png\n", encoding="utf-8")

    summary, rows = validate_annotations(images, labels, splits)
    assert summary["training_ready"]
    assert summary["split_image_count"] == 3
    assert summary["images_outside_current_split_without_labels"] == 1
    assert next(row for row in rows if row["image_filename"] == "4.png")["status"] == "excluded_from_current_split"


def test_unique_cvat_export_contains_only_canonical_image_files(tmp_path) -> None:
    images = tmp_path / "images"
    _write_image(images / "1.png", np.zeros((32, 32, 3), dtype=np.uint8))
    _write_image(images / "2.png", np.full((32, 32, 3), 255, dtype=np.uint8))
    inventory = tmp_path / "unique.json"
    inventory.write_text(json.dumps([{"canonical_filename": "1.png"}]), encoding="utf-8")
    output = tmp_path / "cvat_images.zip"
    assert export_unique_images(images, inventory, output) == 1
    with ZipFile(output) as archive:
        assert archive.namelist() == ["1.png"]


def test_cvat_import_requires_single_upper_arm_class_and_preserves_images(tmp_path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    source_image = np.full((40, 40, 3), 127, dtype=np.uint8)
    _write_image(images / "1.png", source_image)
    original_bytes = (images / "1.png").read_bytes()
    archive_path = tmp_path / "cvat_export.zip"
    with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("data.yaml", "names:\n  0: upper_arm\n")
        archive.writestr("labels/train/1.txt", "0 0.1 0.1 0.9 0.1 0.5 0.9\n")

    result = import_cvat_archive(archive_path, images, labels)
    assert result["imported_label_files"] == 1
    assert (labels / "1.txt").read_text(encoding="utf-8") == "0 0.1 0.1 0.9 0.1 0.5 0.9\n"
    assert (images / "1.png").read_bytes() == original_bytes
    assert parse_class_names("names: [upper_arm]\n") == ["upper_arm"]


def test_cvat_import_rejects_wrong_class_without_copying_any_label(tmp_path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    _write_image(images / "1.png", np.zeros((32, 32, 3), dtype=np.uint8))
    archive_path = tmp_path / "wrong_class.zip"
    with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("data.yaml", "names:\n  0: person\n")
        archive.writestr("labels/train/1.txt", "0 0.1 0.1 0.9 0.1 0.5 0.9\n")
    with pytest.raises(ValueError, match="exactly one class named upper_arm"):
        import_cvat_archive(archive_path, images, labels)
    assert not labels.exists()


def test_duplicate_groups_stay_in_same_reproducible_split(tmp_path) -> None:
    images = tmp_path / "images"
    split_a = tmp_path / "splits-a"
    split_b = tmp_path / "splits-b"
    rng = np.random.default_rng(1)
    source = rng.integers(0, 255, size=(100, 100, 3), dtype=np.uint8)
    _write_image(images / "1.png", source)
    _write_image(images / "2.png", source.copy())  # exact duplicate
    near = source.copy()
    near[-1, -1] = 255 - near[-1, -1]
    _write_image(images / "3.png", near)  # near duplicate, distinct file bytes
    for index in range(4, 14):
        _write_image(images / f"{index}.png", rng.integers(0, 255, size=(100, 100, 3), dtype=np.uint8))

    files = sorted(images.glob("*.png"), key=lambda path: int(path.stem))
    groups, pairs = discover_duplicate_groups(files, near_threshold=4)
    duplicate_group = next(group for group in groups if len(group) == 3)
    assert {path.name for path in duplicate_group} == {"1.png", "2.png", "3.png"}
    assert any(pair["exact_duplicate"] for pair in pairs)
    assert any(pair["first"] == "1.png" and pair["second"] == "3.png" and not pair["exact_duplicate"] for pair in pairs)

    report_a, splits_a = create_splits(images, split_a, seed=44, near_threshold=4)
    report_b, splits_b = create_splits(images, split_b, seed=44, near_threshold=4)
    name_to_split = {path.name: split for split, paths in splits_a.items() for path in paths}
    assert "2.png" not in name_to_split  # Exact copy remains on disk but is omitted from splits.
    assert name_to_split["1.png"] == name_to_split["3.png"]
    assert report_a["excluded_exact_duplicate_count"] == 1
    assert report_a["no_duplicate_or_near_duplicate_group_leakage"]
    assert report_a["group_leakage_pairs"] == []
    assert sum(report_a["split_counts"].values()) == 12
    assert {key: [path.name for path in value] for key, value in splits_a.items()} == {
        key: [path.name for path in value] for key, value in splits_b.items()
    }
    assert report_a["split_counts"] == report_b["split_counts"]


def test_training_stops_when_masks_are_missing(capsys, monkeypatch) -> None:
    import scripts.train_arm_segmentation as training_module

    monkeypatch.setattr(
        training_module,
        "validate_annotations",
        lambda **_kwargs: (
            {
                "training_ready": False,
                "image_count": 2,
                "valid_annotated_images": 1,
                "missing_labels": 1,
                "invalid_annotations_or_orphan_labels": 0,
            },
            [],
        ),
    )
    result = train_main([])
    captured = capsys.readouterr()
    assert result == 2
    assert "Training stopped" in captured.err
    assert "No masks or labels were generated automatically" in captured.err


def test_evaluation_validates_active_splits_before_checkpoint(monkeypatch, capsys, tmp_path) -> None:
    import scripts.evaluate_arm_segmentation as evaluation_module

    def fake_validation(*, split_dir):
        assert split_dir == evaluation_module.ROOT / "dataset" / "splits"
        return {"training_ready": True, "valid_annotated_images": 120, "image_count": 120}, []

    monkeypatch.setattr(evaluation_module, "validate_annotations", fake_validation)
    missing_weights = tmp_path / "not-trained.pt"
    result = evaluate_main(["--weights", str(missing_weights)])
    captured = capsys.readouterr()
    assert result == 2
    assert "trained checkpoint not found" in captured.err


def test_segment_api_returns_polygon_mask_and_confidence(monkeypatch) -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    class FakeBoxes:
        cls = np.array([0])
        conf = np.array([0.91])
        xyxy = np.array([[8.0, 7.0, 48.0, 56.0]])

    class FakeMasks:
        xy = [np.array([[8, 7], [48, 7], [48, 56], [8, 56]], dtype=np.float32)]

    class FakeResult:
        boxes = FakeBoxes()
        masks = FakeMasks()

    class FakeModel:
        def predict(self, **_kwargs):
            return [FakeResult()]

    # This test exercises the explicit model-unavailable response; isolate it
    # from a real local checkpoint that may be installed in the test runtime.
    monkeypatch.setenv("POSHANSCAN_SEGMENTATION_MODEL", "legacy")
    with TestClient(main_module.app) as client:
        unavailable = client.post("/segment", files={"image": ("new.png", encoded.tobytes(), "image/png")})
        assert unavailable.status_code == 503

        monkeypatch.setattr(segment, "_YOLO_MODEL", FakeModel())
        monkeypatch.setattr(segment, "_SEGMENTATION_MODE", "yolo")
        monkeypatch.setattr(segment, "_mask_predictor_class", lambda _threshold: object())
        response = client.post("/segment", files={"image": ("new.png", encoded.tobytes(), "image/png")})

    assert response.status_code == 200
    body = response.json()
    assert body["detected"] is True
    assert body["instances"][0]["class_name"] == "upper_arm"
    assert body["instances"][0]["confidence"] == pytest.approx(0.91)
    assert body["instances"][0]["bbox_xyxy"] == [8.0, 7.0, 48.0, 56.0]
    assert len(body["instances"][0]["polygon"]) == 4
    mask_bytes = base64.b64decode(body["mask_png_base64"])
    mask = cv2.imdecode(np.frombuffer(mask_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    assert mask is not None and mask.shape == (64, 64) and np.any(mask)


def test_verified_checkpoint_sha256_matches_supplied_digest() -> None:
    checkpoint = Path(__file__).resolve().parents[1] / "app" / "models" / "arm_segmentation" / "best.pt"
    assert checkpoint.is_file()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == "dc42cf4f893051abeb6faca2bef0f0c56f44f8182b12f2ec904eec71a2a026b1"


def test_full_image_yolo_inference_preserves_dimensions_and_uses_cpu_fallback(monkeypatch, tmp_path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"mock")
    image = np.full((64, 96, 3), 90, dtype=np.uint8)
    mask_data = np.zeros((1, 32, 48), dtype=np.float32)
    mask_data[0, 5:25, 10:38] = 0.8
    predict_kwargs = {}

    class FakeModel:
        task = "segment"
        names = {0: "upper_arm"}

        def predict(self, **kwargs):
            predict_kwargs.update(kwargs)
            return [SimpleNamespace(
                masks=SimpleNamespace(
                    data=mask_data,
                    xy=[np.array([[20, 10], [76, 10], [76, 50], [20, 50]], dtype=np.float32)],
                ),
                boxes=SimpleNamespace(
                    cls=np.array([0]),
                    conf=np.array([0.93]),
                    xyxy=np.array([[20, 10, 76, 50]], dtype=np.float32),
                ),
            )]

    monkeypatch.setattr(segment, "_load_yolo_model", lambda _path: FakeModel())
    monkeypatch.setattr(segment, "_select_yolo_device", lambda: "cpu")
    monkeypatch.setattr(segment, "_mask_predictor_class", lambda _threshold: object())
    assert segment.configure_segmentation_model("yolo", weights)
    prediction = segment.predict_yolo_image(image)
    assert prediction["mask"].shape == image.shape[:2]
    assert prediction["visualization"].shape == image.shape
    assert prediction["detected"] is True
    assert prediction["instances"][0]["class_name"] == "upper_arm"
    assert predict_kwargs["classes"] == [0]
    assert predict_kwargs["retina_masks"] is True
    assert predict_kwargs["device"] == "cpu"
    raised_mask_threshold = segment.predict_yolo_image(image, mask_threshold=0.9)
    assert raised_mask_threshold["detected"] is False
    assert segment.configure_segmentation_model("legacy")


def test_empty_and_invalid_yolo_inference_are_explicit(monkeypatch, tmp_path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"mock")

    class FakeModel:
        task = "segment"
        names = {0: "upper_arm"}

        def predict(self, **_kwargs):
            return [SimpleNamespace(masks=None, boxes=None)]

    monkeypatch.setattr(segment, "_load_yolo_model", lambda _path: FakeModel())
    monkeypatch.setattr(segment, "_select_yolo_device", lambda: "cpu")
    monkeypatch.setattr(segment, "_mask_predictor_class", lambda _threshold: object())
    assert segment.configure_segmentation_model("yolo", weights)
    empty = segment.predict_yolo_image(np.zeros((48, 64, 3), dtype=np.uint8))
    assert empty["detected"] is False
    assert empty["mask"].shape == (48, 64)
    assert not np.any(empty["mask"])
    with pytest.raises(ValueError, match="image"):
        segment.predict_yolo_image(np.array([]))
    assert segment.configure_segmentation_model("legacy")


def test_yolo_mask_is_integrated_with_aruco_exclusion(monkeypatch, tmp_path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"mock")
    full_arm = np.zeros((400, 400), dtype=np.uint8)
    full_arm[50:350, 150:250] = 1
    polygon = np.array([[150, 50], [249, 50], [249, 349], [150, 349]], dtype=np.float32)

    class FakeModel:
        task = "segment"
        names = {0: "upper_arm"}

        def predict(self, **_kwargs):
            return [SimpleNamespace(
                masks=SimpleNamespace(data=full_arm[None, ...], xy=[polygon]),
                boxes=SimpleNamespace(cls=np.array([0]), conf=np.array([0.96])),
            )]

    monkeypatch.setattr(segment, "_load_yolo_model", lambda _path: FakeModel())
    monkeypatch.setattr(segment, "_select_yolo_device", lambda: "cpu")
    monkeypatch.setattr(segment, "_mask_predictor_class", lambda _threshold: object())
    assert segment.configure_segmentation_model("yolo", weights)
    corners = np.array([[180, 180], [220, 180], [220, 220], [180, 220]], dtype=np.float32)
    monkeypatch.setattr(run_module, "detect_reference_marker", lambda *_args: {
        "detected": True,
        "scale_px_per_mm": 4.0,
        "confidence": 0.98,
        "marker_id": 7,
        "marker_corners": corners,
    })
    monkeypatch.setattr(run_module, "locate_either_arm", lambda _image: {
        "detected": True,
        "midpoint_px": (200.0, 200.0),
        "shoulder_px": (200.0, 50.0),
        "elbow_px": (200.0, 350.0),
        "arm_angle_degrees": 90.0,
        "confidence": 0.95,
    })

    marker_observation = {}
    original_remove = run_module.remove_polygon_from_mask

    def observe_marker_removal(mask, marker_corners, origin):
        marker_observation["before"] = int(mask[140, 140])
        cleaned = original_remove(mask, marker_corners, origin)
        marker_observation["after"] = int(cleaned[140, 140])
        return cleaned

    monkeypatch.setattr(run_module, "remove_polygon_from_mask", observe_marker_removal)
    result = run_module.run_muac_pipeline(np.zeros((400, 400, 3), dtype=np.uint8), 10.0, "child_6_59m")
    assert result["ok"] is True
    assert marker_observation == {"before": 255, "after": 0}
    assert segment.configure_segmentation_model("legacy")


def test_yolo_model_is_loaded_once_and_invalid_prediction_falls_back(monkeypatch, tmp_path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"test placeholder; loader is mocked")
    load_count = {"value": 0}

    class FakeModel:
        task = "segment"
        names = {0: "upper_arm"}

        def predict(self, **_kwargs):
            load_count["inference"] += 1
            empty_masks = np.zeros((1, 200, 200), dtype=np.float32)
            return [SimpleNamespace(
                masks=SimpleNamespace(
                    data=empty_masks,
                    xy=[np.array([[0, 0], [239, 0], [239, 239], [0, 239]], dtype=np.float32)],
                ),
                boxes=SimpleNamespace(cls=np.array([0]), conf=np.array([0.95])),
            )]

    def fake_loader(_path):
        load_count["value"] += 1
        return FakeModel()

    load_count["inference"] = 0
    monkeypatch.setattr(segment, "_load_yolo_model", fake_loader)
    monkeypatch.setattr(segment, "_mask_predictor_class", lambda _threshold: object())
    assert segment.configure_segmentation_model("yolo", weights)
    image = np.full((240, 240, 3), 255, dtype=np.uint8)
    output = segment.segment_arm(image, (120, 120), crop_size_px=200)
    assert load_count["value"] == 1
    assert load_count["inference"] == 1
    assert output["segmentation_method"] == "legacy_fallback"
    assert output["segmentation_confidence"] == 0.0
    assert segment.configure_segmentation_model("legacy")


def test_valid_yolo_mask_is_used_and_empty_image_is_safe(monkeypatch, tmp_path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"mock")

    class FakeModel:
        task = "segment"
        names = {0: "upper_arm"}

        def predict(self, **_kwargs):
            mask = np.zeros((2, 240, 240), dtype=np.float32)
            mask[0, 20:220, 80:110] = 1.0
            mask[1, 20:220, 131:161] = 1.0
            return [SimpleNamespace(
                masks=SimpleNamespace(
                    data=mask,
                    xy=[
                        np.array([[80, 20], [109, 20], [109, 219], [80, 219]], dtype=np.float32),
                        np.array([[131, 20], [160, 20], [160, 219], [131, 219]], dtype=np.float32),
                    ],
                ),
                boxes=SimpleNamespace(cls=np.array([0, 0]), conf=np.array([0.9, 0.85])),
            )]

    monkeypatch.setattr(segment, "_load_yolo_model", lambda _path: FakeModel())
    monkeypatch.setattr(segment, "_mask_predictor_class", lambda _threshold: object())
    assert segment.configure_segmentation_model("yolo", weights)
    image = np.full((240, 240, 3), 255, dtype=np.uint8)
    output = segment.segment_arm(image, (120, 120), crop_size_px=200)
    assert output["segmentation_method"] == "yolo"
    assert output["segmentation_confidence"] == pytest.approx(0.9)
    assert np.count_nonzero(output["mask"]) == 200 * 60
    assert output["mask"][100, 100] == 0  # simulated marker gap at the MUAC midpoint
    empty = segment.segment_arm(np.array([]), (0, 0), crop_size_px=200)
    assert empty["mask"].shape == (200, 200)
    segment.configure_segmentation_model("legacy")


def test_yolo_missing_checkpoint_wrong_task_or_class_use_legacy(monkeypatch, tmp_path) -> None:
    assert not segment.configure_segmentation_model("yolo", tmp_path / "missing.pt")
    assert segment.segmentation_mode() == "legacy"
    weights = tmp_path / "detect.pt"
    weights.write_bytes(b"mock")
    monkeypatch.setattr(segment, "_load_yolo_model", lambda _path: SimpleNamespace(task="detect"))
    assert not segment.configure_segmentation_model("yolo", weights)
    assert segment.segmentation_mode() == "legacy"

    monkeypatch.setattr(segment, "_load_yolo_model", lambda _path: SimpleNamespace(
        task="segment",
        names={0: "person"},
    ))
    assert not segment.configure_segmentation_model("yolo", weights)
    assert segment.segmentation_mode() == "legacy"


def test_binary_mask_metrics_are_correct() -> None:
    truth = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    pred = np.array([[1, 0, 1], [0, 0, 0]], dtype=np.uint8)
    metrics = binary_mask_metrics(pred, truth)
    assert metrics["iou"] == pytest.approx(1 / 3)
    assert metrics["dice"] == pytest.approx(0.5)
    aggregate = binary_metrics_from_counts(1, 1, 1)
    assert aggregate["iou"] == pytest.approx(1 / 3)


def test_fastapi_startup_loads_yolo_once_and_shutdown_restores_legacy(monkeypatch, tmp_path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"mock checkpoint")
    load_count = {"value": 0}

    class FakeModel:
        task = "segment"
        names = {0: "upper_arm"}

    def fake_loader(_path):
        load_count["value"] += 1
        return FakeModel()

    monkeypatch.setattr(segment, "_load_yolo_model", fake_loader)
    monkeypatch.setattr(segment, "_mask_predictor_class", lambda _threshold: object())
    monkeypatch.setenv("POSHANSCAN_SEGMENTATION_MODEL", "yolo")
    monkeypatch.setenv("POSHANSCAN_SEGMENTATION_WEIGHTS", str(weights))
    monkeypatch.delenv("POSHANSCAN_ENABLE_SYNTHETIC_CALIBRATION", raising=False)

    with TestClient(main_module.app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert segment.segmentation_mode() == "yolo"
        assert load_count["value"] == 1
    assert segment.segmentation_mode() == "legacy"
