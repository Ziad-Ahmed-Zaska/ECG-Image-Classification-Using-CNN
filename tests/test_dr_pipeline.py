"""
Unit tests for the DR multi-lesion staged learning pipeline.

Tests cover model construction, forward passes, loss computation,
metric calculation, and dataset manifest logic.  They use synthetic
data so no real datasets are needed.
"""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

def test_config_constants():
    from dr_pipeline.config import (
        DR_GRADES,
        GLOBAL_VIEW_SIZE,
        LESION_CLASSES,
        NUM_DR_GRADES,
        NUM_LESION_CLASSES,
    )
    assert NUM_LESION_CLASSES == 4
    assert NUM_DR_GRADES == 5
    assert len(LESION_CLASSES) == NUM_LESION_CLASSES
    assert len(DR_GRADES) == NUM_DR_GRADES
    assert GLOBAL_VIEW_SIZE == (512, 512)


def test_ensure_dirs(tmp_path, monkeypatch):
    from dr_pipeline import config
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(config, "CHECKPOINT_DIR", tmp_path / "out" / "ckpt")
    monkeypatch.setattr(config, "LOG_DIR", tmp_path / "out" / "log")
    monkeypatch.setattr(config, "FIGURE_DIR", tmp_path / "out" / "fig")
    config.ensure_dirs()
    assert (tmp_path / "out" / "ckpt").exists()
    assert (tmp_path / "out" / "log").exists()
    assert (tmp_path / "out" / "fig").exists()


# ---------------------------------------------------------------------------
# Dataset manifest tests
# ---------------------------------------------------------------------------

class TestDatasetManifest:
    def _write_csv(self, path, rows):
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    def test_patient_level_splits(self, tmp_path):
        from dr_pipeline.datasets.dataset_manifest import DatasetManifest

        # 10 patients, 2 images each
        rows = []
        for pid in range(10):
            for eye in ("L", "R"):
                rows.append({
                    "image_path": f"img_{pid}_{eye}.jpg",
                    "patient_id": str(pid),
                    "label": "no_dr" if pid < 5 else "mild",
                })

        csv_path = tmp_path / "meta.csv"
        self._write_csv(csv_path, rows)

        m = DatasetManifest(
            dataset_name="test",
            root_dir=str(tmp_path),
            train_ratio=0.6,
            val_ratio=0.2,
        )
        m.load_metadata(str(csv_path))
        m.build_patient_splits()

        train = m.get_split("train")
        val = m.get_split("val")
        test = m.get_split("test")

        # No patient appears in multiple splits
        train_pids = {r["patient_id"] for r in train}
        val_pids = {r["patient_id"] for r in val}
        test_pids = {r["patient_id"] for r in test}
        assert train_pids.isdisjoint(val_pids)
        assert train_pids.isdisjoint(test_pids)
        assert val_pids.isdisjoint(test_pids)

        # All images accounted for
        assert len(train) + len(val) + len(test) == 20

    def test_save_and_load(self, tmp_path):
        from dr_pipeline.datasets.dataset_manifest import DatasetManifest

        rows = [
            {"image_path": f"img_{i}.jpg", "patient_id": str(i), "label": "no_dr"}
            for i in range(6)
        ]
        csv_path = tmp_path / "meta.csv"
        self._write_csv(csv_path, rows)

        m = DatasetManifest(dataset_name="test", root_dir=str(tmp_path))
        m.load_metadata(str(csv_path))
        m.build_patient_splits()

        out_dir = tmp_path / "manifest_out"
        m.save(str(out_dir))

        assert (out_dir / "manifest.json").exists()
        assert (out_dir / "train.csv").exists()

        loaded = DatasetManifest.load(str(out_dir))
        assert loaded.dataset_name == "test"


# ---------------------------------------------------------------------------
# Model tests (forward pass with synthetic tensors)
# ---------------------------------------------------------------------------

class TestEncoder:
    def test_forward_shape(self):
        from dr_pipeline.models.encoder import DREncoder
        enc = DREncoder(backbone_name="resnet18", pretrained_imagenet=False)
        x = torch.randn(2, 3, 224, 224)
        out = enc(x)
        assert out.shape == (2, 512)

    def test_forward_features(self):
        from dr_pipeline.models.encoder import DREncoder
        enc = DREncoder(backbone_name="resnet18", pretrained_imagenet=False)
        x = torch.randn(1, 3, 224, 224)
        feats = enc.forward_features(x)
        assert "s1" in feats
        assert "s4" in feats
        assert "pool" in feats
        assert feats["pool"].shape == (1, 512)


class TestSimCLR:
    def test_forward_loss(self):
        from dr_pipeline.models.encoder import DREncoder
        from dr_pipeline.models.self_supervised import SimCLRPretrainer
        enc = DREncoder(backbone_name="resnet18", pretrained_imagenet=False)
        model = SimCLRPretrainer(encoder=enc, projection_dim=64)
        v1 = torch.randn(4, 3, 224, 224)
        v2 = torch.randn(4, 3, 224, 224)
        loss, z1, z2 = model(v1, v2)
        assert loss.dim() == 0  # scalar
        assert z1.shape == (4, 64)


class TestDRGrading:
    def test_forward_shape(self):
        from dr_pipeline.models.dr_grading import DRGradingModel
        from dr_pipeline.models.encoder import DREncoder
        enc = DREncoder(backbone_name="resnet18", pretrained_imagenet=False)
        model = DRGradingModel(encoder=enc)
        x = torch.randn(2, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (2, 5)


class TestLesionClassifier:
    def test_forward_and_loss(self):
        from dr_pipeline.models.encoder import DREncoder
        from dr_pipeline.models.lesion_classifier import MultiLesionClassifier
        enc = DREncoder(backbone_name="resnet18", pretrained_imagenet=False)
        model = MultiLesionClassifier(encoder=enc)
        x = torch.randn(2, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (2, 4)

        targets = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)
        loss = model.compute_loss(logits, targets)
        assert loss.dim() == 0

    def test_predict_with_thresholds(self):
        from dr_pipeline.models.encoder import DREncoder
        from dr_pipeline.models.lesion_classifier import MultiLesionClassifier
        enc = DREncoder(backbone_name="resnet18", pretrained_imagenet=False)
        model = MultiLesionClassifier(encoder=enc)
        x = torch.randn(2, 3, 224, 224)
        preds = model.predict(x, thresholds=torch.tensor([0.3, 0.5, 0.5, 0.7]))
        assert preds.shape == (2, 4)
        assert set(preds.unique().tolist()).issubset({0.0, 1.0})


class TestMultiScale:
    def test_forward_shape(self):
        from dr_pipeline.models.encoder import DREncoder
        from dr_pipeline.models.multi_scale import MultiScaleLesionClassifier
        enc = DREncoder(backbone_name="resnet18", pretrained_imagenet=False)
        model = MultiScaleLesionClassifier(encoder=enc)
        gv = torch.randn(2, 3, 224, 224)
        patches = torch.randn(2, 4, 3, 128, 128)
        logits, attn = model(gv, patches)
        assert logits.shape == (2, 4)
        assert attn.shape == (2, 4)


class TestSegmentation:
    def test_forward_shape(self):
        from dr_pipeline.models.encoder import DREncoder
        from dr_pipeline.models.segmentation import LesionSegmentationModel
        enc = DREncoder(backbone_name="resnet18", pretrained_imagenet=False)
        model = LesionSegmentationModel(encoder=enc, decoder_type="unet",
                                         loss_type="dice_bce")
        x = torch.randn(1, 3, 256, 256)
        out = model(x)
        assert out.shape[0] == 1
        assert out.shape[1] == 4  # 4 lesion classes

    def test_loss_computation(self):
        from dr_pipeline.models.encoder import DREncoder
        from dr_pipeline.models.segmentation import LesionSegmentationModel
        enc = DREncoder(backbone_name="resnet18", pretrained_imagenet=False)
        model = LesionSegmentationModel(encoder=enc)
        x = torch.randn(1, 3, 256, 256)
        pred = model(x)
        target = torch.zeros(1, 4, pred.shape[2], pred.shape[3])
        loss = model.compute_loss(pred, target)
        assert loss.dim() == 0


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_per_lesion_auc(self):
        from dr_pipeline.evaluation.metrics import per_lesion_auc
        np.random.seed(42)
        y_true = np.random.randint(0, 2, (100, 4)).astype(float)
        y_score = np.random.rand(100, 4)
        result = per_lesion_auc(y_true, y_score)
        assert len(result) == 4
        for v in result.values():
            assert 0.0 <= v <= 1.0 or np.isnan(v)

    def test_multilabel_f1(self):
        from dr_pipeline.evaluation.metrics import multilabel_f1
        y_true = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
        y_pred = np.array([[1, 0, 0, 0], [0, 1, 0, 1]])
        result = multilabel_f1(y_true, y_pred)
        assert "f1_macro" in result
        assert "f1_micro" in result

    def test_dice_score(self):
        from dr_pipeline.evaluation.metrics import dice_score
        mask = np.ones((10, 10))
        assert dice_score(mask, mask) > 0.99
        assert dice_score(mask, np.zeros_like(mask)) < 0.01


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_ece(self):
        from dr_pipeline.evaluation.calibration import expected_calibration_error
        y_true = np.random.randint(0, 2, (50, 4)).astype(float)
        y_prob = np.random.rand(50, 4)
        ece, acc, conf, counts = expected_calibration_error(y_true, y_prob)
        assert 0.0 <= ece <= 1.0

    def test_coverage_risk(self):
        from dr_pipeline.evaluation.calibration import coverage_risk_curve
        y_true = np.random.randint(0, 2, 100).astype(float)
        y_prob = np.random.rand(100)
        uncertainty = np.random.rand(100)
        cov, risk = coverage_risk_curve(y_true, y_prob, uncertainty)
        assert len(cov) == 100
        assert cov[-1] == pytest.approx(1.0, abs=0.02)


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_grey_world(self):
        from dr_pipeline.datasets.preprocessing import DRPreprocessor
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = DRPreprocessor.grey_world(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_detect_fov(self):
        from dr_pipeline.datasets.preprocessing import DRPreprocessor
        # Synthetic image with a bright circle
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2_imported = True
        try:
            import cv2
            cv2.circle(img, (100, 100), 80, (200, 200, 200), -1)
        except ImportError:
            cv2_imported = False

        if cv2_imported:
            cx, cy, r = DRPreprocessor.detect_fov(img)
            assert 50 < cx < 150
            assert 50 < cy < 150
            assert r > 50
