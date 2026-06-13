"""
tests/test_model.py
-------------------
Unit tests for the uncertainty-aware model components.
Run with: pytest tests/ -v
"""

import pytest
import torch
import numpy as np


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_batch():
    B, S, C = 2, 64, 7
    N = 5  # entities per doc
    return {
        "input_ids":      torch.randint(0, 1000, (B, S)),
        "attention_mask": torch.ones(B, S, dtype=torch.long),
        "bbox":           torch.randint(0, 1000, (B, S, 4)),
        "image":          torch.randn(B, 3, 224, 224),
        "labels":         torch.randint(-100, C, (B, S)),
        "entity_labels":  torch.randint(0, 4, (B, N)),
        "relation_matrix":torch.randint(0, 2, (B, N, N)),
        "num_entities":   [N, N],
    }


# ─────────────────────────────────────────────────────────────
# SER Head tests
# ─────────────────────────────────────────────────────────────

class TestSERHead:
    def test_output_shape(self):
        from models.ser_head import SERHead
        head = SERHead(hidden_size=64, num_labels=7)
        x = torch.randn(2, 32, 64)
        logits = head(x)
        assert logits.shape == (2, 32, 7)

    def test_loss_computes(self):
        from models.ser_head import SERHead
        head = SERHead(hidden_size=64, num_labels=7)
        logits = torch.randn(2, 32, 7)
        # Labels: either -100 (ignore) or valid class index [0, 6]
        labels = torch.randint(0, 7, (2, 32))
        labels[0, 0] = -100  # one ignored position
        loss = head.compute_loss(logits, labels)
        assert loss.item() >= 0.0

    def test_decode_shape(self):
        from models.ser_head import SERHead
        head = SERHead(hidden_size=64, num_labels=7)
        logits = torch.randn(2, 32, 7)
        preds = head.decode(logits)
        assert preds.shape == (2, 32)
        assert (preds >= 0).all() and (preds < 7).all()


# ─────────────────────────────────────────────────────────────
# RE Head tests
# ─────────────────────────────────────────────────────────────

class TestREHead:
    def test_output_shape(self):
        from models.re_head import REHead
        head = REHead(hidden_size=64, num_entity_types=4, proj_dim=32)
        seq = torch.randn(2, 32, 64)
        ent_labels = torch.randint(0, 4, (2, 5))
        logits = head(seq, ent_labels, num_entities=[5, 5])
        assert logits is not None
        assert logits.shape == (2, 5, 5, 2)

    def test_loss_computes(self):
        from models.re_head import REHead
        head = REHead(hidden_size=64, num_entity_types=4, proj_dim=32)
        seq = torch.randn(2, 32, 64)
        ent_labels = torch.randint(0, 4, (2, 5))
        logits = head(seq, ent_labels, [5, 5])
        rel_mat = torch.randint(0, 2, (2, 5, 5))
        loss = head.compute_loss(logits, rel_mat, [5, 5])
        assert loss.item() >= 0.0


# ─────────────────────────────────────────────────────────────
# Uncertainty Module tests
# ─────────────────────────────────────────────────────────────

class TestUncertaintyModule:
    def setup_method(self):
        from models.uncertainty_module import UncertaintyModule
        self.module = UncertaintyModule(num_labels=7, mc_passes=5, confidence_threshold=0.70)

    def test_token_uncertainty_shapes(self):
        T, B, S, C = 5, 2, 32, 7
        probs_stack = torch.softmax(torch.randn(T, B, S, C), dim=-1)
        attn = torch.ones(B, S)
        out = self.module.compute_token_uncertainty(probs_stack, attn)

        assert out["mean_probs"].shape  == (B, S, C)
        assert out["confidence"].shape  == (B, S)
        assert out["entropy"].shape     == (B, S)
        assert out["variance"].shape    == (B, S)
        assert out["mutual_info"].shape == (B, S)

    def test_entropy_range(self):
        T, B, S, C = 5, 2, 32, 7
        probs_stack = torch.softmax(torch.randn(T, B, S, C), dim=-1)
        attn = torch.ones(B, S)
        out = self.module.compute_token_uncertainty(probs_stack, attn)
        # Entropy should be non-negative
        assert (out["entropy"] >= -1e-5).all()

    def test_document_confidence_range(self):
        conf = torch.rand(2, 32)
        mask = torch.ones(2, 32)
        doc_conf = self.module.aggregate_document_confidence(conf, mask)
        assert doc_conf.shape == (2,)
        assert (doc_conf >= 0.0).all() and (doc_conf <= 1.0).all()

    def test_review_trigger(self):
        # Should trigger review for low-confidence docs
        doc_conf = torch.tensor([0.50, 0.85])
        flags = self.module.should_trigger_review(doc_conf, threshold=0.70)
        assert flags[0].item() == True    # 0.50 < 0.70
        assert flags[1].item() == False   # 0.85 ≥ 0.70

    def test_temperature_scaling(self):
        logits = torch.randn(4, 7)
        scaled = self.module.temperature_scale(logits)
        assert scaled.shape == logits.shape


# ─────────────────────────────────────────────────────────────
# ECE / Calibration tests
# ─────────────────────────────────────────────────────────────

class TestCalibrationMetrics:
    def test_ece_perfect_calibration(self):
        """Perfect calibration: ECE should be near 0."""
        from models.uncertainty_module import compute_ece
        # Perfect: 70% confidence → 70% accuracy
        n = 1000
        conf = np.linspace(0, 1, n)
        acc  = (np.random.rand(n) < conf).astype(float)
        ece = compute_ece(conf, acc, n_bins=10)
        assert ece < 0.15  # loose bound for randomness

    def test_ece_overconfident(self):
        """Overconfident model: high confidence but low accuracy → high ECE."""
        from models.uncertainty_module import compute_ece
        n = 1000
        conf = np.ones(n) * 0.95   # always 95% confident
        acc  = np.zeros(n)          # always wrong
        ece = compute_ece(conf, acc, n_bins=10)
        assert ece > 0.5

    def test_reliability_diagram_data(self):
        from models.uncertainty_module import reliability_diagram_data
        n = 500
        conf = np.random.rand(n)
        acc  = (np.random.rand(n) < conf).astype(float)
        data = reliability_diagram_data(conf, acc, n_bins=10)
        assert len(data["bin_centers"])     == 10
        assert len(data["bin_accuracies"])  == 10
        assert len(data["bin_confidences"]) == 10


# ─────────────────────────────────────────────────────────────
# Dataset helper tests
# ─────────────────────────────────────────────────────────────

class TestDatasetUtils:
    def test_normalize_bbox(self):
        from data.xfund_dataset import normalize_bbox
        bbox = [100, 200, 300, 400]
        norm = normalize_bbox(bbox, width=1000, height=1000, scale=1000)
        assert norm == [100, 200, 300, 400]

    def test_normalize_bbox_clamps(self):
        from data.xfund_dataset import normalize_bbox
        bbox = [-10, 0, 1100, 500]
        norm = normalize_bbox(bbox, width=1000, height=1000, scale=1000)
        assert norm[0] == 0    # clamped to 0
        assert norm[2] == 1000 # clamped to scale

    def test_label_maps_consistent(self):
        from data.xfund_dataset import LABEL2ID, ID2LABEL
        for label, idx in LABEL2ID.items():
            assert ID2LABEL[idx] == label


# ─────────────────────────────────────────────────────────────
# Metrics tests
# ─────────────────────────────────────────────────────────────

class TestMetrics:
    def test_ser_metrics_empty(self):
        from evaluation.metrics import compute_ser_metrics
        metrics = compute_ser_metrics([], [])
        assert metrics["ser_f1"] == 0.0

    def test_ser_metrics_perfect(self):
        from evaluation.metrics import compute_ser_metrics
        preds  = [0, 1, 2, 3, 4, 5, 6]
        labels = [0, 1, 2, 3, 4, 5, 6]
        metrics = compute_ser_metrics(preds, labels)
        assert metrics["ser_f1"] == pytest.approx(1.0, abs=0.01)

    def test_re_metrics_binary(self):
        from evaluation.metrics import compute_re_metrics
        preds  = [0, 1, 1, 0, 1]
        labels = [0, 1, 0, 0, 1]
        metrics = compute_re_metrics(preds, labels)
        assert "re_f1" in metrics
        assert 0.0 <= metrics["re_f1"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
