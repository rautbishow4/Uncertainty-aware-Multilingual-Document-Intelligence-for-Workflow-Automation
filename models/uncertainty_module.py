"""
uncertainty_module.py
---------------------
Bayesian uncertainty estimation for document intelligence.

Methods implemented:
  1. Monte Carlo Dropout (Gal & Ghahramani, 2016)
     - Run T stochastic forward passes
     - Compute predictive entropy and variance
  2. Temperature Scaling (Guo et al., 2017)
     - Post-hoc calibration of softmax confidence
  3. Expected Calibration Error (ECE)
     - Evaluates how well confidence aligns with accuracy

Inspired by:
  - Ngartera et al. (2025), Bayesian RAG for financial documents
  - Kadavath et al. (2022), language model calibration
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple


class UncertaintyModule(nn.Module):
    """
    Uncertainty quantification module.

    Computes per-token and per-document uncertainty scores from
    MC-Dropout ensemble outputs.

    Args:
        num_labels: Number of SER token labels.
        mc_passes: Number of stochastic forward passes.
        confidence_threshold: Below this → flag for human review.
    """

    def __init__(
        self,
        num_labels: int = 7,
        mc_passes: int = 20,
        confidence_threshold: float = 0.70,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.mc_passes = mc_passes
        self.confidence_threshold = confidence_threshold

        # Learnable temperature for calibration
        self.temperature = nn.Parameter(torch.ones(1))

    def compute_token_uncertainty(
        self,
        probs_stack: torch.Tensor,         # [T, B, seq_len, C]
        attention_mask: torch.Tensor,      # [B, seq_len]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute token-level uncertainty metrics from MC-Dropout ensemble.

        Args:
            probs_stack: Stacked softmax probabilities from T forward passes.
            attention_mask: Binary mask (1 = real token, 0 = padding).

        Returns:
            Dict with:
              - mean_probs:   [B, seq_len, C] — mean prediction
              - confidence:   [B, seq_len]    — max probability
              - entropy:      [B, seq_len]    — predictive entropy H[p]
              - variance:     [B, seq_len]    — total variance across passes
              - mutual_info:  [B, seq_len]    — mutual information (epistemic)
        """
        T, B, seq_len, C = probs_stack.shape
        eps = 1e-8

        # Mean over MC passes → predictive distribution
        mean_probs = probs_stack.mean(dim=0)  # [B, seq_len, C]

        # Confidence = max probability under mean distribution
        confidence = mean_probs.max(dim=-1).values  # [B, seq_len]

        # Predictive entropy (total uncertainty)
        # H[p(y|x)] = -sum_c p_c log(p_c)
        entropy = -(mean_probs * (mean_probs + eps).log()).sum(dim=-1)  # [B, seq_len]

        # Variance across MC passes (another uncertainty measure)
        variance = probs_stack.var(dim=0).sum(dim=-1)  # [B, seq_len]

        # Mutual information = epistemic uncertainty
        # MI = H[E_p] - E[H[p]]
        # = predictive entropy - expected entropy over MC passes
        per_pass_entropy = -(probs_stack * (probs_stack + eps).log()).sum(dim=-1)  # [T, B, seq]
        expected_entropy = per_pass_entropy.mean(dim=0)                             # [B, seq]
        mutual_info = entropy - expected_entropy                                    # [B, seq]
        mutual_info = mutual_info.clamp(min=0.0)  # MI ≥ 0

        # Mask out padding tokens
        mask = attention_mask.float()
        confidence  = confidence  * mask
        entropy     = entropy     * mask
        variance    = variance    * mask
        mutual_info = mutual_info * mask

        return {
            "mean_probs":  mean_probs,
            "confidence":  confidence,
            "entropy":     entropy,
            "variance":    variance,
            "mutual_info": mutual_info,
        }

    def aggregate_document_confidence(
        self,
        token_confidence: torch.Tensor,     # [B, seq_len]
        attention_mask: torch.Tensor,       # [B, seq_len]
        aggregation: str = "mean",
    ) -> torch.Tensor:
        """
        Aggregate token-level confidence to document-level score.

        Args:
            token_confidence: Per-token confidence scores.
            attention_mask: Padding mask.
            aggregation: "mean" | "min" | "weighted_mean"

        Returns:
            doc_confidence: [B] — scalar confidence per document.
        """
        real_tokens = attention_mask.float()
        counts = real_tokens.sum(dim=1).clamp(min=1.0)

        if aggregation == "mean":
            doc_conf = (token_confidence * real_tokens).sum(dim=1) / counts

        elif aggregation == "min":
            # Conservative: use worst-token confidence
            large_val = torch.ones_like(token_confidence)
            masked = torch.where(real_tokens.bool(), token_confidence, large_val)
            doc_conf = masked.min(dim=1).values

        elif aggregation == "weighted_mean":
            # Weight by inverse-entropy: low-entropy tokens get higher weight
            eps = 1e-8
            weights = 1.0 / (token_confidence + eps)
            weights = weights * real_tokens
            weight_sum = weights.sum(dim=1).clamp(min=eps)
            doc_conf = (token_confidence * weights).sum(dim=1) / weight_sum

        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        return doc_conf  # [B]

    def should_trigger_review(
        self,
        doc_confidence: torch.Tensor,  # [B]
        threshold: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Returns a boolean mask indicating which documents need human review.

        Args:
            doc_confidence: Document-level confidence scores.
            threshold: Confidence threshold (default: self.confidence_threshold).

        Returns:
            review_flags: [B] bool tensor — True = needs review.
        """
        t = threshold if threshold is not None else self.confidence_threshold
        return doc_confidence < t

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply learned temperature scaling for calibration.
        T < 1 sharpens confidence; T > 1 flattens it.
        """
        return logits / self.temperature.clamp(min=0.01)

    def forward(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Single-pass uncertainty from calibrated logits.
        Used during evaluation (not MC-Dropout ensemble).
        """
        calibrated_logits = self.temperature_scale(logits)
        probs = torch.softmax(calibrated_logits, dim=-1)
        confidence = probs.max(dim=-1).values

        eps = 1e-8
        entropy = -(probs * (probs + eps).log()).sum(dim=-1)

        doc_confidence = self.aggregate_document_confidence(confidence, attention_mask)
        review_flags = self.should_trigger_review(doc_confidence)

        return {
            "probs":          probs,
            "confidence":     confidence,
            "entropy":        entropy,
            "doc_confidence": doc_confidence,
            "needs_review":   review_flags,
        }


class TemperatureScalingCalibrator:
    """
    Post-hoc calibration using temperature scaling.
    Fits a single scalar temperature on a validation set
    by minimizing negative log-likelihood.
    """

    def __init__(self, model: UncertaintyModule):
        self.model = model

    def calibrate(
        self,
        val_logits: torch.Tensor,   # [N, C]
        val_labels: torch.Tensor,   # [N]
        max_iter: int = 100,
        lr: float = 0.01,
    ) -> float:
        """
        Optimize temperature on validation logits/labels.

        Returns:
            Optimal temperature value.
        """
        optimizer = torch.optim.LBFGS(
            [self.model.temperature], lr=lr, max_iter=max_iter
        )
        loss_fn = nn.CrossEntropyLoss()

        def eval_step():
            optimizer.zero_grad()
            scaled = val_logits / self.model.temperature.clamp(min=0.01)
            loss = loss_fn(scaled, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        return self.model.temperature.item()


def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE).

    Bins predictions by confidence, measures average gap between
    confidence and accuracy within each bin.

    Args:
        confidences: [N] predicted confidence scores.
        accuracies:  [N] binary correct/incorrect per prediction.
        n_bins: Number of equal-width bins.

    Returns:
        ECE as a float in [0, 1].
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(confidences)

    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences >= lower) & (confidences < upper)
        n_in_bin = in_bin.sum()

        if n_in_bin > 0:
            bin_acc  = accuracies[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            ece += (n_in_bin / N) * abs(bin_conf - bin_acc)

    return float(ece)


def compute_mce(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Maximum Calibration Error (MCE) — worst-bin calibration gap."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    max_gap = 0.0

    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences >= lower) & (confidences < upper)
        if in_bin.sum() > 0:
            gap = abs(confidences[in_bin].mean() - accuracies[in_bin].mean())
            max_gap = max(max_gap, gap)

    return float(max_gap)


def reliability_diagram_data(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, List[float]]:
    """
    Compute data for a reliability diagram.

    Returns:
        Dict with 'bin_centers', 'bin_accuracies', 'bin_confidences', 'bin_counts'.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers, bin_accs, bin_confs, bin_counts = [], [], [], []

    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences >= lower) & (confidences < upper)
        n = int(in_bin.sum())

        bin_centers.append(float((lower + upper) / 2))
        bin_counts.append(n)

        if n > 0:
            bin_accs.append(float(accuracies[in_bin].mean()))
            bin_confs.append(float(confidences[in_bin].mean()))
        else:
            bin_accs.append(0.0)
            bin_confs.append(float((lower + upper) / 2))

    return {
        "bin_centers":     bin_centers,
        "bin_accuracies":  bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts":      bin_counts,
    }
