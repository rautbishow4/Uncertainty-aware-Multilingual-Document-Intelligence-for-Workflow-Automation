"""
layout_xlm_uncertainty.py
--------------------------
Uncertainty-aware LayoutXLM model for multilingual document intelligence.

Architecture:
  ┌─────────────────────────────────────────────┐
  │  LayoutXLM Encoder (text + layout + image)  │
  │  with MC-Dropout active at inference         │
  └──────────┬──────────────┬───────────────────┘
             │              │
       ┌─────▼──────┐  ┌────▼───────────┐
       │  SER Head  │  │   RE Head      │
       │ (BIO tags) │  │ (biaffine cls) │
       └─────┬──────┘  └────┬───────────┘
             │              │
       ┌─────▼──────────────▼───┐
       │   Uncertainty Module   │
       │  (T stochastic passes) │
       └────────────────────────┘
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from typing import Optional, Dict

from models.ser_head import SERHead
from models.re_head import REHead
from models.uncertainty_module import UncertaintyModule


def _load_config(model_name: str):
    try:
        return AutoConfig.from_pretrained(model_name, local_files_only=True)
    except OSError:
        return AutoConfig.from_pretrained(model_name)


def _load_model(model_name: str, config):
    try:
        return AutoModel.from_pretrained(model_name, config=config, local_files_only=True)
    except OSError:
        return AutoModel.from_pretrained(model_name, config=config)


class LayoutXLMUncertainty(nn.Module):
    """
    Full uncertainty-aware multilingual document intelligence model.

    Wraps LayoutXLM with:
      - Semantic Entity Recognition (SER) head (token classification, BIO scheme)
      - Relation Extraction (RE) head (biaffine entity-pair classification)
      - MC-Dropout uncertainty estimation module

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        num_labels_ser: Number of BIO token labels (default 7).
        num_entity_types: Number of entity type classes (default 4).
        hidden_size: Encoder hidden dimension.
        dropout_rate: Standard dropout.
        mc_dropout_rate: Dropout rate kept active during MC inference.
        mc_forward_passes: Number of stochastic passes for uncertainty.
    """

    def __init__(
        self,
        model_name_or_path: str = "microsoft/layoutxlm-base",
        num_labels_ser: int = 7,
        num_entity_types: int = 4,
        hidden_size: int = 768,
        dropout_rate: float = 0.1,
        mc_dropout_rate: float = 0.1,
        mc_forward_passes: int = 20,
    ):
        super().__init__()
        self.model_name = model_name_or_path
        self.num_labels_ser = num_labels_ser
        self.mc_forward_passes = mc_forward_passes
        self.text_only_encoder = True

        config = _load_config(model_name_or_path)
        if config.model_type in {"layoutlmv2", "layoutxlm"}:
            # LayoutLMv2/LayoutXLM need detectron2. Keep the old Windows-safe
            # fallback for those models, but allow LayoutLMv3 to run natively.
            self.model_name = "bert-base-multilingual-cased"
            config = _load_config(self.model_name)
        elif config.model_type in {"layoutlm", "layoutlmv2", "layoutlmv3"}:
            self.text_only_encoder = False
        if hasattr(config, "hidden_dropout_prob"):
            config.hidden_dropout_prob = dropout_rate
        if hasattr(config, "attention_probs_dropout_prob"):
            config.attention_probs_dropout_prob = dropout_rate
        self.encoder = _load_model(self.model_name, config=config)

        # MC-Dropout layer (kept active at inference for uncertainty sampling)
        self.mc_dropout = MCDropout(p=mc_dropout_rate)

        # Task heads
        self.ser_head = SERHead(hidden_size=hidden_size, num_labels=num_labels_ser, dropout=dropout_rate)
        self.re_head  = REHead(hidden_size=hidden_size, num_entity_types=num_entity_types, dropout=dropout_rate)

        # Uncertainty estimation
        self.uncertainty_module = UncertaintyModule(num_labels=num_labels_ser, mc_passes=mc_forward_passes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bbox: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        entity_labels: Optional[torch.Tensor] = None,
        relation_matrix: Optional[torch.Tensor] = None,
        num_entities: Optional[list] = None,
        return_uncertainty: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
            bbox: [B, seq_len, 4]  — normalized 0-1000
            image: [B, 3, H, W]
            labels: [B, seq_len]  — SER BIO labels (-100 = ignore)
            entity_labels: [B, num_entities]
            relation_matrix: [B, num_entities, num_entities]
            num_entities: list of int per batch item
            return_uncertainty: if True, run MC-Dropout ensemble

        Returns:
            dict with keys: ser_logits, re_logits, loss (optional),
                            ser_loss (optional), re_loss (optional),
                            uncertainty (optional, if return_uncertainty=True)
        """
        # Encode document. Native Windows uses the text-only fallback above.
        if self.text_only_encoder:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=image,
            )
        sequence_output = encoder_outputs.last_hidden_state
        sequence_output = sequence_output[:, : input_ids.size(1), :]   # keep text tokens only
        sequence_output = self.mc_dropout(sequence_output)

        # SER: token-level logits
        ser_logits = self.ser_head(sequence_output)            # [B, seq_len, num_labels]

        # RE: entity-pair logits
        re_logits = self.re_head(
            sequence_output=sequence_output,
            entity_labels=entity_labels,
            num_entities=num_entities,
        )  # [B, num_entities, num_entities, 2]

        outputs = {
            "ser_logits": ser_logits,
            "re_logits":  re_logits,
        }

        # Compute losses if labels provided
        if labels is not None:
            ser_loss = self.ser_head.compute_loss(ser_logits, labels)
            outputs["ser_loss"] = ser_loss

        if relation_matrix is not None and re_logits is not None:
            re_loss = self.re_head.compute_loss(re_logits, relation_matrix, num_entities)
            outputs["re_loss"] = re_loss

        if "ser_loss" in outputs and "re_loss" in outputs:
            outputs["loss"] = outputs["ser_loss"] + outputs["re_loss"]

        # Uncertainty estimation via MC-Dropout ensemble
        if return_uncertainty:
            uncertainty = self._estimate_uncertainty(
                input_ids, attention_mask, bbox, image, entity_labels, num_entities
            )
            outputs["uncertainty"] = uncertainty

        return outputs

    def _estimate_uncertainty(
        self,
        input_ids, attention_mask, bbox, image, entity_labels, num_entities
    ) -> Dict[str, torch.Tensor]:
        """
        Run T stochastic forward passes with MC-Dropout active.
        Computes predictive entropy and variance as uncertainty estimates.
        """
        self.train()  # activate dropout
        all_ser_probs = []
        all_re_probs  = []

        with torch.no_grad():
            for _ in range(self.mc_forward_passes):
                if self.text_only_encoder:
                    enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                else:
                    enc = self.encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        bbox=bbox,
                        pixel_values=image,
                    ).last_hidden_state
                    enc = enc[:, : input_ids.size(1), :]
                enc = self.mc_dropout(enc)

                ser_logits_t = self.ser_head(enc)
                ser_probs_t  = torch.softmax(ser_logits_t, dim=-1)  # [B, seq, C]
                all_ser_probs.append(ser_probs_t.unsqueeze(0))

                re_logits_t = self.re_head(enc, entity_labels, num_entities)
                if re_logits_t is not None:
                    re_probs_t = torch.softmax(re_logits_t, dim=-1)
                    all_re_probs.append(re_probs_t.unsqueeze(0))

        self.eval()

        # Stack: [T, B, seq, C]
        all_ser_probs = torch.cat(all_ser_probs, dim=0)

        # Mean prediction
        mean_ser_probs = all_ser_probs.mean(dim=0)  # [B, seq, C]

        # Predictive entropy: H[p] = -sum(p * log(p))
        eps = 1e-8
        entropy = -(mean_ser_probs * (mean_ser_probs + eps).log()).sum(dim=-1)  # [B, seq]

        # Variance across passes
        variance = all_ser_probs.var(dim=0).sum(dim=-1)  # [B, seq]

        # Token-level confidence = max predicted probability
        confidence = mean_ser_probs.max(dim=-1).values  # [B, seq]

        uncertainty_out = {
            "mean_ser_probs": mean_ser_probs,
            "entropy":        entropy,
            "variance":       variance,
            "confidence":     confidence,
        }

        if all_re_probs:
            all_re_probs = torch.cat(all_re_probs, dim=0)
            mean_re_probs = all_re_probs.mean(dim=0)
            re_confidence = mean_re_probs[..., 1]  # prob of relation=1
            uncertainty_out["re_confidence"] = re_confidence

        return uncertainty_out

    def predict(
        self,
        input_ids, attention_mask, bbox, image=None,
        entity_labels=None, num_entities=None,
        mc_passes: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference entry point. Runs MC-Dropout ensemble and returns
        predictions with uncertainty scores.
        """
        if mc_passes is not None:
            prev = self.mc_forward_passes
            self.mc_forward_passes = mc_passes

        self.eval()
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            image=image,
            entity_labels=entity_labels,
            num_entities=num_entities,
            return_uncertainty=True,
        )

        if mc_passes is not None:
            self.mc_forward_passes = prev

        return outputs

    @classmethod
    def from_pretrained_checkpoint(cls, checkpoint_path: str, **kwargs) -> "LayoutXLMUncertainty":
        """Load model from a saved checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_kwargs = checkpoint.get("model_config", {})
        model_kwargs.update(kwargs)
        model = cls(**model_kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def save_checkpoint(self, path: str, extra: Optional[dict] = None):
        """Save model checkpoint with config."""
        state = {
            "model_state_dict": self.state_dict(),
            "model_config": {
                "model_name_or_path": self.model_name,
                "num_labels_ser": self.num_labels_ser,
                "mc_forward_passes": self.mc_forward_passes,
            },
        }
        if extra:
            state.update(extra)
        torch.save(state, path)


class MCDropout(nn.Module):
    """
    MC-Dropout: standard Dropout but stays active during eval() too.
    Used to approximate Bayesian inference via sampling.
    """
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always apply dropout (training=True) for MC sampling
        return nn.functional.dropout(x, p=self.p, training=True)
