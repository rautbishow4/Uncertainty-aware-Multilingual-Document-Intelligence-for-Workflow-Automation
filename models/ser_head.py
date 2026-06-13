"""
ser_head.py
-----------
Semantic Entity Recognition head.
Token classification with BIO labeling scheme.
"""

import torch
import torch.nn as nn
from typing import Optional


class SERHead(nn.Module):
    """
    Feed-forward network head for Semantic Entity Recognition (token classification).

    Given sequence representations from the encoder, projects to BIO tag logits.
    Uses cross-entropy loss with label smoothing and ignores padding tokens (-100).

    Args:
        hidden_size: Encoder output dimension.
        num_labels: Number of BIO token labels.
        dropout: Dropout probability before projection.
        label_smoothing: Label smoothing factor for cross-entropy.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_labels: int = 7,
        dropout: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.num_labels = num_labels

        self.dropout = nn.Dropout(p=dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=label_smoothing,
        )

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence_output: [B, seq_len, hidden_size]

        Returns:
            logits: [B, seq_len, num_labels]
        """
        x = self.dropout(sequence_output)
        x = self.dense(x)
        x = self.activation(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-entropy loss over non-padding positions.

        Args:
            logits: [B, seq_len, num_labels]
            labels: [B, seq_len]  — -100 marks ignored positions

        Returns:
            Scalar loss tensor.
        """
        B, seq_len, C = logits.shape
        return self.loss_fn(logits.view(B * seq_len, C), labels.view(B * seq_len))

    def decode(
        self,
        logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Greedy decoding: argmax over label dimension.

        Args:
            logits: [B, seq_len, num_labels]
            attention_mask: [B, seq_len]  — 0 positions → label 0

        Returns:
            predictions: [B, seq_len]
        """
        preds = logits.argmax(dim=-1)
        if attention_mask is not None:
            preds = preds * attention_mask
        return preds
