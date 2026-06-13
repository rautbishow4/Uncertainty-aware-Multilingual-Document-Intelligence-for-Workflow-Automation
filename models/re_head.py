"""
re_head.py
----------
Relation Extraction head using biaffine attention classifier.

Based on the LayoutXLM paper (Xu et al., 2021).
For each pair of predicted entities (head, tail), classifies whether
a key-value relation exists between them.

Architecture per entity pair (e_i, e_j):
  h_head = FFN_head([h_i ; e_head_type])
  h_tail = FFN_tail([h_j ; e_tail_type])
  score  = h_head^T U h_tail + W(h_head ⊙ h_tail) + b
"""

import torch
import torch.nn as nn
from typing import Optional, List


class BiaffineClassifier(nn.Module):
    """
    Biaffine scoring for entity pair relation classification.
    score(h, t) = h^T U t + W(h ⊙ t) + b
    """

    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        # Bilinear weight: [C, D, D]
        self.U = nn.Parameter(torch.randn(num_classes, input_dim, input_dim) * 0.01)
        # Element-wise weight: [C, D]
        self.W = nn.Linear(input_dim, num_classes, bias=True)

    def forward(self, h_head: torch.Tensor, h_tail: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_head: [B, N, D]
            h_tail: [B, N, D]

        Returns:
            scores: [B, N, N, C]
        """
        B, N, D = h_head.shape
        C = self.num_classes

        # Bilinear: h_head [B, N, 1, D] x U [C, D, D] x h_tail [B, 1, N, D]
        h_head_exp = h_head.unsqueeze(2)   # [B, N, 1, D]
        h_tail_exp = h_tail.unsqueeze(1)   # [B, 1, N, D]

        # For each class: score = h_head @ U_c @ h_tail^T
        bilinear_scores = torch.zeros(B, N, N, C, device=h_head.device)
        for c in range(C):
            # [B, N, D] x [D, D] = [B, N, D]
            tmp = torch.matmul(h_head, self.U[c])           # [B, N, D]
            # [B, N, D] x [B, D, N] = [B, N, N]
            bilinear_scores[:, :, :, c] = torch.bmm(tmp, h_tail.transpose(1, 2))

        # Element-wise component: W(h_head ⊙ h_tail)
        # Broadcast: [B, N, 1, D] * [B, 1, N, D] = [B, N, N, D]
        elem = h_head_exp * h_tail_exp                       # [B, N, N, D]
        elem_scores = self.W(elem)                           # [B, N, N, C]

        return bilinear_scores + elem_scores


class REHead(nn.Module):
    """
    Relation Extraction head with biaffine classifier.

    For each entity, takes the first token representation and
    concatenates with a learnable entity type embedding.
    Then projects with separate FFNs for head and tail roles.

    Args:
        hidden_size: Encoder hidden dimension.
        num_entity_types: Number of entity type classes.
        proj_dim: Projection size for head/tail FFNs.
        num_relation_classes: 2 (no relation / key-value relation).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_entity_types: int = 4,
        proj_dim: int = 256,
        num_relation_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_entity_types = num_entity_types
        self.entity_type_embed = nn.Embedding(num_entity_types, hidden_size)

        # Head and tail FFNs
        self.ffn_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
        )
        self.ffn_tail = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
        )

        self.biaffine = BiaffineClassifier(input_dim=proj_dim, num_classes=num_relation_classes)
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8]))  # upweight positive

    def _gather_entity_reps(
        self,
        sequence_output: torch.Tensor,
        entity_labels: torch.Tensor,
        num_entities: List[int],
    ) -> torch.Tensor:
        """
        Build per-entity representations by taking the first token of each entity.
        Since we don't have direct token→entity alignment here, we use a
        mean-pooling approximation over non-padding entity tokens.

        This implementation uses a simple heuristic: for entity i, we take
        position i+1 from the encoder output (skipping [CLS]).
        In full production, use the word_ids_map for exact mapping.

        Returns:
            entity_reps: [B, max_entities, hidden_size]
        """
        B = sequence_output.size(0)
        max_ent = max(num_entities) if num_entities else entity_labels.size(1)
        H = sequence_output.size(-1)

        entity_reps = torch.zeros(B, max_ent, H, device=sequence_output.device)
        for b_idx in range(B):
            n = num_entities[b_idx] if num_entities else max_ent
            # Heuristic: take tokens at positions 1..n+1 (skip CLS)
            n_avail = min(n, sequence_output.size(1) - 1)
            entity_reps[b_idx, :n_avail] = sequence_output[b_idx, 1:n_avail+1]

        return entity_reps

    def forward(
        self,
        sequence_output: torch.Tensor,
        entity_labels: Optional[torch.Tensor] = None,
        num_entities: Optional[List[int]] = None,
    ) -> Optional[torch.Tensor]:
        """
        Args:
            sequence_output: [B, seq_len, H]
            entity_labels: [B, max_entities] — entity type IDs
            num_entities: list of actual entity counts per item

        Returns:
            re_logits: [B, max_entities, max_entities, 2] or None
        """
        if entity_labels is None:
            return None

        B = sequence_output.size(0)
        max_ent = entity_labels.size(1)
        if num_entities is None:
            num_entities = [max_ent] * B

        # Gather per-entity representations
        entity_reps = self._gather_entity_reps(sequence_output, entity_labels, num_entities)
        # [B, max_ent, H]

        # Entity type embeddings
        type_embs = self.entity_type_embed(entity_labels)  # [B, max_ent, H]

        # Concatenate representation + type embedding
        combined = torch.cat([entity_reps, type_embs], dim=-1)  # [B, max_ent, 2H]

        h_head = self.ffn_head(combined)  # [B, max_ent, proj_dim]
        h_tail = self.ffn_tail(combined)  # [B, max_ent, proj_dim]

        re_logits = self.biaffine(h_head, h_tail)  # [B, max_ent, max_ent, 2]
        return re_logits

    def compute_loss(
        self,
        re_logits: torch.Tensor,
        relation_matrix: torch.Tensor,
        num_entities: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Binary cross-entropy over valid entity pairs.

        Args:
            re_logits: [B, N, N, 2]
            relation_matrix: [B, N, N]  — 0/1 ground truth
            num_entities: list of valid entity counts

        Returns:
            Scalar loss.
        """
        B, N, _, C = re_logits.shape

        # Build mask for valid pairs
        mask = torch.zeros(B, N, N, dtype=torch.bool, device=re_logits.device)
        for b, n in enumerate(num_entities or [N] * B):
            mask[b, :n, :n] = True

        # Exclude self-relations
        eye = torch.eye(N, dtype=torch.bool, device=re_logits.device).unsqueeze(0)
        mask = mask & ~eye

        logits_flat = re_logits[mask]        # [M, 2]
        labels_flat = relation_matrix[mask]  # [M]

        if logits_flat.numel() == 0:
            return re_logits.sum() * 0.0  # zero loss, preserve gradients

        return self.loss_fn(logits_flat, labels_flat)
