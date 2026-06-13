"""
trainer.py
----------
Training loop for the uncertainty-aware multilingual document intelligence system.

Supports:
  - Language-specific fine-tuning
  - Zero-shot transfer (train on EN/FUNSD, eval on target lang)
  - Multitask fine-tuning (all 7 XFUND languages jointly)
  - W&B logging
  - Checkpointing by best avg F1
  - Early stopping
  - Mixed precision (fp16)
"""

import os
import time
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from models.layout_xlm_uncertainty import LayoutXLMUncertainty
from evaluation.metrics import compute_ser_metrics, compute_re_metrics

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    num_epochs: int = 30
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    fp16: bool = True
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 50
    early_stopping_patience: int = 5
    ser_loss_weight: float = 1.0
    re_loss_weight: float = 1.0
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    use_wandb: bool = True
    wandb_project: str = "xfund-docai-uncertainty"
    seed: int = 42


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0

    def __call__(self, score: float) -> bool:
        """Returns True if training should stop."""
        if self.best_score is None:
            self.best_score = score
            return False
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class XFUNDTrainer:
    """
    Full training harness for LayoutXLMUncertainty on XFUND.

    Usage:
        trainer = XFUNDTrainer(model, train_loader, val_loader, config)
        trainer.train()
    """

    def __init__(
        self,
        model: LayoutXLMUncertainty,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {
                "params": [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(params, lr=config.learning_rate)

        # Scheduler
        total_steps = (len(train_loader) // config.gradient_accumulation_steps) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

        # Mixed precision
        self.scaler = GradScaler() if config.fp16 else None

        # Tracking
        self.global_step = 0
        self.best_metric = 0.0
        self.early_stopper = EarlyStopping(patience=config.early_stopping_patience)

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        # W&B
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.init(project=config.wandb_project)

    def train(self):
        logger.info(f"Starting training on {self.device}")
        logger.info(f"  Epochs          : {self.config.num_epochs}")
        logger.info(f"  Batch size      : {self.config.batch_size}")
        logger.info(f"  Gradient accum  : {self.config.gradient_accumulation_steps}")
        logger.info(f"  Learning rate   : {self.config.learning_rate}")

        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            t0 = time.time()

            for step, batch in enumerate(self.train_loader):
                loss = self._train_step(batch)
                epoch_loss += loss

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        lr = self.scheduler.get_last_lr()[0]
                        avg_loss = epoch_loss / (step + 1)
                        logger.info(
                            f"Epoch {epoch+1} | Step {self.global_step} | "
                            f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                        )
                        if WANDB_AVAILABLE and self.config.use_wandb:
                            wandb.log({"train/loss": avg_loss, "train/lr": lr}, step=self.global_step)

                    # Evaluation
                    if self.global_step % self.config.eval_steps == 0:
                        metrics = self.evaluate()
                        avg_f1 = (metrics["ser_f1"] + metrics["re_f1"]) / 2
                        logger.info(f"Eval @ step {self.global_step}: SER={metrics['ser_f1']:.4f} RE={metrics['re_f1']:.4f} AvgF1={avg_f1:.4f}")

                        if WANDB_AVAILABLE and self.config.use_wandb:
                            wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=self.global_step)

                        # Save best checkpoint
                        if avg_f1 > self.best_metric:
                            self.best_metric = avg_f1
                            self._save_checkpoint("best_model.pt", metrics)
                            logger.info(f"  ✓ New best model saved (avg F1={avg_f1:.4f})")

                        # Early stopping
                        if self.early_stopper(avg_f1):
                            logger.info(f"Early stopping triggered after {self.config.early_stopping_patience} patience steps.")
                            return

            elapsed = time.time() - t0
            logger.info(f"Epoch {epoch+1} done in {elapsed:.1f}s | Avg Loss: {epoch_loss/len(self.train_loader):.4f}")

        # Save final checkpoint
        self._save_checkpoint("final_model.pt", {"epoch": self.config.num_epochs})
        logger.info("Training complete.")

    def _train_step(self, batch: dict) -> float:
        """Single training step, returns scalar loss."""
        batch = self._move_to_device(batch)

        if self.scaler:
            with autocast():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    bbox=batch["bbox"],
                    image=batch["image"],
                    labels=batch["labels"],
                    entity_labels=batch["entity_labels"],
                    relation_matrix=batch["relation_matrix"],
                    num_entities=batch["num_entities"],
                )
                loss = self._compute_combined_loss(outputs)
                loss = loss / self.config.gradient_accumulation_steps

            self.scaler.scale(loss).backward()
        else:
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                bbox=batch["bbox"],
                image=batch["image"],
                labels=batch["labels"],
                entity_labels=batch["entity_labels"],
                relation_matrix=batch["relation_matrix"],
                num_entities=batch["num_entities"],
            )
            loss = self._compute_combined_loss(outputs)
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

        return loss.item() * self.config.gradient_accumulation_steps

    def _compute_combined_loss(self, outputs: dict) -> torch.Tensor:
        """Weighted sum of SER and RE losses."""
        loss = torch.tensor(0.0, device=self.device)
        if "ser_loss" in outputs:
            loss = loss + self.config.ser_loss_weight * outputs["ser_loss"]
        if "re_loss" in outputs:
            loss = loss + self.config.re_loss_weight * outputs["re_loss"]
        return loss

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set, return SER+RE F1 and calibration metrics."""
        self.model.eval()

        all_ser_preds, all_ser_labels = [], []
        all_re_preds, all_re_labels   = [], []
        all_confidences, all_correct   = [], []

        for batch in self.val_loader:
            batch = self._move_to_device(batch)
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                bbox=batch["bbox"],
                image=batch["image"],
                entity_labels=batch["entity_labels"],
                num_entities=batch["num_entities"],
                return_uncertainty=False,
            )

            # SER predictions
            ser_preds = outputs["ser_logits"].argmax(dim=-1)  # [B, seq]
            label_ids = batch["labels"]

            mask = label_ids != -100
            all_ser_preds.extend(ser_preds[mask].cpu().tolist())
            all_ser_labels.extend(label_ids[mask].cpu().tolist())

            # Confidence (for ECE)
            probs = outputs["ser_logits"].softmax(dim=-1)
            conf  = probs.max(dim=-1).values[mask].cpu().numpy()
            corr  = (ser_preds[mask] == label_ids[mask]).cpu().numpy()
            all_confidences.extend(conf.tolist())
            all_correct.extend(corr.tolist())

            # RE predictions
            if outputs["re_logits"] is not None:
                re_preds = outputs["re_logits"].argmax(dim=-1)    # [B, N, N]
                rel_mat  = batch["relation_matrix"]
                for b_idx, n in enumerate(batch["num_entities"]):
                    all_re_preds.extend(re_preds[b_idx, :n, :n].flatten().cpu().tolist())
                    all_re_labels.extend(rel_mat[b_idx, :n, :n].flatten().cpu().tolist())

        ser_metrics = compute_ser_metrics(all_ser_preds, all_ser_labels)
        re_metrics  = compute_re_metrics(all_re_preds, all_re_labels) if all_re_preds else {"re_f1": 0.0}

        import numpy as np
        from models.uncertainty_module import compute_ece
        ece = compute_ece(np.array(all_confidences), np.array(all_correct))

        metrics = {
            **ser_metrics,
            **re_metrics,
            "ece": ece,
        }

        self.model.train()
        return metrics

    def _move_to_device(self, batch: dict) -> dict:
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            else:
                result[k] = v
        return result

    def _save_checkpoint(self, filename: str, extra: dict = None):
        path = os.path.join(self.config.checkpoint_dir, filename)
        self.model.save_checkpoint(path, extra={
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            **(extra or {}),
        })
        logger.info(f"Checkpoint saved: {path}")
