"""
train.py
--------
Main entry point for training the uncertainty-aware multilingual document intelligence model.

Supports:
  --mode language_specific  : Train+eval on one language
  --mode zero_shot          : Train on EN (FUNSD), eval on target
  --mode multitask          : Train on all 7 XFUND languages jointly

Example:
    python train.py --config configs/base_config.yaml --mode multitask
    python train.py --config configs/base_config.yaml --mode language_specific --lang zh
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import torch
import yaml

# Set up logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train uncertainty-aware multilingual document intelligence")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--mode", type=str, choices=["language_specific", "zero_shot", "multitask"],
                        default="multitask")
    parser.add_argument("--lang", type=str, default=None,
                        help="Target language for language_specific/zero_shot modes")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    # Load config
    config_dict = load_config(args.config)
    if args.data_dir:
        config_dict["paths"]["data_dir"] = args.data_dir
    if args.checkpoint_dir:
        config_dict["paths"]["checkpoint_dir"] = args.checkpoint_dir
    if args.no_wandb:
        config_dict["wandb"]["enabled"] = False

    set_seed(config_dict["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Determine languages ──────────────────────────────────────
    all_langs = config_dict["data"]["languages"]  # ["zh", "ja", "es", "fr", "it", "de", "pt"]

    if args.mode == "multitask":
        train_langs = all_langs
        eval_langs  = all_langs
    elif args.mode == "language_specific":
        assert args.lang, "--lang required for language_specific mode"
        train_langs = [args.lang]
        eval_langs  = [args.lang]
    elif args.mode == "zero_shot":
        # Train on FUNSD (English), eval on target
        train_langs = ["en"]
        eval_langs  = [args.lang] if args.lang else all_langs

    logger.info(f"Training mode : {args.mode}")
    logger.info(f"Train languages: {train_langs}")
    logger.info(f"Eval  languages: {eval_langs}")

    # ── Build dataset index ──────────────────────────────────────
    from data.download_xfund import build_dataset_index
    data_dir = config_dict["paths"]["data_dir"]
    data_index = build_dataset_index(data_dir)

    if not data_index:
        logger.error(
            f"No XFUND data found in {data_dir}.\n"
            f"Run: python data/download_xfund.py --output_dir {data_dir}"
        )
        sys.exit(1)

    logger.info(f"Dataset index: {list(data_index.keys())}")

    # ── Build dataloaders ────────────────────────────────────────
    from data.xfund_dataset import build_dataloaders
    train_loader, val_loader = build_dataloaders(
        data_index=data_index,
        tokenizer_name=config_dict["model"]["backbone"],
        max_seq_length=config_dict["data"]["max_seq_length"],
        batch_size=config_dict["training"]["batch_size"],
        languages=train_langs,
        num_workers=4,
    )
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Build model ──────────────────────────────────────────────
    from models.layout_xlm_uncertainty import LayoutXLMUncertainty

    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        model = LayoutXLMUncertainty.from_pretrained_checkpoint(args.resume_from)
    else:
        model = LayoutXLMUncertainty(
            model_name_or_path=config_dict["model"]["backbone"],
            num_labels_ser=config_dict["model"]["num_labels_ser"],
            num_entity_types=config_dict["model"]["num_entity_types"],
            hidden_size=config_dict["model"]["hidden_size"],
            dropout_rate=config_dict["model"]["dropout_rate"],
            mc_dropout_rate=config_dict["model"]["mc_dropout_rate"],
            mc_forward_passes=config_dict["model"]["mc_forward_passes"],
        )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {num_params:,}")

    # ── Build trainer ────────────────────────────────────────────
    from training.trainer import XFUNDTrainer, TrainingConfig

    train_cfg = TrainingConfig(
        num_epochs=config_dict["training"]["num_epochs"],
        batch_size=config_dict["training"]["batch_size"],
        gradient_accumulation_steps=config_dict["training"]["gradient_accumulation_steps"],
        learning_rate=config_dict["training"]["learning_rate"],
        weight_decay=config_dict["training"]["weight_decay"],
        warmup_ratio=config_dict["training"]["warmup_ratio"],
        max_grad_norm=config_dict["training"]["max_grad_norm"],
        fp16=config_dict["training"]["fp16"] and torch.cuda.is_available(),
        eval_steps=config_dict["training"]["eval_steps"],
        save_steps=config_dict["training"]["save_steps"],
        logging_steps=config_dict["training"]["logging_steps"],
        early_stopping_patience=config_dict["training"]["early_stopping_patience"],
        ser_loss_weight=config_dict["training"]["ser_loss_weight"],
        re_loss_weight=config_dict["training"]["re_loss_weight"],
        checkpoint_dir=config_dict["paths"]["checkpoint_dir"],
        log_dir=config_dict["paths"]["log_dir"],
        use_wandb=config_dict["wandb"]["enabled"] and not args.no_wandb,
        wandb_project=config_dict["wandb"]["project"],
        seed=config_dict["training"]["seed"],
    )

    trainer = XFUNDTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        device=device,
    )

    # ── Train ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    trainer.train()

    # ── Final evaluation per language ────────────────────────────
    logger.info("\nFinal per-language evaluation:")
    from evaluation.metrics import format_results_table

    per_lang_results = {}
    for lang in eval_langs:
        if lang not in data_index:
            continue
        lang_paths = data_index[lang]
        if "val" not in lang_paths:
            continue

        from data.xfund_dataset import XFUNDDataset, xfund_collate_fn
        from torch.utils.data import DataLoader

        lang_val_ds = XFUNDDataset(
            json_path=lang_paths["val"],
            lang=lang,
            tokenizer_name=config_dict["model"]["backbone"],
            max_seq_length=config_dict["data"]["max_seq_length"],
            for_training=False,
        )
        lang_val_loader = DataLoader(lang_val_ds, batch_size=8, collate_fn=xfund_collate_fn)

        # Quick eval
        from evaluation.metrics import compute_ser_metrics
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in lang_val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                out = model(**{k: batch[k] for k in ["input_ids", "attention_mask", "bbox", "image",
                                                       "entity_labels", "num_entities"]})
                preds = out["ser_logits"].argmax(dim=-1)
                lbls  = batch["labels"]
                mask  = lbls != -100
                all_preds.extend(preds[mask].cpu().tolist())
                all_labels.extend(lbls[mask].cpu().tolist())

        metrics = compute_ser_metrics(all_preds, all_labels)
        per_lang_results[lang] = metrics
        logger.info(f"  [{lang.upper()}] SER F1: {metrics['ser_f1']:.4f}")

    if per_lang_results:
        table = format_results_table(per_lang_results, ["ser_f1"])
        logger.info("\nResults Table:\n" + table)


if __name__ == "__main__":
    main()
