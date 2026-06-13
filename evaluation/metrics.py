"""
metrics.py
----------
Evaluation metrics for the document intelligence system.

- SER: Sequence-level F1 using seqeval (entity-level, not token-level)
- RE: Binary F1 for relation existence
- ECE: Expected Calibration Error
- Calibration curves data
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

from data.xfund_dataset import ID2LABEL, LABEL2ID

try:
    from seqeval.metrics import f1_score as seq_f1
    from seqeval.metrics import precision_score as seq_precision
    from seqeval.metrics import recall_score as seq_recall
    from seqeval.metrics import classification_report as seq_report
    SEQEVAL_AVAILABLE = True
except ImportError:
    SEQEVAL_AVAILABLE = False


def ids_to_bio_tags(label_ids: List[int]) -> List[str]:
    """Convert label IDs to BIO tag strings, skipping ignored (-100) tokens."""
    return [ID2LABEL[lid] for lid in label_ids if lid != -100]


def compute_ser_metrics(
    predictions: List[int],
    labels: List[int],
) -> Dict[str, float]:
    """
    Compute SER metrics.

    Uses seqeval for entity-level F1 if available,
    falls back to token-level F1 via sklearn.

    Args:
        predictions: Flat list of predicted token label IDs.
        labels: Flat list of ground-truth token label IDs (-100 excluded upstream).

    Returns:
        Dict with ser_f1, ser_precision, ser_recall, and per-class metrics.
    """
    if not predictions:
        return {"ser_f1": 0.0, "ser_precision": 0.0, "ser_recall": 0.0}

    preds_arr = np.array(predictions)
    labels_arr = np.array(labels)

    # Token-level F1 (always available)
    token_f1 = f1_score(labels_arr, preds_arr, average="micro", zero_division=0)
    token_prec = precision_score(labels_arr, preds_arr, average="micro", zero_division=0)
    token_rec = recall_score(labels_arr, preds_arr, average="micro", zero_division=0)

    metrics = {
        "ser_f1":           float(token_f1),
        "ser_precision":    float(token_prec),
        "ser_recall":       float(token_rec),
        "ser_token_f1":     float(token_f1),
    }

    # Per-class F1
    class_f1 = f1_score(labels_arr, preds_arr, average=None, zero_division=0)
    label_names = [ID2LABEL[i] for i in range(len(ID2LABEL))]
    for i, name in enumerate(label_names):
        if i < len(class_f1):
            metrics[f"ser_f1_{name}"] = float(class_f1[i])

    # Entity-level F1 via seqeval
    if SEQEVAL_AVAILABLE:
        try:
            # seqeval expects List[List[str]] (one list per document)
            # Here we treat the whole flat list as one "document" for simplicity
            pred_tags  = [ids_to_bio_tags(predictions)]
            label_tags = [ids_to_bio_tags(labels)]

            entity_f1   = seq_f1(label_tags, pred_tags, zero_division=0)
            entity_prec = seq_precision(label_tags, pred_tags, zero_division=0)
            entity_rec  = seq_recall(label_tags, pred_tags, zero_division=0)

            metrics["ser_f1"]          = float(entity_f1)
            metrics["ser_precision"]   = float(entity_prec)
            metrics["ser_recall"]      = float(entity_rec)
            metrics["ser_entity_f1"]   = float(entity_f1)
        except Exception:
            pass  # fall through to token-level

    return metrics


def compute_re_metrics(
    predictions: List[int],
    labels: List[int],
) -> Dict[str, float]:
    """
    Compute Relation Extraction F1.

    Binary classification: 0 = no relation, 1 = key-value relation.

    Args:
        predictions: Flat list of predicted relation labels.
        labels: Flat list of ground-truth relation labels.

    Returns:
        Dict with re_f1, re_precision, re_recall.
    """
    if not predictions:
        return {"re_f1": 0.0, "re_precision": 0.0, "re_recall": 0.0}

    preds_arr = np.array(predictions)
    labels_arr = np.array(labels)

    # Binary F1 on positive class (relation=1)
    re_f1   = f1_score(labels_arr, preds_arr, pos_label=1, average="binary", zero_division=0)
    re_prec = precision_score(labels_arr, preds_arr, pos_label=1, average="binary", zero_division=0)
    re_rec  = recall_score(labels_arr, preds_arr, pos_label=1, average="binary", zero_division=0)

    return {
        "re_f1":        float(re_f1),
        "re_precision": float(re_prec),
        "re_recall":    float(re_rec),
    }


def compute_per_language_metrics(
    per_lang_preds:  Dict[str, List[int]],
    per_lang_labels: Dict[str, List[int]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute SER + RE metrics per language.

    Args:
        per_lang_preds:  {lang: [pred_ids, ...]}
        per_lang_labels: {lang: [label_ids, ...]}

    Returns:
        {lang: {metric: value, ...}}
    """
    results = {}
    for lang in per_lang_preds:
        preds  = per_lang_preds[lang]
        labels = per_lang_labels[lang]
        results[lang] = compute_ser_metrics(preds, labels)
    return results


def compute_aggregate_metrics(
    per_lang_metrics: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """
    Macro-average across languages.

    Returns:
        Aggregate metrics with 'avg_' prefix.
    """
    all_f1s = [m["ser_f1"] for m in per_lang_metrics.values() if "ser_f1" in m]
    return {
        "avg_ser_f1": float(np.mean(all_f1s)) if all_f1s else 0.0,
        "std_ser_f1": float(np.std(all_f1s)) if all_f1s else 0.0,
    }


def format_results_table(
    per_lang_metrics: Dict[str, Dict[str, float]],
    metrics_to_show: List[str] = ("ser_f1", "re_f1"),
) -> str:
    """
    Format a markdown-style results table (mirrors the LayoutXLM paper format).
    """
    langs = sorted(per_lang_metrics.keys())
    header = "| Metric | " + " | ".join(l.upper() for l in langs) + " | Avg |"
    sep    = "|--------|" + "--------|" * (len(langs) + 1)
    rows = [header, sep]

    for metric in metrics_to_show:
        values = [per_lang_metrics.get(l, {}).get(metric, 0.0) for l in langs]
        avg = np.mean(values)
        row = f"| {metric} | " + " | ".join(f"{v:.4f}" for v in values) + f" | {avg:.4f} |"
        rows.append(row)

    return "\n".join(rows)
