"""
pipeline.py
-----------
End-to-end inference pipeline for document intelligence.

Implements the full 7-step architecture from the diagram:
  1. Input Document (PDF or image)
  2. OCR / Text Extraction
  3. Layout-aware Multilingual Model (LayoutXLM + MC-Dropout)
  4. Field Extraction (SER → named fields)
  5. Confidence Estimation (MC-Dropout ensemble)
  6. Human-review Trigger (confidence < threshold)
  7. Structured JSON Output

Usage:
    pipeline = DocumentIntelligencePipeline.from_checkpoint("checkpoints/best_model.pt")
    result = pipeline.process("invoice.pdf", uncertainty_threshold=0.70)
    print(result.to_json())
"""

import json
import os
import time
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np
from PIL import Image

from inference.ocr_engine import OCREngine
from models.layout_xlm_uncertainty import LayoutXLMUncertainty
from models.uncertainty_module import UncertaintyModule
from data.xfund_dataset import LABEL2ID, ID2LABEL, normalize_bbox, load_tokenizer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Output data classes
# ─────────────────────────────────────────────

@dataclass
class ExtractedField:
    """A single extracted key-value field from a document."""
    field_name: str
    value: str
    confidence: float
    bbox: Optional[List[int]] = None
    uncertainty: Optional[float] = None
    label: Optional[str] = None


@dataclass
class RelationPair:
    """An extracted key-value relation."""
    head_text: str
    tail_text:  str
    confidence: float


@dataclass
class DocumentResult:
    """Complete result for a single document."""
    doc_id: str
    language: str
    fields: List[ExtractedField]
    relations: List[RelationPair]
    overall_confidence: float
    needs_human_review: bool
    review_reason: Optional[str]
    processing_time_ms: float
    raw_ocr_text: Optional[str] = None
    mc_passes_used: int = 20
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string matching the architecture output format."""
        result_dict = {
            "doc_id": self.doc_id,
            "language": self.language,
            "overall_confidence": round(self.overall_confidence, 4),
            "needs_human_review": self.needs_human_review,
            "review_reason": self.review_reason,
            "fields": {
                f.field_name: {
                    "value":       f.value,
                    "confidence":  round(f.confidence, 4),
                    "uncertainty": round(f.uncertainty, 4) if f.uncertainty is not None else None,
                    "bbox":        f.bbox,
                }
                for f in self.fields
            },
            "relations": [
                {
                    "head": r.head_text,
                    "tail": r.tail_text,
                    "confidence": round(r.confidence, 4),
                }
                for r in self.relations
            ],
            "processing_time_ms": round(self.processing_time_ms, 1),
            "mc_passes_used": self.mc_passes_used,
        }
        return json.dumps(result_dict, ensure_ascii=False, indent=indent)


# ─────────────────────────────────────────────
# Field name mapping (SER label → human field)
# ─────────────────────────────────────────────

FIELD_LABEL_MAP = {
    "B-HEADER":   "section_header",
    "B-QUESTION": "field_key",
    "B-ANSWER":   "field_value",
}


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────

class DocumentIntelligencePipeline:
    """
    Full document intelligence pipeline with uncertainty estimation.

    Steps:
      1. Load document (PDF → images or image files)
  2. Run OCR with EasyOCR → words + bboxes
      3. Tokenize + encode for LayoutXLM
      4. Forward pass with MC-Dropout (T passes)
      5. Decode entities + relations
      6. Compute confidence; check threshold
      7. Return structured JSON

    Args:
        model: Trained LayoutXLMUncertainty model.
        tokenizer_name: HuggingFace tokenizer ID.
        ocr_engine: EasyOCR backend name. Kept as an internal parameter for compatibility.
        confidence_threshold: Below this → trigger human review.
        mc_passes: Number of MC-Dropout forward passes.
        max_seq_length: Token sequence length.
        device: torch device.
    """

    def __init__(
        self,
        model: LayoutXLMUncertainty,
        tokenizer_name: str = "microsoft/layoutlmv3-base",
        ocr_engine: str = "easyocr",
        confidence_threshold: float = 0.70,
        mc_passes: int = 20,
        max_seq_length: int = 512,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.model.eval()
        self.confidence_threshold = confidence_threshold
        self.mc_passes = mc_passes
        self.max_seq_length = max_seq_length
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.tokenizer = load_tokenizer(tokenizer_name)
        self.ocr = OCREngine(engine=ocr_engine)
        self.uncertainty_module = UncertaintyModule(
            confidence_threshold=confidence_threshold
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        **kwargs,
    ) -> "DocumentIntelligencePipeline":
        """Load pipeline from a saved model checkpoint."""
        model = LayoutXLMUncertainty.from_pretrained_checkpoint(checkpoint_path)
        kwargs.setdefault("tokenizer_name", model.model_name)
        return cls(model=model, **kwargs)

    def process(
        self,
        input_path: str,
        doc_id: Optional[str] = None,
        language: str = "auto",
        uncertainty_threshold: Optional[float] = None,
    ) -> DocumentResult:
        """
        Process a single document end-to-end.

        Args:
            input_path: Path to PDF or image file.
            doc_id: Optional document identifier.
            language: Language code hint ("auto" = detect from model).
            uncertainty_threshold: Override default confidence threshold.

        Returns:
            DocumentResult with all extracted fields and metadata.
        """
        t_start = time.time()
        threshold = uncertainty_threshold or self.confidence_threshold
        doc_id = doc_id or Path(input_path).stem

        # ── Step 1+2: Load document and run OCR ──────────────────
        logger.info(f"[{doc_id}] Step 1-2: Loading document and running OCR...")
        image, words, word_bboxes, raw_text = self._load_and_ocr(input_path)
        logger.info(f"  OCR extracted {len(words)} words")

        # ── Step 3: Tokenize and encode ──────────────────────────
        logger.info(f"[{doc_id}] Step 3: Encoding with document-layout model...")
        inputs = self._encode(words, word_bboxes, image)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        # ── Step 4+5: Model forward with uncertainty ─────────────
        logger.info(f"[{doc_id}] Step 4-5: Running MC-Dropout inference ({self.mc_passes} passes)...")
        with torch.no_grad():
            outputs = self.model.predict(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                bbox=inputs["bbox"],
                image=inputs.get("image"),
                mc_passes=self.mc_passes,
            )

        uncertainty = outputs["uncertainty"]
        mean_probs  = uncertainty["mean_ser_probs"]  # [1, seq, C]
        confidence  = uncertainty["confidence"]      # [1, seq]
        entropy     = uncertainty["entropy"]         # [1, seq]

        # Document-level confidence (mean over real tokens)
        mask = inputs["attention_mask"].float()
        doc_confidence = float(
            (confidence * mask).sum() / mask.sum().clamp(min=1.0)
        )

        # ── Step 6: Human-review trigger ─────────────────────────
        needs_review = doc_confidence < threshold
        review_reason = None
        if needs_review:
            review_reason = f"Confidence {doc_confidence:.2%} < threshold {threshold:.0%}"
            logger.warning(f"[{doc_id}] {review_reason} → flagging for human review")

        # ── Step 4: Decode entities ───────────────────────────────
        logger.info(f"[{doc_id}] Step 4: Decoding extracted fields...")
        pred_labels = mean_probs[0].argmax(dim=-1).cpu().tolist()   # [seq]
        token_conf  = confidence[0].cpu().tolist()                  # [seq]
        token_ent   = entropy[0].cpu().tolist()                     # [seq]

        fields = self._decode_fields(
            pred_labels, token_conf, token_ent,
            words, word_bboxes,
            inputs["attention_mask"][0].cpu().tolist(),
        )

        # Decode relations if available
        relations = []
        if "re_confidence" in uncertainty:
            re_conf = uncertainty["re_confidence"][0].cpu()  # [N, N]
            relations = self._decode_relations(fields, re_conf)

        # ── Step 7: Build output ──────────────────────────────────
        t_elapsed_ms = (time.time() - t_start) * 1000
        logger.info(f"[{doc_id}] Done in {t_elapsed_ms:.1f}ms | Confidence: {doc_confidence:.2%} | Review: {needs_review}")

        return DocumentResult(
            doc_id=doc_id,
            language=language,
            fields=fields,
            relations=relations,
            overall_confidence=doc_confidence,
            needs_human_review=needs_review,
            review_reason=review_reason,
            processing_time_ms=t_elapsed_ms,
            raw_ocr_text=raw_text,
            mc_passes_used=self.mc_passes,
            metadata={
                "num_words": len(words),
                "avg_entropy": float(np.mean([e for e in token_ent if e > 0])) if token_ent else 0.0,
            },
        )

    def _load_and_ocr(self, input_path: str):
        """Load document and run OCR. Returns (image, words, bboxes, raw_text)."""
        path = Path(input_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            # Convert PDF first page to image
            try:
                from pdf2image import convert_from_path
                pages = convert_from_path(str(path), dpi=150)
                image = pages[0]
            except Exception as e:
                logger.warning(f"pdf2image failed ({e}), creating blank image")
                image = Image.new("RGB", (1000, 1000), "white")
        else:
            image = Image.open(str(path)).convert("RGB")

        words, bboxes, raw_text = self.ocr.extract(image)
        return image, words, bboxes, raw_text

    def _encode(self, words: List[str], word_bboxes: List[List[int]], image: Image.Image) -> dict:
        """Tokenize words and build model inputs."""
        W, H = image.size
        tokens, token_bboxes = [], []

        for word, bbox in zip(words, word_bboxes):
            word_toks = self.tokenizer.tokenize(word) or [self.tokenizer.unk_token]
            norm_bbox = normalize_bbox(bbox, W, H)
            for tok in word_toks:
                tokens.append(tok)
                token_bboxes.append(norm_bbox)

        # Truncate
        max_t = self.max_seq_length - 2
        tokens = tokens[:max_t]
        token_bboxes = token_bboxes[:max_t]

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        input_ids = [cls_id] + self.tokenizer.convert_tokens_to_ids(tokens) + [sep_id]
        bbox_seq  = [[0,0,0,0]] + token_bboxes + [[1000]*4]

        seq_len = len(input_ids)
        pad_len = self.max_seq_length - seq_len
        attn_mask = [1]*seq_len + [0]*pad_len
        input_ids = input_ids + [pad_id]*pad_len
        bbox_seq  = bbox_seq  + [[0,0,0,0]]*pad_len

        # Process image
        img = image.resize((224, 224))
        img_arr = torch.tensor(list(img.getdata()), dtype=torch.float32)
        img_arr = img_arr.reshape(224, 224, 3).permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img_arr = (img_arr - mean) / std

        return {
            "input_ids":      torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attn_mask], dtype=torch.long),
            "bbox":           torch.tensor([bbox_seq],  dtype=torch.long),
            "image":          img_arr.unsqueeze(0),
        }

    def _decode_fields(
        self,
        pred_labels: List[int],
        token_conf:  List[float],
        token_ent:   List[float],
        words:       List[str],
        word_bboxes: List[List[int]],
        attn_mask:   List[int],
    ) -> List[ExtractedField]:
        """
        Decode BIO predictions into named entity spans.
        Maps token spans back to word spans via simple alignment.
        """
        fields = []
        current_entity = None

        # Skip CLS at position 0; real tokens start at 1
        real_tokens = [(i, pred_labels[i], token_conf[i], token_ent[i])
                       for i in range(1, len(pred_labels))
                       if attn_mask[i] == 1 and i < len(pred_labels) - 1]

        for pos, label_id, conf, ent in real_tokens:
            tag = ID2LABEL.get(label_id, "O")

            if tag.startswith("B-"):
                # Save previous entity
                if current_entity:
                    fields.append(self._finalize_entity(current_entity))

                entity_type = tag[2:]   # strip B-
                word_idx = min(pos - 1, len(words) - 1)
                current_entity = {
                    "type":   entity_type,
                    "tokens": [words[word_idx] if word_idx < len(words) else ""],
                    "confs":  [conf],
                    "ents":   [ent],
                    "bbox":   word_bboxes[word_idx] if word_idx < len(word_bboxes) else None,
                }

            elif tag.startswith("I-") and current_entity:
                entity_type = tag[2:]
                word_idx = min(pos - 1, len(words) - 1)
                if entity_type == current_entity["type"]:
                    current_entity["tokens"].append(words[word_idx] if word_idx < len(words) else "")
                    current_entity["confs"].append(conf)
                    current_entity["ents"].append(ent)
            else:
                if current_entity:
                    fields.append(self._finalize_entity(current_entity))
                    current_entity = None

        if current_entity:
            fields.append(self._finalize_entity(current_entity))

        return fields

    def _finalize_entity(self, entity: dict) -> ExtractedField:
        text = " ".join(entity["tokens"])
        avg_conf = float(np.mean(entity["confs"]))
        avg_ent  = float(np.mean(entity["ents"]))

        type_to_field = {
            "HEADER":   "section_header",
            "QUESTION": "key",
            "ANSWER":   "value",
        }
        field_name = type_to_field.get(entity["type"], entity["type"].lower())

        return ExtractedField(
            field_name=field_name,
            value=text,
            confidence=avg_conf,
            bbox=entity.get("bbox"),
            uncertainty=avg_ent,
            label=entity["type"],
        )

    def _decode_relations(
        self,
        fields: List[ExtractedField],
        re_confidence: torch.Tensor,   # [N, N]
    ) -> List[RelationPair]:
        """Extract high-confidence key-value pairs from RE output."""
        relations = []
        keys   = [f for f in fields if f.label == "QUESTION"]
        values = [f for f in fields if f.label == "ANSWER"]

        for i, key in enumerate(keys):
            for j, val in enumerate(values):
                if i < re_confidence.size(0) and j < re_confidence.size(1):
                    conf = float(re_confidence[i, j])
                    if conf > 0.5:
                        relations.append(RelationPair(
                            head_text=key.value,
                            tail_text=val.value,
                            confidence=conf,
                        ))

        return sorted(relations, key=lambda r: -r.confidence)

    def process_batch(
        self,
        input_paths: List[str],
        **kwargs,
    ) -> List[DocumentResult]:
        """Process multiple documents."""
        return [self.process(p, **kwargs) for p in input_paths]
