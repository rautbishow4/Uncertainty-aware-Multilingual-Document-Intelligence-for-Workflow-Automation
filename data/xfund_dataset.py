"""
xfund_dataset.py
----------------
PyTorch Dataset for the XFUND benchmark.

XFUND JSON format per document:
{
  "id": "...",
  "uid": "...",
  "document": [
    {
      "id": int,
      "text": str,
      "box": [x0, y0, x1, y1],     # pixel coords on original image
      "linking": [[id_head, id_tail], ...],
      "label": "header" | "question" | "answer" | "other",
      "words": [{"text": str, "box": [...]}]
    }, ...
  ],
  "img": {
    "fname": str,
    "width": int,
    "height": int
  }
}

The XFUND release embeds document images as base64 inside the JSON.
"""

import json
import base64
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer

# BIO tag mapping
LABEL2ID = {
    "O":          0,
    "B-HEADER":   1,
    "I-HEADER":   2,
    "B-QUESTION": 3,
    "I-QUESTION": 4,
    "B-ANSWER":   5,
    "I-ANSWER":   6,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

ENTITY_LABEL2ID = {
    "other":    0,
    "header":   1,
    "question": 2,
    "answer":   3,
}

# Map raw XFUND entity labels to BIO prefix
ENTITY_TO_BIO_BASE = {
    "header":   "HEADER",
    "question": "QUESTION",
    "answer":   "ANSWER",
    "other":    None,          # → O tag
}


def load_tokenizer(tokenizer_name: str):
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    except OSError:
        return AutoTokenizer.from_pretrained(tokenizer_name)


def normalize_bbox(bbox: List[int], width: int, height: int, scale: int = 1000) -> List[int]:
    """Normalize pixel bounding box to [0, scale] range."""
    x0, y0, x1, y1 = bbox
    return [
        max(0, min(int(x0 / width * scale), scale)),
        max(0, min(int(y0 / height * scale), scale)),
        max(0, min(int(x1 / width * scale), scale)),
        max(0, min(int(y1 / height * scale), scale)),
    ]


def decode_image_from_json(img_data: dict) -> Optional[Image.Image]:
    """
    XFUND stores images as base64 under img['base64'].
    Falls back to None if not present.
    """
    b64 = img_data.get("base64") or img_data.get("data")
    if b64:
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    return None


class XFUNDDocument:
    """Parsed representation of a single XFUND document."""

    def __init__(self, raw: dict, lang: str):
        self.id = raw.get("id", raw.get("uid", ""))
        self.lang = lang
        self.img_meta = raw.get("img", {})
        self.width = self.img_meta.get("width", 1000)
        self.height = self.img_meta.get("height", 1000)

        # Decode image
        self.image: Optional[Image.Image] = decode_image_from_json(self.img_meta)
        if self.image is not None:
            self.width, self.height = self.image.size

        # Parse entities & words
        self.entities = []        # list of entity dicts
        self.relations = []       # list of (head_id, tail_id) tuples
        self._parse_document(raw["document"])

    def _parse_document(self, doc_items: list):
        entity_map = {}  # entity_id -> entity dict

        for item in doc_items:
            entity_id = item["id"]
            label = item.get("label", "other").lower()

            # Collect word-level tokens & bboxes
            words = []
            word_bboxes = []
            for word in item.get("words", []):
                words.append(word["text"])
                bbox = normalize_bbox(word["box"], self.width, self.height)
                word_bboxes.append(bbox)

            # Fallback: use entity-level text/box if no words
            if not words:
                words = [item.get("text", "")]
                bbox = normalize_bbox(item.get("box", [0, 0, 0, 0]), self.width, self.height)
                word_bboxes = [bbox]

            entity = {
                "id": entity_id,
                "label": label,
                "words": words,
                "bboxes": word_bboxes,
                "entity_label_id": ENTITY_LABEL2ID.get(label, 0),
            }
            self.entities.append(entity)
            entity_map[entity_id] = entity

            # Collect relations
            for link in item.get("linking", []):
                if len(link) == 2:
                    self.relations.append((link[0], link[1]))


class XFUNDDataset(Dataset):
    """
    PyTorch Dataset for XFUND.

    Tokenizes words with a document-layout tokenizer, attaches normalized bounding boxes,
    and constructs BIO labels for SER and binary relation labels for RE.

    Args:
        json_path: Path to a XFUND split JSON file.
        lang: Language code (used for metadata).
        tokenizer_name: HuggingFace tokenizer name.
        max_seq_length: Maximum token sequence length.
        image_size: Target image size (H, W) for visual encoder.
        bbox_scale: Normalization scale for bounding boxes.
        for_training: If True, returns labels; else inference mode.
    """

    def __init__(
        self,
        json_path: str,
        lang: str,
        tokenizer_name: str = "microsoft/layoutlmv3-base",
        max_seq_length: int = 512,
        image_size: Tuple[int, int] = (224, 224),
        bbox_scale: int = 1000,
        for_training: bool = True,
    ):
        self.lang = lang
        self.max_seq_length = max_seq_length
        self.image_size = image_size
        self.bbox_scale = bbox_scale
        self.for_training = for_training

        self.tokenizer = load_tokenizer(tokenizer_name)
        self.documents = self._load(json_path, lang)
        self.features = [self._encode(doc) for doc in self.documents]

    def _load(self, json_path: str, lang: str) -> List[XFUNDDocument]:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [XFUNDDocument(raw, lang) for raw in data["documents"]]

    def _encode(self, doc: XFUNDDocument) -> dict:
        """Convert an XFUNDDocument into model-ready tensors."""
        tokens = []
        token_bboxes = []
        token_labels = []
        word_ids_map = []        # maps token position → entity index

        for ent_idx, entity in enumerate(doc.entities):
            label = entity["label"]
            bio_base = ENTITY_TO_BIO_BASE.get(label, None)

            for w_idx, (word, bbox) in enumerate(zip(entity["words"], entity["bboxes"])):
                word_tokens = self.tokenizer.tokenize(word)
                if not word_tokens:
                    continue

                for t_idx, tok in enumerate(word_tokens):
                    tokens.append(tok)
                    token_bboxes.append(bbox)
                    word_ids_map.append(ent_idx)

                    if bio_base is None:
                        token_labels.append(LABEL2ID["O"])
                    elif t_idx == 0 and w_idx == 0:
                        token_labels.append(LABEL2ID[f"B-{bio_base}"])
                    else:
                        token_labels.append(LABEL2ID[f"I-{bio_base}"])

        # Truncate to max_seq_length - 2 (for [CLS] and [SEP])
        max_tokens = self.max_seq_length - 2
        tokens = tokens[:max_tokens]
        token_bboxes = token_bboxes[:max_tokens]
        token_labels = token_labels[:max_tokens]
        word_ids_map = word_ids_map[:max_tokens]

        # Add special tokens
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        input_ids = [cls_id] + self.tokenizer.convert_tokens_to_ids(tokens) + [sep_id]
        bbox_seq = [[0, 0, 0, 0]] + token_bboxes + [[self.bbox_scale]*4]
        label_seq = [-100] + token_labels + [-100]     # -100 = ignore in loss

        # Pad to max_seq_length
        seq_len = len(input_ids)
        pad_len = self.max_seq_length - seq_len
        attention_mask = [1] * seq_len + [0] * pad_len

        input_ids = input_ids + [pad_id] * pad_len
        bbox_seq = bbox_seq + [[0, 0, 0, 0]] * pad_len
        label_seq = label_seq + [-100] * pad_len

        # Build relation matrix [max_entities, max_entities]
        entity_labels = [e["entity_label_id"] for e in doc.entities]
        num_entities = len(doc.entities)
        entity_ids_map = {e["id"]: i for i, e in enumerate(doc.entities)}
        relation_matrix = torch.zeros(num_entities, num_entities, dtype=torch.long)
        for head_id, tail_id in doc.relations:
            h_idx = entity_ids_map.get(head_id, -1)
            t_idx = entity_ids_map.get(tail_id, -1)
            if h_idx >= 0 and t_idx >= 0:
                relation_matrix[h_idx, t_idx] = 1

        # Process image
        image_tensor = self._process_image(doc.image)

        feature = {
            "input_ids":      torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "bbox":           torch.tensor(bbox_seq, dtype=torch.long),
            "image":          image_tensor,
            "labels":         torch.tensor(label_seq, dtype=torch.long),
            "entity_labels":  torch.tensor(entity_labels, dtype=torch.long),
            "relation_matrix": relation_matrix,
            "word_ids_map":   word_ids_map,
            "doc_id":         doc.id,
            "lang":           doc.lang,
            "num_entities":   num_entities,
        }
        return feature

    def _process_image(self, image: Optional[Image.Image]) -> torch.Tensor:
        """Resize and normalize document image."""
        H, W = self.image_size
        if image is None:
            return torch.zeros(3, H, W)

        img = image.resize((W, H))
        img_array = torch.tensor(list(img.getdata()), dtype=torch.float32)
        img_array = img_array.reshape(H, W, 3).permute(2, 0, 1) / 255.0

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_array = (img_array - mean) / std
        return img_array

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int) -> dict:
        return self.features[idx]


def xfund_collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate function that handles variable-size relation matrices.
    Pads relation matrices to the max entity count in the batch.
    """
    max_entities = max(item["num_entities"] for item in batch)
    collated = {}

    tensor_keys = ["input_ids", "attention_mask", "bbox", "image", "labels"]
    for key in tensor_keys:
        collated[key] = torch.stack([item[key] for item in batch])

    # Pad entity labels
    entity_labels_padded = torch.zeros(len(batch), max_entities, dtype=torch.long)
    for i, item in enumerate(batch):
        n = item["num_entities"]
        entity_labels_padded[i, :n] = item["entity_labels"]
    collated["entity_labels"] = entity_labels_padded

    # Pad relation matrices
    rel_matrices = torch.zeros(len(batch), max_entities, max_entities, dtype=torch.long)
    for i, item in enumerate(batch):
        n = item["num_entities"]
        rel_matrices[i, :n, :n] = item["relation_matrix"]
    collated["relation_matrix"] = rel_matrices

    collated["num_entities"] = [item["num_entities"] for item in batch]
    collated["doc_ids"]      = [item["doc_id"] for item in batch]
    collated["langs"]        = [item["lang"] for item in batch]
    collated["word_ids_map"] = [item["word_ids_map"] for item in batch]

    return collated


def build_dataloaders(
    data_index: Dict[str, Dict[str, str]],
    tokenizer_name: str = "microsoft/layoutlmv3-base",
    max_seq_length: int = 512,
    batch_size: int = 8,
    languages: Optional[List[str]] = None,
    num_workers: int = 4,
):
    """
    Build train and val DataLoaders for one or multiple XFUND languages.

    Args:
        data_index: Output of download_xfund.build_dataset_index().
        languages: Subset of languages; None = all available.

    Returns:
        (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader, ConcatDataset

    langs = languages or list(data_index.keys())

    train_datasets, val_datasets = [], []
    for lang in langs:
        paths = data_index.get(lang, {})
        if "train" in paths:
            train_datasets.append(
                XFUNDDataset(paths["train"], lang, tokenizer_name, max_seq_length, for_training=True)
            )
        if "val" in paths:
            val_datasets.append(
                XFUNDDataset(paths["val"], lang, tokenizer_name, max_seq_length, for_training=False)
            )

    train_loader = DataLoader(
        ConcatDataset(train_datasets),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=xfund_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        ConcatDataset(val_datasets),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=xfund_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
