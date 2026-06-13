"""
ocr_engine.py
-------------
EasyOCR-based text extraction for document images.

Returns:
  - words: list of extracted word strings
  - bboxes: list of [x0, y0, x1, y1] in pixel coordinates
  - raw_text: concatenated text
"""

import logging
from typing import List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


class OCREngine:
    """
    EasyOCR interface used by the inference pipeline.

    Args:
        engine: Kept for backward compatibility. Only "easyocr" is supported.
        languages: EasyOCR language codes. Defaults to English.
        confidence_threshold: minimum OCR word confidence to include.
        use_gpu: whether EasyOCR should use GPU for OCR itself.
    """

    def __init__(
        self,
        engine: str = "easyocr",
        languages: Optional[List[str]] = None,
        confidence_threshold: float = 0.0,
        use_gpu: bool = False,
    ):
        if engine.lower() != "easyocr":
            raise ValueError("Only EasyOCR is supported for text extraction.")

        self.engine = "easyocr"
        self.confidence_threshold = confidence_threshold
        self._backend = None
        self._init_easyocr(languages or ["en"], use_gpu=use_gpu)

    def _init_easyocr(self, languages: List[str], use_gpu: bool):
        try:
            import easyocr

            self._backend = easyocr.Reader(languages, gpu=use_gpu)
            logger.info("EasyOCR initialized (languages=%s, gpu=%s)", languages, use_gpu)
        except ImportError:
            logger.warning("easyocr not installed. Install with: pip install easyocr")
            self._backend = None

    def extract(
        self,
        image: Image.Image,
    ) -> Tuple[List[str], List[List[int]], str]:
        """
        Run EasyOCR on a PIL image.

        Returns:
            words: list of word strings
            bboxes: list of [x0, y0, x1, y1] pixel bounding boxes
            raw_text: full concatenated text
        """
        if self._backend is None:
            logger.warning("EasyOCR backend is unavailable. Returning empty extraction.")
            return [], [], ""

        try:
            import numpy as np

            img_array = np.array(image)
            results = self._backend.readtext(img_array, detail=1)
        except Exception as exc:
            logger.error("EasyOCR extraction failed: %s", exc)
            return [], [], ""

        words, bboxes = [], []
        for bbox_pts, text, confidence in results:
            text = text.strip()
            if not text or confidence < self.confidence_threshold:
                continue

            xs = [point[0] for point in bbox_pts]
            ys = [point[1] for point in bbox_pts]
            x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

            word_list = text.split()
            if not word_list:
                continue

            word_width = max(1, (x1 - x0) // len(word_list))
            for index, word in enumerate(word_list):
                wx0 = x0 + index * word_width
                wx1 = min(wx0 + word_width, x1)
                words.append(word)
                bboxes.append([wx0, y0, wx1, y1])

        raw_text = " ".join(words)
        return words, bboxes, raw_text


class MockOCREngine:
    """
    Mock OCR for tests or demos where EasyOCR should not be loaded.
    Returns synthetic words at evenly-spaced bounding boxes.
    """

    def extract(
        self,
        image: Image.Image,
    ) -> Tuple[List[str], List[List[int]], str]:
        width, height = image.size
        mock_words = [
            "Invoice", "No:", "INV-2024-00123",
            "Date:", "15-04-2024",
            "Vendor:", "ABC", "Solutions", "Pvt.", "Ltd.",
            "Total", "Amount:", "$12,540.00",
            "Tax", "Amount:", "$1,254.00",
        ]
        bboxes = []
        cols = 3
        word_width = width // cols
        word_height = height // (len(mock_words) // cols + 1)
        for index, _ in enumerate(mock_words):
            col = index % cols
            row = index // cols
            x0 = col * word_width
            y0 = row * word_height
            bboxes.append([x0, y0, x0 + word_width - 4, y0 + word_height - 4])

        return mock_words, bboxes, " ".join(mock_words)
