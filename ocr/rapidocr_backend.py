"""RapidOCR ONNX local fallback engine for fast, zero-GPU offline extraction."""

import logging
from typing import Dict, Any, List
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class RapidOCREngine:
    """Lightweight, CPU-optimized OCR engine using ONNX runtime."""

    def __init__(self):
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                self._engine = RapidOCR()
                logger.info("RapidOCR engine initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize RapidOCR: {e}")
                raise
        return self._engine

    def extract_text(self, image: Image.Image) -> str:
        """Runs OCR on image and returns formatted text preserving vertical structure."""
        engine = self._get_engine()
        img_np = np.array(image.convert("RGB"))
        
        result, elapse_list = engine(img_np)
        if not result:
            return ""

        # result is a list of [box, text, score]
        # Sort lines vertically by top y-coordinate
        lines = []
        for item in result:
            box, text, score = item
            # box has 4 points: [top-left, top-right, bottom-right, bottom-left]
            top_y = min(p[1] for p in box)
            left_x = min(p[0] for p in box)
            lines.append((top_y, left_x, text.strip(), score))

        # Sort primarily by y (with a line height tolerance), secondarily by x
        lines.sort(key=lambda x: (round(x[0] / 15) * 15, x[1]))

        extracted_lines = [item[2] for item in lines if item[2]]
        return "\n".join(extracted_lines)
