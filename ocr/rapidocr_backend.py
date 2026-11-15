"""RapidOCR ONNX local engine with Hybrid TrOCR Handwriting Enhancement."""

import logging
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

from ocr.handwriting_ocr import HandwritingOCREngine, fuse_rapid_and_trocr, normalize_illumination_rgb
from normalization.medicine_matcher import MedicineMatcher

logger = logging.getLogger(__name__)


class RapidOCREngine:
    """Lightweight, CPU-optimized OCR engine with automatic TrOCR handwriting enhancement."""

    HANDWRITING_SCORE_THRESHOLD = 0.78

    def __init__(self):
        self._engine = None
        self._hw_engine: Optional[HandwritingOCREngine] = None
        self._matcher: Optional[MedicineMatcher] = None

    def _get_engine(self):
        if self._engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                self._engine = RapidOCR(
                    det_db_thresh=0.25,
                    det_db_box_thresh=0.4,
                    det_db_unclip_ratio=1.8,
                )
                logger.info("RapidOCR engine initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize RapidOCR: {e}")
                raise
        return self._engine

    def _get_hw_engine(self) -> HandwritingOCREngine:
        if self._hw_engine is None:
            self._hw_engine = HandwritingOCREngine()
        return self._hw_engine

    def _get_matcher(self) -> MedicineMatcher:
        if self._matcher is None:
            self._matcher = MedicineMatcher()
        return self._matcher

    @staticmethod
    def _group_into_lines(result: List[Any]) -> List[Dict[str, Any]]:
        """Groups OCR bounding boxes into coherent horizontal text lines."""
        raw_items = []
        for item in result:
            box, text, score = item
            txt = text.strip()
            if not txt:
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            top_y, bot_y = float(min(ys)), float(max(ys))
            left_x, right_x = float(min(xs)), float(max(xs))
            raw_items.append({
                "top_y": top_y,
                "bot_y": bot_y,
                "center_y": (top_y + bot_y) / 2.0,
                "height": max(1.0, bot_y - top_y),
                "left_x": left_x,
                "right_x": right_x,
                "text": txt,
                "score": float(score),
            })

        if not raw_items:
            return []

        raw_items.sort(key=lambda it: (it["center_y"], it["left_x"]))

        lines: List[List[Dict[str, Any]]] = []
        for it in raw_items:
            placed = False
            for line_group in lines:
                ref = line_group[-1]
                avg_h = (ref["height"] + it["height"]) / 2.0
                if abs(it["center_y"] - ref["center_y"]) <= max(10.0, avg_h * 0.55):
                    line_group.append(it)
                    placed = True
                    break
            if not placed:
                lines.append([it])

        merged_lines = []
        for group in lines:
            group.sort(key=lambda it: it["left_x"])
            merged_text = " ".join(it["text"] for it in group)
            min_score = min(it["score"] for it in group)
            avg_score = sum(it["score"] for it in group) / len(group)
            merged_lines.append({
                "top_y": min(it["top_y"] for it in group),
                "bot_y": max(it["bot_y"] for it in group),
                "left_x": min(it["left_x"] for it in group),
                "right_x": max(it["right_x"] for it in group),
                "text": merged_text,
                "min_score": min_score,
                "avg_score": avg_score,
            })

        return merged_lines

    @classmethod
    def _needs_handwriting_pass(cls, line: Dict[str, Any]) -> bool:
        """Determines whether a line should be refined by the handwritten TrOCR model."""
        import re
        txt = line["text"].strip()
        width = line["right_x"] - line["left_x"]
        # Do not let TrOCR hallucinate over clear dosing schedules or metadata lines
        if re.search(r"(?:\d-\d-\d|-\d-\d\s*x|\bx\s*\d+\s*day)", txt, re.IGNORECASE):
            return False
        if re.match(r"^(?:dr\.|date\s*:|age\s*/?\s*sex)", txt, re.IGNORECASE) and line["min_score"] >= 0.72:
            return False
        if line["min_score"] < cls.HANDWRITING_SCORE_THRESHOLD:
            return True
        if len(txt) <= 2 and width > 40:
            return True
        return False

    @staticmethod
    def _detect_handwritten_line_boxes_cv(img_np: np.ndarray) -> List[Dict[str, Any]]:
        """OpenCV morphological line-band fallback when DBNet misses faint/messy handwriting."""
        norm_rgb = normalize_illumination_rgb(img_np)
        gray = cv2.cvtColor(norm_rgb, cv2.COLOR_RGB2GRAY)
        _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Connect characters horizontally into line bands
        h_img, w_img = gray.shape[:2]
        kx = max(25, w_img // 25)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 3))
        dilated = cv2.dilate(inv, kernel, iterations=1)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

        boxes = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if w >= 40 and 10 <= h <= h_img * 0.25 and area >= 200:
                boxes.append({
                    "top_y": float(y),
                    "bot_y": float(y + h),
                    "left_x": float(x),
                    "right_x": float(x + w),
                    "text": "",
                    "min_score": 0.0,
                    "avg_score": 0.0,
                })
        boxes.sort(key=lambda b: (b["top_y"], b["left_x"]))
        return boxes

    def extract_text(self, image: Image.Image, enable_handwriting_hybrid: bool = True) -> str:
        """Runs OCR on image, automatically engaging TrOCR for handwritten lines when needed."""
        engine = self._get_engine()
        rgb_img = image.convert("RGB")
        img_np = np.array(rgb_img)

        result, _ = engine(img_np)
        merged_lines = self._group_into_lines(result) if result else []

        # Multi-pass detection for faint, shadowed, or messy phone-photo prescriptions
        if len(merged_lines) < 4 and rgb_img.height > 350:
            norm_np = normalize_illumination_rgb(img_np)
            res_norm, _ = engine(norm_np)
            norm_lines = self._group_into_lines(res_norm) if res_norm else []
            if len(norm_lines) > len(merged_lines):
                merged_lines = norm_lines

        # Morphological CV line-band fallback if detector still missed faint handwriting
        if len(merged_lines) < 2 and rgb_img.height > 200:
            cv_lines = self._detect_handwritten_line_boxes_cv(img_np)
            if len(cv_lines) > len(merged_lines):
                merged_lines = cv_lines

        if not merged_lines:
            return ""

        full_preview = " ".join(ln["text"].lower() for ln in merged_lines)
        is_printed_label = any(
            kw in full_preview
            for kw in ("tablets ip", "capsules ip", "each uncoated", "each film", "mfg. lic", "batch no", "mrp:", "schedule h")
        )

        hw_indices = [
            idx for idx, line in enumerate(merged_lines)
            if self._needs_handwriting_pass(line)
        ]

        if enable_handwriting_hybrid and hw_indices and not is_printed_label:
            hw_engine = self._get_hw_engine()
            if hw_engine.is_available():
                crops = []
                for idx in hw_indices:
                    ln = merged_lines[idx]
                    h_line = ln["bot_y"] - ln["top_y"]
                    pad_y = max(4, int(h_line * 0.25))
                    pad_x = 10
                    x1 = max(0, int(ln["left_x"]) - pad_x)
                    y1 = max(0, int(ln["top_y"]) - pad_y)
                    x2 = min(rgb_img.width, int(ln["right_x"]) + pad_x)
                    y2 = min(rgb_img.height, int(ln["bot_y"]) + pad_y)
                    crops.append(rgb_img.crop((x1, y1, x2, y2)))

                trocr_preds = hw_engine.recognize_crops(crops)
                matcher = self._get_matcher()
                for idx, tr_txt in zip(hw_indices, trocr_preds):
                    if tr_txt:
                        fused = fuse_rapid_and_trocr(
                            rapid_text=merged_lines[idx]["text"],
                            trocr_text=tr_txt,
                            matcher=matcher,
                        )
                        merged_lines[idx]["text"] = fused
            else:
                # Lightweight CPU fallback when TrOCR is not cached: 2x upscale + unsharp pass
                avg_conf = sum(ln["avg_score"] for ln in merged_lines) / len(merged_lines)
                if avg_conf < 0.85:
                    up = cv2.resize(img_np, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                    blur = cv2.GaussianBlur(up, (0, 0), 2.0)
                    sharp = cv2.addWeighted(up, 1.4, blur, -0.4, 0)
                    res_up, _ = engine(sharp)
                    if res_up:
                        up_lines = self._group_into_lines(res_up)
                        if up_lines:
                            up_conf = sum(ln["avg_score"] for ln in up_lines) / len(up_lines)
                            if up_conf > avg_conf:
                                merged_lines = up_lines

        extracted_lines = [ln["text"].strip() for ln in merged_lines if ln["text"].strip()]
        return "\n".join(extracted_lines)
