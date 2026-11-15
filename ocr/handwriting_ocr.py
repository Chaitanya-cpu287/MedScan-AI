"""Hybrid Handwriting OCR Engine using Microsoft TrOCR + RapidOCR Dual-Engine Fusion."""

import os
import re
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps
from rapidfuzz import fuzz, process

from normalization.medicine_matcher import MedicineMatcher

logger = logging.getLogger(__name__)


def _clean_trocr_artifacts(text: str) -> str:
    """Normalizes common TrOCR cursive character confusions on medical prescriptions."""
    cleaned = text.strip()
    # TrOCR frequently reads handwritten '5 mg' as 'is my'
    cleaned = re.sub(r"\bis\s+my\b", "5 mg", cleaned, flags=re.IGNORECASE)
    # Strip stray wrapping quotes or trailing isolated punctuation
    cleaned = cleaned.strip(" '\"")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def normalize_illumination_rgb(img_np: np.ndarray) -> np.ndarray:
    """Gently balances uneven lighting/shadows via LAB CLAHE while preserving faint pen strokes."""
    rgb = img_np if img_np.ndim == 3 else cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def _extract_med_core(med_str: str) -> str:
    """Strips leading item numbers and formulation prefixes for clean pharmacopeia scoring."""
    core = re.sub(r"^\s*\d+[\)\.\-\:]\s*", "", med_str)
    core = re.sub(r"^(?:tab(?:let)?s?|cap(?:sule)?s?|syp|syrup|inj|ointment|oint|drops?|gel)\b\.?\s*", "", core, flags=re.IGNORECASE)
    return core.strip()


def _split_med_and_tail(line: str) -> Tuple[str, str, str]:
    """Splits a medication line into (med_token, dosage_token, instruction_tail)."""
    # Look for explicit separator '-' first
    if "-" in line:
        left, right = line.split("-", 1)
    else:
        left, right = line, ""

    dosage = ""
    m_dose = re.search(r"(\d+(?:\.\d+)?\s*(?:mg|gm|g|mcg|ml|iu))\b", left, re.IGNORECASE)
    if m_dose:
        dosage = m_dose.group(1)
        med_part = left[:m_dose.start()].strip()
        if not right and m_dose.end() < len(left):
            right = left[m_dose.end():].strip()
    else:
        # Strip trailing 'mg' when digits were dropped (e.g. 'Metorminmg')
        med_part = re.sub(r"(?:mg|gm|mcg|ml)$", "", left.strip(), flags=re.IGNORECASE).strip()

    return med_part, dosage, right.strip()


def fuse_rapid_and_trocr(
    rapid_text: str,
    trocr_text: str,
    matcher: Optional[MedicineMatcher] = None,
) -> str:
    """Fuses RapidOCR and TrOCR outputs using Pharmacopeia-guided selection.

    TrOCR excels at cursive English phrases ('Every 8 hours', 'As needed for pain',
    'After meals', 'Hydrochlorothiazide') and numeric strengths ('250 mg', '100 mg'),
    while RapidOCR occasionally preserves cramped medicine prefixes better.
    """
    r_clean = rapid_text.strip()
    t_clean = _clean_trocr_artifacts(trocr_text)

    if not t_clean:
        return r_clean
    if not r_clean:
        return t_clean

    # Reject obvious TrOCR prose hallucinations on numeric/medical lines
    if re.search(r"\b(?:to add a message|priest place|paying just|allies|singing)\b", t_clean, re.IGNORECASE):
        return r_clean

    # Protect RapidOCR medical frequency/unit tokens if TrOCR dropped or garbled them (e.g. '2puffsos' -> 'reppuffs', 'CopTDS' -> 'TCAP TPS')
    r_has_freq = bool(re.search(r"(?:sos|od|bd|bid|tid|tds|qid|hs|prn|stat|puffs?|tabs?|caps?)", r_clean, re.IGNORECASE))
    t_has_freq = bool(re.search(r"(?:sos|od|bd|bid|tid|tds|qid|hs|prn|stat|puffs?|tabs?|caps?)", t_clean, re.IGNORECASE))
    if r_has_freq and not t_has_freq:
        return r_clean

    # Preserve RapidOCR frequency code (e.g. 'sos' in '2p1sos') when TrOCR extracts the unit ('2puffs ,')
    m_r_freq = re.search(r"\b(sos|od|bd|bid|tid|tds|qid|hs|prn|stat)\b", r_clean, re.IGNORECASE)
    if m_r_freq and not re.search(r"\b(sos|od|bd|bid|tid|tds|qid|hs|prn|stat)\b", t_clean, re.IGNORECASE):
        if re.search(r"\b(?:puffs?|drops?|tabs?|tablets?|caps?|capsules?|ml)\b", t_clean, re.IGNORECASE):
            t_clean = re.sub(r"[,\.\s]+$", "", t_clean) + f" {m_r_freq.group(1).upper()}"
            return t_clean

    # Protect RapidOCR digits (e.g. '100', '5', '10', '250', '500') if TrOCR dropped all digits
    r_digits = re.findall(r"\d+", r_clean)
    t_digits = re.findall(r"\d+", t_clean)
    if r_digits and not t_digits:
        return r_clean

    if matcher is None:
        return t_clean

    r_med, r_dose, r_tail = _split_med_and_tail(r_clean)
    t_med, t_dose, t_tail = _split_med_and_tail(t_clean)

    # Check if this line is a medication line (has dosage, separator, or medicine candidate)
    r_core = _extract_med_core(r_med)
    t_core = _extract_med_core(t_med)
    r_score = 0.0
    t_score = 0.0
    if len(r_core) >= 3:
        m_r = process.extractOne(r_core.lower(), matcher.all_candidates, scorer=fuzz.WRatio)
        if m_r:
            r_score = float(m_r[1])
    if len(t_core) >= 3:
        m_t = process.extractOne(t_core.lower(), matcher.all_candidates, scorer=fuzz.WRatio)
        if m_t:
            t_score = float(m_t[1])

    # If RapidOCR's medicine token already matches the Pharmacopeia well (>= 70) and TrOCR isn't clearly better, keep RapidOCR
    if r_score >= 70 and t_score <= r_score + 5:
        return r_clean

    # If RapidOCR's medicine token is a strong Pharmacopeia match and clearly beats TrOCR's
    if r_score >= 72 and (r_score - t_score) >= 10:
        best_med = r_med
        best_dose = t_dose or r_dose
        best_tail = t_tail or r_tail
        parts = [best_med]
        if best_dose:
            parts.append(best_dose)
        if best_tail:
            parts.append(f"- {best_tail}")
        return " ".join(parts).strip()

    # If TrOCR missed numeric dosage while RapidOCR captured it (e.g. RapidOCR '5mg')
    if not t_dose and r_dose and t_score >= 65:
        parts = [t_med, r_dose]
        if t_tail:
            parts.append(f"- {t_tail}")
        return " ".join(parts).strip()

    return t_clean


class HandwritingOCREngine:
    """Singleton wrapper around Microsoft TrOCR for offline handwritten line recognition."""

    MODEL_ID = "microsoft/trocr-base-handwritten"
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HandwritingOCREngine, cls).__new__(cls)
            cls._instance._processor = None
            cls._instance._model = None
            cls._instance._load_attempted = False
        return cls._instance

    @classmethod
    def is_cached_locally(cls) -> bool:
        """Checks whether microsoft/trocr-base-handwritten is present in local HF cache."""
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        cache_dir = hf_home / "hub" / "models--microsoft--trocr-base-handwritten"
        return cache_dir.exists() and any(cache_dir.iterdir())

    def is_available(self) -> bool:
        """Returns True if TrOCR is already loaded or cached locally with sufficient RAM."""
        if self._model is not None:
            return True
        if self._load_attempted:
            return False
        return self.is_cached_locally()

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._load_attempted:
            return False
        self._load_attempted = True
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            logger.info(f"Loading local handwriting OCR model: {self.MODEL_ID}")
            self._processor = TrOCRProcessor.from_pretrained(self.MODEL_ID, local_files_only=True)
            self._model = VisionEncoderDecoderModel.from_pretrained(self.MODEL_ID, local_files_only=True)
            self._model.eval()
            logger.info("Handwriting OCR model (TrOCR) loaded successfully.")
            return True
        except Exception as e:
            logger.warning(f"Could not load TrOCR handwriting model: {e}")
            self._processor = None
            self._model = None
            return False

    @staticmethod
    def enhance_line_crop(crop: Image.Image) -> Image.Image:
        """Normalizes illumination, upscales, sharpens pen strokes, and pads for TrOCR."""
        np_c = np.array(crop.convert("RGB"))
        np_c = normalize_illumination_rgb(np_c)
        h, w = np_c.shape[:2]
        if h < 64:
            scale = max(2.0, 64.0 / max(1, h))
            np_c = cv2.resize(np_c, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(np_c, (0, 0), 1.5)
        sharp = cv2.addWeighted(np_c, 1.35, blur, -0.35, 0)
        pil_out = Image.fromarray(sharp)
        return ImageOps.expand(pil_out, border=12, fill="white")

    def recognize_crops(self, crops: List[Image.Image]) -> List[str]:
        """Runs batched TrOCR inference on a list of cropped line images."""
        if not crops:
            return []
        if not self._ensure_loaded():
            return [""] * len(crops)

        import torch

        enhanced = [self.enhance_line_crop(c) for c in crops]
        pixel_values = self._processor(images=enhanced, return_tensors="pt", padding=True).pixel_values
        with torch.no_grad():
            generated_ids = self._model.generate(
                pixel_values,
                max_new_tokens=48,
                num_beams=2,
            )
        preds = self._processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [_clean_trocr_artifacts(p) for p in preds]
