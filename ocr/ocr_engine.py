"""High-level Prescription OCR Engine supporting KushagraWadhwa/medical-prescription-ocr-india & fallbacks."""

import os
import time
import json
import logging
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
from PIL import Image

from config.settings import CONFIG
from models.memory_manager import MemoryManager
from models.inference import run_local_vlm_inference, run_hf_cloud_inference
from ocr.prompts import (
    FINETUNED_MODEL_SYSTEM_PROMPT,
    FINETUNED_MODEL_USER_PROMPT,
    RAW_OCR_FAITHFUL_SYSTEM_PROMPT,
    RAW_OCR_FAITHFUL_USER_PROMPT,
)
from ocr.rapidocr_backend import RapidOCREngine

logger = logging.getLogger(__name__)


class OCRMode(str, Enum):
    AUTO = "auto"
    LOCAL_QWEN = "local_qwen"
    CLOUD_HF = "cloud_hf"
    RAPID_OCR = "rapid_ocr"


@dataclass
class OCRResult:
    raw_text: str
    backend_used: str
    inference_time_sec: float
    is_structured_json: bool
    parsed_json: Optional[Dict[str, Any]] = None
    character_count: int = 0
    line_count: int = 0
    warnings: list = field(default_factory=list)


class PrescriptionOCREngine:
    """Primary OCR extraction engine implementing the exact fine-tuned Qwen2.5-VL-3B workflow."""

    def __init__(self, default_mode: OCRMode = OCRMode.AUTO):
        self.default_mode = default_mode
        self._rapid_engine = None

    def _get_rapid_engine(self) -> RapidOCREngine:
        if self._rapid_engine is None:
            self._rapid_engine = RapidOCREngine()
        return self._rapid_engine

    def extract(
        self,
        image: Image.Image,
        mode: Optional[Union[OCRMode, str]] = None,
        hf_token: Optional[str] = None,
        allow_cpu_low_ram: bool = False,
    ) -> OCRResult:
        """Extracts text from a prescription image using the specified or auto-selected engine."""
        selected_mode = OCRMode(mode) if mode else self.default_mode
        start_time = time.time()
        backend_used = ""
        raw_text = ""
        parsed_json = None
        is_json = False
        warnings = []

        hw = MemoryManager.get_hardware_profile()

        # Resolve AUTO mode
        if selected_mode == OCRMode.AUTO:
            if hw.can_run_local_3b:
                selected_mode = OCRMode.LOCAL_QWEN
            elif hf_token or "HF_TOKEN" in os.environ:
                selected_mode = OCRMode.CLOUD_HF
            else:
                selected_mode = OCRMode.RAPID_OCR
                warnings.append(
                    f"Low RAM detected ({hw.available_ram_gb} GB free, no CUDA GPU). "
                    "Automatically engaged RapidOCR local engine to prevent OS freezing. "
                    "Provide an HF_TOKEN to run KushagraWadhwa/medical-prescription-ocr-india in cloud GPU mode."
                )

        # 1. LOCAL QWEN EXECUTION
        if selected_mode == OCRMode.LOCAL_QWEN:
            try:
                backend_used = f"Local Qwen2.5-VL ({CONFIG.model.model_id})"
                # First run faithful prompt
                raw_text = run_local_vlm_inference(
                    image=image,
                    system_prompt=RAW_OCR_FAITHFUL_SYSTEM_PROMPT,
                    user_prompt=RAW_OCR_FAITHFUL_USER_PROMPT,
                    allow_cpu_low_ram=allow_cpu_low_ram,
                )
            except Exception as e:
                logger.warning(f"Local Qwen inference failed: {e}. Falling back to RapidOCR.")
                warnings.append(f"Local Qwen error: {str(e)}. Switched to fallback OCR.")
                backend_used = "RapidOCR (Fallback after Local Qwen error)"
                raw_text = self._get_rapid_engine().extract_text(image)

        # 2. CLOUD HF INFERENCE EXECUTION
        elif selected_mode == OCRMode.CLOUD_HF:
            try:
                backend_used = f"Cloud HF Inference API ({CONFIG.model.model_id})"
                raw_text = run_hf_cloud_inference(
                    image=image,
                    system_prompt=RAW_OCR_FAITHFUL_SYSTEM_PROMPT,
                    user_prompt=RAW_OCR_FAITHFUL_USER_PROMPT,
                    hf_token=hf_token,
                )
            except Exception as e:
                logger.warning(f"Cloud HF inference failed: {e}. Falling back to RapidOCR.")
                warnings.append(f"Cloud HF error: {str(e)}. Switched to RapidOCR.")
                backend_used = "RapidOCR (Fallback after Cloud HF error)"
                raw_text = self._get_rapid_engine().extract_text(image)

        # 3. RAPIDOCR LOCAL EXECUTION
        elif selected_mode == OCRMode.RAPID_OCR:
            backend_used = "RapidOCR ONNX (Local CPU)"
            raw_text = self._get_rapid_engine().extract_text(image)

        # Check if output is JSON formatted (e.g. from the fine-tuned model)
        cleaned_text = raw_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()

        if cleaned_text.startswith("{") and cleaned_text.endswith("}"):
            try:
                parsed_json = json.loads(cleaned_text)
                is_json = True
            except Exception:
                pass

        elapsed = time.time() - start_time
        lines = [l for l in raw_text.splitlines() if l.strip()]

        return OCRResult(
            raw_text=raw_text,
            backend_used=backend_used,
            inference_time_sec=round(elapsed, 2),
            is_structured_json=is_json,
            parsed_json=parsed_json,
            character_count=len(raw_text),
            line_count=len(lines),
            warnings=warnings,
        )
