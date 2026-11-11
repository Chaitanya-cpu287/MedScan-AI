"""Configuration settings for Medical Prescription OCR & Normalization System."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class HardwareSettings:
    min_ram_for_local_3b_gb: float = 6.0
    preferred_torch_dtype: str = "bfloat16"
    fallback_torch_dtype: str = "float32"
    device_map: str = "auto"
    enable_cpu_offload: bool = True

@dataclass
class ModelSettings:
    model_id: str = "KushagraWadhwa/medical-prescription-ocr-india"
    base_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    max_new_tokens: int = 512
    temperature: float = 0.0
    do_sample: bool = False
    hf_token_env_var: str = "HF_TOKEN"

@dataclass
class PreprocessingSettings:
    supported_extensions: tuple = (".jpg", ".jpeg", ".png", ".webp")
    max_dimension: int = 1800
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple = (8, 8)
    denoise_strength: int = 3
    deskew_max_angle: float = 15.0

@dataclass
class NormalizationSettings:
    fuzzy_match_threshold: int = 80
    high_confidence_threshold: int = 90
    ambiguity_threshold: int = 70
    unclear_placeholder: str = "Unclear - Requires verification"
    missing_placeholder: str = "Not specified"

@dataclass
class SystemConfig:
    base_dir: Path = BASE_DIR
    output_dir: Path = BASE_DIR / "outputs"
    sample_data_dir: Path = BASE_DIR / "sample_data"
    hardware: HardwareSettings = field(default_factory=HardwareSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    preprocessing: PreprocessingSettings = field(default_factory=PreprocessingSettings)
    normalization: NormalizationSettings = field(default_factory=NormalizationSettings)

    def ensure_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_data_dir.mkdir(parents=True, exist_ok=True)

# Default global instance
CONFIG = SystemConfig()
CONFIG.ensure_directories()
