"""Reusable model loader for KushagraWadhwa/medical-prescription-ocr-india."""

import logging
import threading
from typing import Optional, Tuple, Any

from config.settings import CONFIG
from models.memory_manager import MemoryManager, HardwareProfile

logger = logging.getLogger(__name__)

class ModelLoader:
    """Singleton model loader to ensure the model is loaded only once and reused across prescriptions."""

    _instance: Optional["ModelLoader"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelLoader, cls).__new__(cls)
                cls._instance._model = None
                cls._instance._processor = None
                cls._instance._loaded_model_id = None
                cls._instance._device = None
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def get_hardware_profile(self) -> HardwareProfile:
        return MemoryManager.get_hardware_profile()

    def load_model(
        self,
        model_id: Optional[str] = None,
        force_reload: bool = False,
        allow_cpu_low_ram: bool = False,
    ) -> Tuple[Any, Any]:
        """Loads KushagraWadhwa/medical-prescription-ocr-india with memory checks and singleton reuse.

        Args:
            model_id: Hugging Face model ID. Defaults to CONFIG.model.model_id.
            force_reload: Force re-download / reload if True.
            allow_cpu_low_ram: Explicitly bypass RAM safety guard if user wants to attempt CPU load.

        Returns:
            Tuple of (model, processor).
        """
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        target_model_id = model_id or CONFIG.model.model_id

        with self._lock:
            if self.is_loaded and not force_reload and self._loaded_model_id == target_model_id:
                logger.info(f"Reusing already loaded model: {target_model_id}")
                return self._model, self._processor

            # Memory check before loading
            hw = self.get_hardware_profile()
            logger.info(f"Hardware status: {hw.device_name}, RAM Free: {hw.available_ram_gb} GB, CUDA: {hw.has_cuda}")

            if not hw.can_run_local_3b and not allow_cpu_low_ram:
                raise MemoryError(
                    f"Safety Guard: Cannot load 7GB model locally on this machine. {hw.reason} "
                    f"To attempt loading anyway, set allow_cpu_low_ram=True (risk of freezing Windows), "
                    f"or use Cloud HF Inference API with this exact model."
                )

            # Determine precision and device configuration
            if hw.has_cuda:
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                device_map = "auto"
            else:
                dtype = torch.float32
                device_map = "cpu"

            logger.info(f"Loading {target_model_id} with dtype={dtype}, device_map={device_map}...")

            # Clean memory before loading
            MemoryManager.clean_memory()

            processor = AutoProcessor.from_pretrained(target_model_id)

            # Quantization support check
            quantization_config = None
            try:
                import bitsandbytes
                if hw.has_cuda:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    logger.info("Using 4-bit NF4 quantization on CUDA.")
            except ImportError:
                quantization_config = None

            load_kwargs = {
                "torch_dtype": dtype,
                "device_map": device_map,
                "low_cpu_mem_usage": True,
            }
            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config

            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                target_model_id,
                **load_kwargs
            )

            self._model = model
            self._processor = processor
            self._loaded_model_id = target_model_id
            self._device = "cuda" if hw.has_cuda else "cpu"

            logger.info(f"Successfully loaded {target_model_id} on {self._device}.")
            return self._model, self._processor

    def unload_model(self) -> None:
        """Unload model from memory."""
        with self._lock:
            self._model = None
            self._processor = None
            self._loaded_model_id = None
            MemoryManager.clean_memory()
            logger.info("Model unloaded and memory cleared.")
