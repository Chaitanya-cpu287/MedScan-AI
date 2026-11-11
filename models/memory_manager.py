"""Hardware detection, memory estimation, and resource management."""

import gc
import psutil
from dataclasses import dataclass
from typing import Dict, Any, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class HardwareProfile:
    total_ram_gb: float
    available_ram_gb: float
    ram_usage_percent: float
    has_cuda: bool
    device_name: str
    total_vram_gb: float
    available_vram_gb: float
    can_run_local_3b: bool
    recommended_backend: str
    reason: str


class MemoryManager:
    """Monitors system memory and enforces safe execution parameters."""

    # 3B parameters in float16 = ~6.5 - 7.0 GB weights + 1-2 GB activation & KV cache = ~8-9 GB memory
    ESTIMATED_3B_FP16_GB = 7.0
    ESTIMATED_3B_INT4_GB = 2.5

    @classmethod
    def get_hardware_profile(cls) -> HardwareProfile:
        vm = psutil.virtual_memory()
        total_ram = vm.total / (1024 ** 3)
        avail_ram = vm.available / (1024 ** 3)
        ram_percent = vm.percent

        has_cuda = False
        device_name = "CPU"
        total_vram = 0.0
        avail_vram = 0.0

        if TORCH_AVAILABLE and torch.cuda.is_available():
            has_cuda = True
            device_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total_vram = props.total_memory / (1024 ** 3)
            # Free VRAM estimation
            allocated_vram = torch.cuda.memory_allocated(0) / (1024 ** 3)
            avail_vram = max(0.0, total_vram - allocated_vram)

        # Decision logic for running Qwen2.5-VL-3B locally:
        # If CUDA with >= 6.0 GB VRAM: Perfect for local GPU
        # If CPU only: Need at least 9 GB free RAM to avoid OS freeze/crash.
        # On an 8GB total RAM machine (with ~0.5-2GB free), loading 7GB float16 on CPU will OOM/freeze.
        if has_cuda and (avail_vram >= 5.5 or total_vram >= 6.0):
            can_run = True
            backend = "local_cuda"
            reason = f"CUDA GPU available ({device_name}) with {avail_vram:.2f} GB available VRAM."
        elif not has_cuda and avail_ram >= 8.5:
            can_run = True
            backend = "local_cpu"
            reason = f"Sufficient CPU RAM available ({avail_ram:.2f} GB free)."
        else:
            can_run = False
            backend = "cloud_api_or_fallback"
            if not has_cuda:
                reason = (
                    f"Local system has {total_ram:.1f} GB RAM ({avail_ram:.2f} GB free) without dedicated CUDA GPU. "
                    f"Loading the 7.0 GB Qwen2.5-VL-3B model into CPU RAM requires >8.5 GB free to avoid crashing Windows. "
                    "Safe execution: Cloud HF Inference API (exact KushagraWadhwa model) or fast local ONNX engine is recommended."
                )
            else:
                reason = (
                    f"CUDA GPU ({device_name}) has only {avail_vram:.2f} GB free VRAM (needs ~6 GB). "
                    "CPU offload or Cloud HF Inference API recommended."
                )

        return HardwareProfile(
            total_ram_gb=round(total_ram, 2),
            available_ram_gb=round(avail_ram, 2),
            ram_usage_percent=round(ram_percent, 1),
            has_cuda=has_cuda,
            device_name=device_name,
            total_vram_gb=round(total_vram, 2),
            available_vram_gb=round(avail_vram, 2),
            can_run_local_3b=can_run,
            recommended_backend=backend,
            reason=reason,
        )

    @classmethod
    def clean_memory(cls) -> None:
        """Force garbage collection and clear PyTorch cache."""
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
