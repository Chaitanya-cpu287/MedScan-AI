"""Inference pipeline for KushagraWadhwa/medical-prescription-ocr-india (Local & Cloud HF)."""

import os
import io
import json
import logging
from typing import Optional, Dict, Any, Union
from PIL import Image

from config.settings import CONFIG
from models.model_loader import ModelLoader
from models.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


def run_local_vlm_inference(
    image: Image.Image,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 512,
    allow_cpu_low_ram: bool = False,
) -> str:
    """Executes local inference on the exact fine-tuned Qwen2.5-VL-3B checkpoint."""
    import torch
    loader = ModelLoader()
    model, processor = loader.load_model(allow_cpu_low_ram=allow_cpu_low_ram)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = getattr(loader, "_device", "cpu")
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    
    # Move inputs to target device
    if device == "cuda" and torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=CONFIG.model.do_sample,
            temperature=CONFIG.model.temperature if CONFIG.model.do_sample else None,
        )

    # Slice generated tokens
    input_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_len:]
    result = processor.decode(generated_tokens, skip_special_tokens=True)
    return result.strip()


def run_hf_cloud_inference(
    image: Image.Image,
    system_prompt: str,
    user_prompt: str,
    model_id: Optional[str] = None,
    hf_token: Optional[str] = None,
    max_new_tokens: int = 512,
) -> str:
    """Executes remote cloud inference on Hugging Face Inference API for the exact KushagraWadhwa model.

    Ideal for laptops with 8 GB RAM where loading 7 GB local weights into memory is constrained.
    """
    import base64
    import requests
    from huggingface_hub import InferenceClient

    target_model = model_id or CONFIG.model.model_id
    token = hf_token or os.environ.get(CONFIG.model.hf_token_env_var)

    # Convert PIL Image to base64 data URI
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{img_b64}"

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    client = InferenceClient(model=target_model, token=token)
    response = client.chat.completions.create(
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=CONFIG.model.temperature,
    )
    return response.choices[0].message.content.strip()


def run_text_only_inference(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 512,
    allow_cpu_low_ram: bool = False,
) -> str:
    """Executes text-only inference using Qwen2.5-VL-3B to test text normalization capabilities."""
    import torch
    loader = ModelLoader()
    model, processor = loader.load_model(allow_cpu_low_ram=allow_cpu_low_ram)

    sys_p = system_prompt or "You are an expert medical transcriptionist. Clean and normalize the raw prescription text."
    messages = [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": prompt},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = getattr(loader, "_device", "cpu")
    inputs = processor(text=[text], return_tensors="pt")

    if device == "cuda" and torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_len:]
    result = processor.decode(generated_tokens, skip_special_tokens=True)
    return result.strip()
