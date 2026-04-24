"""
Model loading utilities for the full pipeline V2.

Reuses the same model loader as V1 — no changes needed for model loading.
Handles:
  - Base model loading (Phi-3.5-mini-instruct)
  - LoRA adapter loading for silver-label extraction
  - GPU placement and dtype optimisation
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel

from .config import BASE_MODEL_ID, CHECKPOINT_PATH

log = logging.getLogger(__name__)


def load_model(
    gpu_id: int = 0,
    base_model_id: str = BASE_MODEL_ID,
    checkpoint_path: Optional[str] = CHECKPOINT_PATH,
    load_adapters: bool = True,
    multi_gpu: bool = False,
) -> Tuple[object, object, str]:
    """
    Load the model on the specified GPU(s).

    Returns:
        (model, tokenizer, device_string)
    """
    num_visible = torch.cuda.device_count()

    if multi_gpu and num_visible > 1:
        log.info(f"Loading base model '{base_model_id}' across {num_visible} GPUs (device_map=auto) ...")
        device = "auto"
    else:
        device = f"cuda:{gpu_id}"
        log.info(f"Loading base model '{base_model_id}' on {device} ...")

    is_phi = "phi" in base_model_id.lower()

    tokenizer_source = checkpoint_path if (load_adapters and checkpoint_path and Path(checkpoint_path).exists()) else base_model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "auto":
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "trust_remote_code": not is_phi,
        }
    else:
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": {"": device},
            "trust_remote_code": not is_phi,
        }
    if is_phi:
        model_kwargs["attn_implementation"] = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)

    if load_adapters and checkpoint_path and Path(checkpoint_path).exists():
        log.info(f"Loading LoRA adapters from '{checkpoint_path}' ...")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        log.info("Merging LoRA weights into base model (merge_and_unload) ...")
        model = model.merge_and_unload()

    model.eval()

    if device == "auto":
        first_param_device = str(next(model.parameters()).device)
        log.info(f"Model sharded across {num_visible} GPUs. First-layer device: {first_param_device}")
        input_device = first_param_device
    else:
        input_device = device

    log.info("Model loaded successfully.")
    return model, tokenizer, input_device


def call_model(
    model,
    tokenizer,
    device: str,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    use_raw_prompt: bool = False,
) -> str:
    """
    Run a single inference call.

    Returns:
        Generated text (assistant response only).
    """
    if use_raw_prompt:
        input_text = (
            f"<|system|>\n{system_prompt}<|end|>\n"
            f"<|user|>\n{user_prompt}<|end|>\n"
            f"<|assistant|>\n"
        )
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    log.debug("=" * 60)
    log.debug("INPUT TO MODEL (first 1500 chars):")
    log.debug(input_text[:1500])
    log.debug("=" * 60)

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_new_tokens + 8192,
    ).to(device)

    input_len = inputs["input_ids"].shape[1]
    log.info(f"Prompt token count: {input_len} | Generation budget: {max_new_tokens}")

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs, streamer=streamer)

    new_tokens = outputs[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=False)

    log.debug("=" * 60)
    log.debug("RAW MODEL OUTPUT (first 1500 chars):")
    log.debug(response[:1500])
    log.debug("=" * 60)

    for tag in ["<|end|>", "<|endoftext|>", "</s>"]:
        response = response.replace(tag, "")
    response = response.strip()

    # Detect degeneration
    if len(response) > 50:
        from collections import Counter
        char_counts = Counter(response)
        most_common_char, most_common_count = char_counts.most_common(1)[0]
        if most_common_count / len(response) > 0.4 and most_common_char not in '{}"[], \n:':
            log.warning(
                f"Degenerate output detected ('{most_common_char}' is "
                f"{most_common_count}/{len(response)} = "
                f"{most_common_count/len(response):.0%}). Truncating."
            )
            degen_pattern = most_common_char * 10
            degen_idx = response.find(degen_pattern)
            if degen_idx > 0:
                response = response[:degen_idx].rstrip()

    return response
