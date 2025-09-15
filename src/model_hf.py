# src/model_hf.py

from typing import Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def load_model(cfg: dict, use_finetuned: bool = False) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    - If use_finetuned=False: load base foundation model from cfg["model_name"] in 4-bit.
    - If use_finetuned=True:  load the model from cfg["output_dir"] (useful later).
    """
    bf16_ok = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16

    # 4-bit quantization config
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model_path = cfg["output_dir"] if use_finetuned else cfg["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=bnb_cfg,
        # torch_dtype is ignored when load_in_4bit=True, but fine to leave None
        trust_remote_code=False,  # set True only if your model repo requires it
    )
    # Ensure caching during generation
    if getattr(model, "config", None) is not None:
        model.config.use_cache = True
        model.generation_config.pad_token_id = tokenizer.eos_token_id  # extra safety

    return model, tokenizer
