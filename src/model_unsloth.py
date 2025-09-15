# src/model_unsloth.py

import torch
from typing import Tuple
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(cfg: dict, use_finetuned: bool = False) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Return (model, tokenizer) on GPU with the right dtype/quantization."""
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16

    if use_finetuned:
        # Load from saved output_dir (adapters merged by trainer)
        tokenizer = AutoTokenizer.from_pretrained(cfg["output_dir"], use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            cfg["output_dir"], device_map="auto", dtype=dtype
        )
        # Encourage caching anyway
        model.config.use_cache = True
        return model, tokenizer

    # Baseline path: Unsloth accelerated + 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["train"]["max_seq_len"],
        load_in_4bit=True,
        dtype=dtype,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ## Enable Unsloth inference mode (sets up KV cache correctly)
    #model = FastLanguageModel.for_inference(model)
    #model.config.use_cache = True

    return model, tokenizer
