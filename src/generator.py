# src/generator.py

from typing import List
import torch
from utils_json import extract_json_array
from templates.prompts import SFT_PROMPT_TEMP

'''
def generate_lists(model, tokenizer, business_descs: List[str], max_new: int, temp: float, top_p: float) -> List[List[str]]:
    """
    Generate a list of domain lists (one list per business description).
    Uses standard HF generation; safe on 4-bit models.
    """
    prompts = [SFT_PROMPT_TEMP.format(desc=b) for b in business_descs]

    # Encode as a batch
    encodings = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=min(getattr(tokenizer, "model_max_length", 2048), 1024),
    ).to(model.device)

    ## Make sure KV cache is enabled
    if getattr(model, "config", None) is not None:
        model.config.use_cache = True

    with torch.inference_mode():
        outputs = model.generate(
            **encodings,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [extract_json_array(t) for t in texts]
'''

def _gen_batch(model, tok, prompts, max_new, temp, top_p):
    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=min(getattr(tok, "model_max_length", 2048), 1024),
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            pad_token_id=tok.eos_token_id,
            use_cache=True,
        )
    texts = tok.batch_decode(out, skip_special_tokens=True)
    return [extract_json_array(t) for t in texts]

def generate_lists(model, tok, business_descs: List[str], max_new: int, temp: float, top_p: float,
                   batch_size: int = 4) -> List[List[str]]:
    """Chunked generation to reduce CUDA OOM/unknown errors."""
    all_out: List[List[str]] = []
    # Prebuild prompts
    prompts = [SFT_PROMPT_TEMP.format(desc=b) for b in business_descs]
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i+batch_size]
        all_out.extend(_gen_batch(model, tok, chunk, max_new, temp, top_p))
        # help the allocator between chunks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return all_out
