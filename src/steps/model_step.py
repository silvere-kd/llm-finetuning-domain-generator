# src/steps/model_step.py
"""
STEP 2 — Model loading & prediction generation
- Load base foundation model (HF 4-bit or Unsloth later)
- Generate predictions on validation prompts
"""

import pathlib, json
from datasets import load_dataset
from model_hf import load_model      # HF 4-bit loader
from generator import generate_lists

def run_model_step(cfg: dict, use_finetuned: bool = False, out_dir: str = "outputs/baseline") -> str:
    """
    Load model + tokenizer, run generation on validation prompts.
    Returns path to predictions.jsonl
    """
    print("[model_step] Loading validation set...")
    val = load_dataset("json", data_files=cfg["dataset"]["processed"]["val_path"])["train"]
    pool = [{"id": i, "business_desc": r["prompt"].split("Business:",1)[-1].strip()}
            for i, r in enumerate(val)]

    print("[model_step] Loading model...")
    model, tok = load_model(cfg, use_finetuned=use_finetuned)

    descs = [p["business_desc"] for p in pool]
    '''
    gens = generate_lists(
        model, tok, descs,
        cfg["baseline"]["max_new_tokens"],
        cfg["baseline"]["temperature"],
        cfg["baseline"]["top_p"]
    )
    '''
    gens = generate_lists(
        model, tok, descs,
        cfg["baseline"]["max_new_tokens"],
        cfg["baseline"]["temperature"],
        cfg["baseline"]["top_p"],
        batch_size=cfg["baseline"].get("gen_batch_size", 4),
    )

    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for item, suggs in zip(pool, gens):
            rec = {"id": item["id"], "business_desc": item["business_desc"], "suggestions": suggs}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[model_step] Predictions saved -> {pred_path}")
    return str(pred_path)
