# src/steps/train_step.py

"""
STEP 1 — Finetuning (QLoRA)
- Select engine: "hf" (Transformers+PEFT+TRL) or "unsloth"
- Saves a **merged** model into cfg["output_dir"].
"""

from typing import Literal, Dict


def run_train_step(cfg: Dict, engine: Literal["hf", "unsloth"] = "hf") -> None:
    """
    Dispatch to the chosen trainer. Trainers are expected to:
      - Read config.yaml
      - Train with QLoRA
      - Merge LoRA -> base weights
      - Save merged model & tokenizer to cfg["output_dir"]
    """
    if engine == "hf":
        # HF-only QLoRA (Transformers+PEFT+TRL)
        from train_hf_qlora import main as train_main
        print("[train_step] Engine=hf → starting QLoRA training...")
        train_main()
    elif engine == "unsloth":
        # Unsloth QLoRA
        from train_unsloth import main as train_main
        print("[train_step] Engine=unsloth → starting QLoRA training...")
        train_main()
    else:
        raise ValueError("engine must be 'hf' or 'unsloth'")

    print("[train_step] Training complete ✅ (merged model in cfg['output_dir'])")
