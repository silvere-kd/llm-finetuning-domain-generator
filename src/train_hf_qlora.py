# src/train_hf_qlora.py

import os, json, yaml
from typing import Dict
import torch
#from datasets import load_dataset
from data_format import load_sft_dataset
from model_hf import load_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from cfg import load_config
import gc

cfg = load_config()

OUTPUT_DIR = cfg["output_dir"]
MAX_LEN = cfg["train"]["max_seq_len"]

# Utilities
## LoRA config
def make_lora_cfg(c: Dict) -> LoraConfig:
    l = c["lora"]
    return LoraConfig(
        r=l["r"],
        lora_alpha=l["alpha"],
        lora_dropout=l["dropout"],
        target_modules=l["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

## Prompt formatting
def format_example(ex):
    """
    Keep the same instruction structure used for dataset_prep to avoid
    train/infer mismatch.
    """
    return f"<s>[INST] {ex['prompt']} [/INST]\n{ex['response']}</s>"

def main():
    torch.manual_seed(cfg["seed"])

    # load base model
    print("[model_step] Loading model...")
    base_model, tokenizer = load_model(cfg, use_finetuned=False)    
    base_model.config.use_cache = False  # better for training

    # prepare for k-bit training & wrap with LoRA
    base_model = prepare_model_for_kbit_training(base_model)
    lora_cfg = make_lora_cfg(cfg)
    lora_model = get_peft_model(base_model, lora_cfg)
    lora_model.print_trainable_parameters()

    # datasets
    '''
    ds_train = load_dataset("json", data_files=cfg["dataset"]["processed"]["train_path"])["train"]
    ds_val   = load_dataset("json", data_files=cfg["dataset"]["processed"]["val_path"])["train"]

    

    # Map dataset columns to what TRL expects in this mode
    def to_prompt_completion(ex):
        # Put your instruction wrapper in the prompt part;
        # leave the raw JSON array as the completion.
        return {
            "prompt": f"<s>[INST] {ex['prompt']} [/INST]\n",  # include trailing newline if you like
            "completion": ex["response"],                     # keep as-is (JSON array string)
        }

    ds_train_pc = ds_train.map(to_prompt_completion, remove_columns=ds_train.column_names)
    ds_val_pc   = ds_val.map(to_prompt_completion,   remove_columns=ds_val.column_names)
    '''

    ds_train, ds_val = load_sft_dataset(cfg)

    # TRL SFTTrainer config
    sft_cfg = SFTConfig(
        output_dir=OUTPUT_DIR,                         # temp while training
        per_device_train_batch_size=cfg["train"]["micro_batch_size"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        num_train_epochs=cfg["train"]["epochs"],
        max_length=MAX_LEN,
        learning_rate=cfg["train"]["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=cfg["train"]["warmup_ratio"],
        weight_decay=cfg["train"]["weight_decay"],
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=3,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
        packing=True,   # pack short samples into full sequences
    )

    trainer = SFTTrainer(
        model=lora_model,
        #tokenizer=tokenizer,
        processing_class=tokenizer,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        #dataset_text_field="text",
        args=sft_cfg,
    )

    trainer.train()

    # MERGE LoRA into base weights for simple inference
    # After training, `trainer.model` is a PEFT model. Merge & save.
    merged = trainer.model.merge_and_unload()   # returns a plain HF model
    # Enable cache for inference
    merged.config.use_cache = True
    # Save merged model & tokenizer to OUTPUT_DIR (overwrites with merged)
    merged.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Free some memory    
    del base_model
    del tokenizer
    del trainer
    del ds_train
    del ds_val
    torch.cuda.empty_cache()
    gc.collect()

    print(f"[train_hf_qlora] Training complete. Merged model saved -> {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
