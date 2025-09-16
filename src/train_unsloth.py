# src/train_unsloth.py

import os, yaml, torch
from typing import Dict
#from datasets import load_dataset
from data_format import load_sft_dataset
from model_unsloth import load_model
from unsloth import FastLanguageModel
from peft import LoraConfig
from cfg import load_config
import gc

cfg = load_config()

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

def fmt(ex: Dict) -> str:
    return f"<s>[INST] {ex['prompt']} [/INST]\n{ex['response']}</s>"

def main():
    torch.manual_seed(cfg["seed"])

    max_len = cfg["train"]["max_seq_len"]
    bf16 = torch.cuda.is_bf16_supported()

    # load base model
    print("[model_step] Loading model...")
    base_model, tokenizer = load_model(cfg, use_finetuned=False)    
    base_model.config.use_cache = False  # better for training

    # Inject LoRA
    lcfg = make_lora_cfg(cfg)
    FastLanguageModel.get_peft_model(
        base_model,
        r=lcfg.r,
        lora_alpha=lcfg.lora_alpha,
        lora_dropout=lcfg.lora_dropout,
        target_modules=lcfg.target_modules,
        bias="none",
        use_gradient_checkpointing=bool(cfg["unsloth"]["gradient_checkpointing"]),
        random_state=cfg["seed"],
    )

    # Datasets -> map to 'text'
    '''
    ds_train = load_dataset("json", data_files=cfg["dataset"]["processed"]["train_path"])["train"]
    ds_val   = load_dataset("json", data_files=cfg["dataset"]["processed"]["val_path"])["train"]
    ds_train = ds_train.map(lambda ex: {"text": fmt(ex)}, remove_columns=ds_train.column_names)
    ds_val   = ds_val.map(lambda ex: {"text": fmt(ex)}, remove_columns=ds_val.column_names)
    '''

    ds_train, ds_val = load_sft_dataset(cfg)

    # Trainer
    trainer = FastLanguageModel.get_efficient_trainer(
        model=base_model,
        tokenizer=tokenizer,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        max_seq_length=max_len,
        per_device_train_batch_size=cfg["train"]["micro_batch_size"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        learning_rate=cfg["train"]["lr"],
        num_train_epochs=cfg["train"]["epochs"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        weight_decay=cfg["train"]["weight_decay"],
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=3,
        lr_scheduler_type="cosine",
        fp16=not bf16,
        bf16=bf16,
        output_dir=cfg["output_dir"],
        packing=True,
    )

    trainer.train()

    # MERGE LoRA → base weights & save as plain HF model 
    # Unsloth returns a PEFT-wrapped model under the hood.
    # merge_and_unload() is available on PEFT models to fold LoRA into the base.
    merged = trainer.model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    # Free some memory    
    del base_model
    del tokenizer
    del trainer
    del ds_train
    del ds_val
    torch.cuda.empty_cache()
    gc.collect()

    print(f"[train_unsloth] Training complete. Merged model saved -> {cfg['output_dir']}")

if __name__ == "__main__":
    main()
