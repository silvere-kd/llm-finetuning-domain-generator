# src/data_format.py

"""
Unified SFT dataset formatting for both HF+TRL and Unsloth trainers.
- Loads JSONL from cfg["dataset"]["processed"]["train_path"] / cfg["dataset"]["processed"]["val_path"]
- Produces datasets with a single column: "text"
"""

from typing import Tuple, Dict
from datasets import load_dataset

def format_example(ex: Dict) -> str:
    """
    Single canonical format used by *both* trainers.
    Mirrors the prompt style you used in baseline/inference.
    """
    return f"<s>[INST] {ex['prompt']} [/INST]\n{ex['response']}</s>"

def _to_text(ex: Dict) -> Dict:
    return {"text": format_example(ex)}

def load_sft_dataset(cfg: Dict) -> Tuple[object, object]:
    """
    Returns (train_ds, val_ds) each with one column: "text".
    """
    ds_train = load_dataset("json", data_files=cfg["dataset"]["processed"]["train_path"])["train"]
    ds_val   = load_dataset("json", data_files=cfg["dataset"]["processed"]["val_path"])["train"]

    ds_train = ds_train.map(_to_text, remove_columns=ds_train.column_names)
    ds_val   = ds_val.map(_to_text, remove_columns=ds_val.column_names)
    return ds_train, ds_val
