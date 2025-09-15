# src/cfg.py

import yaml

def load_config(path: str = "config.yaml") -> dict:
    """Load YAML config file"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
