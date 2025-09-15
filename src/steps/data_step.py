# src/steps/data_step.py

from data_synth import main as synth_main
from data_prep import main as prep_main

def run_data_step():
    """
    Runs the synthetic dataset creation and preparation.
    Writes outputs to:
      - data/raw/synth.jsonl
      - data/processed/train.jsonl
      - data/processed/val.jsonl
    """
    print("[data_step] Generating synthetic data...")
    synth_main()

    print("[data_step] Preparing train/val splits...")
    prep_main()

    print("[data_step] Data step completed ✅")
