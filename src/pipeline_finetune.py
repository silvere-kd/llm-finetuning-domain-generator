# src/pipeline_finetune.py

"""
STEP 2 — Finetuning pipeline (modular)
- Train with chosen engine (HF or Unsloth)
- Generate predictions using merged model
- Score with OpenAI judge
- Analyze results (edge-cases, violations, summary)

Usage:
  python src/pipeline_finetune.py --engine hf
  python src/pipeline_finetune.py --engine unsloth
"""

import argparse, yaml, pathlib
from steps.train_step import run_train_step
from steps.model_step import run_model_step
from steps.scoring_step import run_scoring_step
from steps.analysis_step import run_analysis_step
from steps.compare_step import run_compare_step
from cfg import load_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["hf", "unsloth"], default="hf",
                    help="Choose finetuning backend.")
    args = ap.parse_args()

    cfg = load_config()

    # A) Train (saves merged model to cfg["output_dir"])
    run_train_step(cfg, engine=args.engine)

    # B) Generate predictions with finetuned model
    pred_path = run_model_step(cfg, use_finetuned=True, out_dir="outputs/finetuned")

    # C) Score with GPT-4 judge
    pred_path = "outputs/finetuned/predictions.jsonl"
    score_dir = run_scoring_step(cfg, pred_path, out_dir="outputs/finetuned_eval_openai")

    # D) Analyze & edge-cases
    details_path = str(pathlib.Path(score_dir) / "details.jsonl")
    run_analysis_step(details_path, pred_path, out_dir="outputs/finetuned_analysis")

    # E) Compare baseline vs finetuned (drops deltas)
    run_compare_step(
        baseline_dir="outputs/baseline_eval_openai",
        finetuned_dir="outputs/finetuned_eval_openai",
        out_dir="outputs/compare",
    )


if __name__ == "__main__":
    main()
