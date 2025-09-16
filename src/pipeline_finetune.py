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

import argparse, yaml, pathlib, json
from steps.train_step import run_train_step
from steps.model_step import run_model_step
from steps.scoring_step import run_scoring_step
from steps.analysis_step import run_analysis_step
from steps.compare_step import run_compare_step
from cfg import load_config
from run_manager import RunManager

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["hf", "unsloth"], default="hf",
                    help="Choose finetuning backend.")
    args = ap.parse_args()

    cfg = load_config()
    with RunManager(cfg, run_name_suffix=f"finetune-{args.engine}") as rm:
        rm.log_params({"phase": "finetune", "engine": args.engine})

        # A) Train (saves merged model to cfg["output_dir"])
        run_train_step(cfg, engine=args.engine)
        rm.log_dir(cfg["output_dir"], artifact_path="merged_model", patterns=[".json",".txt",".safetensors",".bin",".model",".vocab",".jsonl"])

        # B) Generate predictions with finetuned model
        pred_path = run_model_step(cfg, use_finetuned=True, out_dir="outputs/finetuned")
        rm.log_artifact(str(pred_path))

        # C) Score with GPT-4 judge
        #pred_path = "outputs/finetuned/predictions.jsonl"
        score_dir = run_scoring_step(cfg, pred_path, out_dir="outputs/finetuned_eval_openai")
        details_path = str(pathlib.Path(score_dir) / "details.jsonl")
        metrics_path = str(pathlib.Path(score_dir) / "metrics.csv")
        rm.log_artifact(details_path, artifact_path="eval")
        rm.log_artifact(metrics_path, artifact_path="eval")

        # D) Analyze & edge-cases
        analysis_dir = run_analysis_step(details_path, pred_path, out_dir="outputs/finetuned_analysis")
        rm.log_dir(analysis_dir, artifact_path="analysis")

        # Optionally log topline metrics to dashboards
        # Parse mean_overall from summary_metrics.json
        summary = json.loads((pathlib.Path(analysis_dir) / "summary_metrics.json").read_text(encoding="utf-8"))
        rm.log_metrics({"finetuned_mean_overall": summary.get("mean_overall", 0.0)})

        # E) Compare baseline vs finetuned (drops deltas)
        compare_dir = run_compare_step(
            baseline_dir="outputs/baseline_eval_openai",
            finetuned_dir="outputs/finetuned_eval_openai",
            out_dir="outputs/compare",
        )
        rm.log_dir(compare_dir, artifact_path="compare")        


if __name__ == "__main__":
    main()
