# src/pipeline_baseline.py
"""
Orchestration of the full baseline pipeline.
"""

import json, yaml, pathlib
from steps.data_step import run_data_step
from steps.model_step import run_model_step
from steps.scoring_step import run_scoring_step
from steps.analysis_step import run_analysis_step
from cfg import load_config
from run_manager import RunManager


def main():
    cfg = load_config()

    with RunManager(cfg, run_name_suffix="baseline") as rm:
        rm.log_params({"phase": "baseline"})

        # STEP 1 — Data
        run_data_step()

        # STEP 2 — Model + Predictions
        pred_path = run_model_step(cfg, out_dir="outputs/baseline")
        rm.log_artifact(str(pred_path))  # predictions.jsonl

        # STEP 3 — Scoring
        #pred_path = "outputs/baseline/predictions.jsonl"
        score_dir = run_scoring_step(cfg, pred_path, out_dir="outputs/baseline_eval_openai")
        details_path = str(pathlib.Path(score_dir) / "details.jsonl")
        metrics_path = str(pathlib.Path(score_dir) / "metrics.csv")
        rm.log_artifact(details_path, artifact_path="eval")
        rm.log_artifact(metrics_path, artifact_path="eval")

        # STEP 4 — Analysis
        analysis_dir = run_analysis_step(details_path, pred_path, out_dir="outputs/baseline_analysis")
        rm.log_dir(analysis_dir, artifact_path="analysis")

        # Optionally log topline metrics to dashboards
        # Parse mean_overall from summary_metrics.json
        summary = json.loads((pathlib.Path(analysis_dir) / "summary_metrics.json").read_text(encoding="utf-8"))
        rm.log_metrics({"baseline_mean_overall": summary.get("mean_overall", 0.0)})

if __name__ == "__main__":
    main()
