# src/steps/scoring_step.py

"""
STEP 3 — Predictions scoring
- Judge predictions with GPT-4 / GPT-4o
"""

import pathlib
#from evaluator import evaluate_predictions
from judge_openai import judge as evaluate_predictions

def run_scoring_step(cfg, pred_path: str, out_dir: str = "outputs/baseline_eval_openai"):
    """
    Runs OpenAI judge on predictions.jsonl
    Writes:
      - details.jsonl
      - metrics.csv
    """
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    print("[scoring_step] Running evaluation with GPT-4...")
    #evaluate_predictions(pred_path=pred_path, out_dir=str(out))
    evaluate_predictions(str(pred_path),
                         out_dir=str(out),
                         model_name=cfg["eval"]["judge_model"],
                         weights=cfg["eval"]["rubric_weights"])
    print(f"[scoring_step] Scoring completed -> {out}")
    return str(out) 
