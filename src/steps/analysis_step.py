# src/steps/analysis_step.py
"""
STEP 4 — Performance analysis
- Summarize scores
- Report rule violations
- Edge-case discovery
"""

import pathlib
from analyze import summarize

def run_analysis_step(details_path: str, preds_path: str, out_dir: str = "outputs/baseline_analysis"):
    """
    Analyze results of baseline run.
    """
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    print("[analysis_step] Analyzing predictions...")
    summarize(details_path=details_path, preds_path=preds_path, out_dir=str(out))
    print(f"[analysis_step] Analysis completed -> {out}")
    return str(out)
