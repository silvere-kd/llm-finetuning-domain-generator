# src/steps/compare_step.py

"""
STEP E — Baseline vs Finetuned comparison
- Calls the comparison utilities (from src/compare_runs.py)
- Produces delta artifacts:
    - comparison.csv
    - summary.json
    - report_compare.md
    - dimensions.json (if details.jsonl available for both runs)
"""

from pathlib import Path
from typing import Optional

# Import the helpers directly to avoid spawning a subprocess.
# (These functions exist in your src/compare_runs.py.)
from compare_runs import (
    read_metrics_csv,
    read_details_jsonl,
    align_and_compare,
    dimension_deltas,
    write_csv,
    write_json,
    write_report_md,
)


def run_compare_step(
    baseline_dir: str = "outputs/baseline_eval_openai",
    finetuned_dir: str = "outputs/finetuned_eval_openai",
    out_dir: str = "outputs/compare",
) -> str:
    """
    Compare baseline vs finetuned results and write delta artifacts.

    Args:
        baseline_dir: folder with baseline metrics.csv (+ optional details.jsonl)
        finetuned_dir: folder with finetuned metrics.csv (+ optional details.jsonl)
        out_dir: folder to write comparison outputs

    Returns:
        The output directory path (as string).
    """
    bdir = Path(baseline_dir)
    fdir = Path(finetuned_dir)
    odir = Path(out_dir)
    odir.mkdir(parents=True, exist_ok=True)

    # --- Load required metrics ---
    base_metrics = read_metrics_csv(bdir / "metrics.csv")
    ft_metrics = read_metrics_csv(fdir / "metrics.csv")

    # --- Core alignment + delta computation ---
    comp_rows, summary = align_and_compare(base_metrics, ft_metrics)

    # --- Optional per-dimension deltas (if details.jsonl exist) ---
    dims = None
    bdet = bdir / "details.jsonl"
    fdet = fdir / "details.jsonl"
    if bdet.exists() and fdet.exists():
        base_details = read_details_jsonl(bdet)
        ft_details = read_details_jsonl(fdet)
        dims = dimension_deltas(base_details, ft_details)

    # --- Write artifacts ---
    write_csv(
        odir / "comparison.csv",
        comp_rows,
        ["id", "business_desc", "baseline_mean_overall", "finetuned_mean_overall", "delta"],
    )
    write_json(odir / "summary.json", summary)
    write_report_md(odir / "report_compare.md", summary, dims)
    if dims:
        write_json(odir / "dimensions.json", dims)

    print(
        "[compare_step] Wrote:\n"
        f"  - {odir/'comparison.csv'}\n"
        f"  - {odir/'summary.json'}\n"
        f"  - {odir/'report_compare.md'}"
        + (f"\n  - {odir/'dimensions.json'}" if dims else "")
    )
    return str(odir)
