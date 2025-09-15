# src/compare_runs.py

"""
Compare baseline vs finetuned runs.

Inputs (defaults match the pipeline outputs):
- Baseline:
    metrics.csv         (id, business_desc, mean_overall)
    details.jsonl       (optional; per-domain scores for dimension-level stats)
- Finetuned:
    metrics.csv
    details.jsonl       (optional)

Outputs (written to outputs/compare/ by default):
- comparison.csv        (id, business_desc, baseline, finetuned, delta)
- summary.json          (aggregated stats incl. win rate, avg delta)
- report_compare.md     (human-readable summary)
- dimensions.json       (optional; mean per-dimension deltas if details are available)

Usage:
    python src/compare_runs.py \
      --baseline_dir outputs/baseline_eval_openai \
      --finetuned_dir outputs/finetuned_eval_openai \
      --out_dir outputs/compare
"""

import argparse
import csv
import json
import pathlib
import statistics
from typing import Dict, List, Tuple, Optional

# ----------------- IO helpers -----------------
def read_metrics_csv(path: pathlib.Path) -> Dict[str, Dict]:
    """
    Return a dict keyed by stringified id:
      id -> { 'id': str, 'business_desc': str, 'mean_overall': float }
    """
    out = {}
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rid = str(row["id"])
            try:
                mo = float(row["mean_overall"])
            except Exception:
                mo = 0.0
            out[rid] = {
                "id": rid,
                "business_desc": row.get("business_desc", ""),
                "mean_overall": mo,
            }
    return out


def read_details_jsonl(path: pathlib.Path) -> Dict[str, List[Dict]]:
    """
    Return a dict keyed by stringified id:
      id -> [ {domain, relevance, memorability, readability, safety, overall}, ... ]
    """
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = str(rec.get("id"))
            out[rid] = rec.get("details", []) or []
    return out


def write_csv(path: pathlib.Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(path: pathlib.Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------- core comparison -----------------
def align_and_compare(
    base: Dict[str, Dict],
    ft: Dict[str, Dict],
) -> Tuple[List[Dict], Dict]:
    """
    Align by id and compute deltas.
    Returns:
      - rows for comparison.csv
      - summary dict with aggregate stats
    """
    ids = sorted(set(base.keys()) & set(ft.keys()), key=lambda x: (len(x), x))

    rows = []
    deltas = []
    wins = 0
    ties = 0

    for rid in ids:
        b = base[rid]["mean_overall"]
        f = ft[rid]["mean_overall"]
        d = round(f - b, 4)
        deltas.append(d)
        if f > b:
            wins += 1
        elif f == b:
            ties += 1

        # Prefer finetuned desc if available, else baseline
        desc = ft[rid].get("business_desc") or base[rid].get("business_desc") or ""
        rows.append(
            {
                "id": rid,
                "business_desc": desc[:200],
                "baseline_mean_overall": round(b, 4),
                "finetuned_mean_overall": round(f, 4),
                "delta": d,
            }
        )

    # Aggregates
    avg_base = round(statistics.mean([base[r]["mean_overall"] for r in ids]), 4) if ids else 0.0
    avg_ft = round(statistics.mean([ft[r]["mean_overall"] for r in ids]), 4) if ids else 0.0
    avg_delta = round(avg_ft - avg_base, 4)
    win_rate = round(100.0 * wins / len(ids), 2) if ids else 0.0

    summary = {
        "n_common": len(ids),
        "wins": wins,
        "ties": ties,
        "losses": len(ids) - wins - ties,
        "win_rate_percent": win_rate,
        "avg_baseline": avg_base,
        "avg_finetuned": avg_ft,
        "avg_delta": avg_delta,
        "delta_min": round(min(deltas), 4) if deltas else 0.0,
        "delta_max": round(max(deltas), 4) if deltas else 0.0,
        "delta_median": round(statistics.median(deltas), 4) if deltas else 0.0,
    }

    return rows, summary


def dimension_deltas(
    base_details: Dict[str, List[Dict]],
    ft_details: Dict[str, List[Dict]],
) -> Optional[Dict]:
    """
    If details.jsonl are provided for both runs, compute mean deltas per dimension:
    relevance, memorability, readability, safety, overall.
    We average per-prompt first (over suggestions), then average across prompts.
    """
    ids = sorted(set(base_details.keys()) & set(ft_details.keys()))
    if not ids:
        return None

    dims = ["relevance", "memorability", "readability", "safety", "overall"]
    deltas = {k: [] for k in dims}

    def mean_dim(rows: List[Dict], key: str) -> float:
        vals = [float(r.get(key, 0.0)) for r in rows]
        return statistics.mean(vals) if vals else 0.0

    for rid in ids:
        b_rows = base_details.get(rid, [])
        f_rows = ft_details.get(rid, [])
        for k in dims:
            b_mean = mean_dim(b_rows, k)
            f_mean = mean_dim(f_rows, k)
            deltas[k].append(f_mean - b_mean)

    out = {}
    for k, arr in deltas.items():
        out[k] = {
            "avg_delta": round(statistics.mean(arr), 4) if arr else 0.0,
            "median_delta": round(statistics.median(arr), 4) if arr else 0.0,
            "min_delta": round(min(arr), 4) if arr else 0.0,
            "max_delta": round(max(arr), 4) if arr else 0.0,
        }
    return out


# ----------------- report -----------------
def write_report_md(out_path: pathlib.Path, summary: Dict, dims: Optional[Dict]) -> None:
    lines = []
    lines.append("# Baseline vs Finetuned — Comparison\n")
    lines.append(f"- Common prompts compared: **{summary['n_common']}**\n")
    lines.append(f"- Win rate (finetuned > baseline): **{summary['win_rate_percent']}%**  \n")
    lines.append(f"- Avg baseline: **{summary['avg_baseline']}**  \n")
    lines.append(f"- Avg finetuned: **{summary['avg_finetuned']}**  \n")
    lines.append(f"- Avg delta: **{summary['avg_delta']}**  \n")
    lines.append(f"- Delta range: {summary['delta_min']} → {summary['delta_max']} (median {summary['delta_median']})\n")
    lines.append(f"- Wins: {summary['wins']} | Ties: {summary['ties']} | Losses: {summary['losses']}\n")

    if dims:
        lines.append("\n## Per-dimension average deltas (finetuned - baseline)\n")
        for k, s in dims.items():
            lines.append(f"- **{k}**: {s['avg_delta']} (median {s['median_delta']}, min {s['min_delta']}, max {s['max_delta']})")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", default="outputs/baseline_eval_openai",
                    help="Folder with baseline metrics.csv and (optionally) details.jsonl")
    ap.add_argument("--finetuned_dir", default="outputs/finetuned_eval_openai",
                    help="Folder with finetuned metrics.csv and (optionally) details.jsonl")
    ap.add_argument("--out_dir", default="outputs/compare",
                    help="Output folder for comparison artifacts")
    args = ap.parse_args()

    base_dir = pathlib.Path(args.baseline_dir)
    ft_dir = pathlib.Path(args.finetuned_dir)
    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # --- Required: metrics.csv
    base_metrics = read_metrics_csv(base_dir / "metrics.csv")
    ft_metrics = read_metrics_csv(ft_dir / "metrics.csv")

    # --- Compare
    comp_rows, summary = align_and_compare(base_metrics, ft_metrics)

    # Write comparison.csv
    write_csv(
        out_dir / "comparison.csv",
        comp_rows,
        ["id", "business_desc", "baseline_mean_overall", "finetuned_mean_overall", "delta"],
    )

    # Write summary.json
    write_json(out_dir / "summary.json", summary)

    # --- Optional: per-dimension deltas (if both details.jsonl exist)
    dims = None
    base_details_path = base_dir / "details.jsonl"
    ft_details_path = ft_dir / "details.jsonl"
    if base_details_path.exists() and ft_details_path.exists():
        base_details = read_details_jsonl(base_details_path)
        ft_details = read_details_jsonl(ft_details_path)
        dims = dimension_deltas(base_details, ft_details)
        if dims:
            write_json(out_dir / "dimensions.json", dims)

    # --- Report
    write_report_md(out_dir / "report_compare.md", summary, dims)

    print(f"[compare] Wrote:\n  - {out_dir/'comparison.csv'}\n  - {out_dir/'summary.json'}")
    if dims:
        print(f"  - {out_dir/'dimensions.json'}")
    print(f"  - {out_dir/'report_compare.md'}")

if __name__ == "__main__":
    main()
