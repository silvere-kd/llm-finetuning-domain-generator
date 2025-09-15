# src/data_prep.py

import json, pathlib, random, re
from typing import Dict, List
from cfg import load_config
from templates.prompts import SFT_PROMPT_TEMP


# Cleaning domain
def _clean_domain(d: str) -> str:
    d = d.strip().lower()
    d = re.sub(r"[^a-z0-9\.-]", "", d)
    d = re.sub(r"-{2,}", "-", d)
    return d.strip(".-")

def _format_record(r: Dict) -> Dict:
    prompt = SFT_PROMPT_TEMP.format(desc=r["business_desc"])
    if r["safety"] == "unsafe":
        response = "[]"
    else:
        tgts = []
        seen = set()
        for x in r.get("targets", []) or []:
            x = _clean_domain(str(x))
            if x and x not in seen:
                seen.add(x); tgts.append(x)
        response = json.dumps(tgts[:5], ensure_ascii=False)
    return {"prompt": prompt, "response": response, "safety": r["safety"]}

# Stratified split
def _stratified_split(safe_rows: List[Dict], unsafe_rows: List[Dict], train_ratio: float, seed: int):
    random.Random(seed).shuffle(safe_rows)
    random.Random(seed + 1).shuffle(unsafe_rows)

    n_safe = len(safe_rows)
    n_unsafe = len(unsafe_rows)
    n_safe_tr = int(round(n_safe * train_ratio))
    n_unsafe_tr = int(round(n_unsafe * train_ratio))

    train = safe_rows[:n_safe_tr] + unsafe_rows[:n_unsafe_tr]
    val   = safe_rows[n_safe_tr:] + unsafe_rows[n_unsafe_tr:]

    # Stable shuffle within each split for better mixing
    random.Random(seed + 2).shuffle(train)
    random.Random(seed + 3).shuffle(val)

    return train, val, (n_safe, n_unsafe, n_safe_tr, n_unsafe_tr)

def main():
    cfg = load_config()
    seed = int(cfg["seed"])
    train_ratio = float(cfg["dataset"]["processed"]["train_ratio"])

    raw_path = pathlib.Path(cfg["dataset"]["raw"]["path"])
    rows = [json.loads(l) for l in raw_path.read_text(encoding="utf-8").splitlines()]

    # Stratify by safety
    safe_rows = [r for r in rows if r.get("safety") == "safe"]
    unsafe_rows = [r for r in rows if r.get("safety") == "unsafe"]

    train_raw, val_raw, stats = _stratified_split(safe_rows, unsafe_rows, train_ratio, seed)
    n_safe, n_unsafe, n_safe_tr, n_unsafe_tr = stats

    # Format to SFT schema
    train_fmt = [_format_record(r) for r in train_raw]
    val_fmt   = [_format_record(r) for r in val_raw]

    # Write
    out_tr = pathlib.Path(cfg["dataset"]["processed"]["train_path"])
    out_va = pathlib.Path(cfg["dataset"]["processed"]["val_path"])
    out_tr.parent.mkdir(parents=True, exist_ok=True)

    with out_tr.open("w", encoding="utf-8") as f:
        for r in train_fmt:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with out_va.open("w", encoding="utf-8") as f:
        for r in val_fmt:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Ratios report
    def _ratio(a, b): return round(a / b, 4) if b else 0.0
    tr_safe = sum(1 for r in train_raw if r["safety"] == "safe")
    tr_unsafe = len(train_raw) - tr_safe
    va_safe = sum(1 for r in val_raw if r["safety"] == "safe")
    va_unsafe = len(val_raw) - va_safe

    print(
        "[data_prep] Stratified split completed\n"
        f"  Total: {len(rows)}  | Safe: {n_safe}  | Unsafe: {n_unsafe}  "
        f"(unsafe ratio = {_ratio(n_unsafe, len(rows))})\n"
        f"  Train: {len(train_raw)} (safe={tr_safe}, unsafe={tr_unsafe}, "
        f"unsafe ratio={_ratio(tr_unsafe, len(train_raw))})\n"
        f"  Val:   {len(val_raw)} (safe={va_safe}, unsafe={va_unsafe}, "
        f"unsafe ratio={_ratio(va_unsafe, len(val_raw))})\n"
        f"  Paths -> {out_tr} | {out_va}"
    )

if __name__ == "__main__":
    main()
