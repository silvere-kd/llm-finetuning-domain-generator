# src/data_synth.py

import os, re, json, time, random, pathlib
from typing import List, Dict, Optional
from dotenv import load_dotenv
from cfg import load_config
from templates.constants import INDUSTRIES, STYLES, TLDS, UNSAFE_THEMES, JSON_ARRAY_REGEX
from templates.prompts import GEN_PROMPT_TEMP

load_dotenv("/workspace/.env")


# Utilities
def _slugify(label: str) -> str:
    label = re.sub(r"\s+", "-", label.lower())
    label = re.sub(r"[^a-z0-9-]", "", label)
    label = re.sub(r"-{2,}", "-", label).strip("-")
    return label[:12]

def _clean_domain(d: str) -> str:
    d = d.strip().lower()
    d = re.sub(r"[^a-z0-9\.-]", "", d)
    d = re.sub(r"-{2,}", "-", d)
    return d.strip(".-")

def _dedup_keep_order(arr: List[str]) -> List[str]:
    seen, out = set(), []
    for x in arr:
        if not x: continue
        if x not in seen:
            seen.add(x); out.append(x)
    return out

# Rule-based generator
def synth_targets_rule(desc: str, n: int) -> List[str]:
    parts = re.findall(r"[a-z0-9]+", desc.lower())
    core = _slugify((parts[0] if parts else "brand") + "-" + (parts[-1] if parts else "shop"))
    palette = {
        "premium":   [core, core+"prime", core+"elite", "haus"+core],
        "playful":   [core+"ly", "go"+core, core+"buddy", core+"fun"],
        "minimalist":[core, core[:8], core.replace("-", "")],
        "techy":     [core+"tech", "get"+core, "try"+core, core+"hub"],
        "eco":       ["green"+core, "eco"+core, core+"earth"],
        "luxury":    [core+"lux", core+"atelier", core+"maison"],
    }
    style = random.choice(STYLES)
    outs = []
    for root in palette[style]:
        root = re.sub(r"-{2,}", "-", root).strip("-")
        outs.append(root + random.choice(TLDS))
    outs = [_clean_domain(x) for x in outs]
    return _dedup_keep_order(outs)[:n]

# LLM-based generator
_openai_client = None
def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY is not set but LLM backend requested.")
        _openai_client = OpenAI(api_key=key)
    return _openai_client

def _llm_prompt(desc: str, n: int) -> str:    
    return GEN_PROMPT_TEMP.format(desc=desc, n=n)

def _parse_llm_domains(text: str) -> List[str]:
    """
    Accept either:
      - A raw JSON array (["a.com", ...])
      - An object like {"domains": ["a.com", ...]}
    """
    try:
        data = json.loads(text)
    except Exception:
        # Fallback: Use regex to find JSON array
        match = re.findall(JSON_ARRAY_REGEX, text, flags=re.S)
        if not match:
            return []
        try:
            data = json.loads(match[-1])
        except Exception:
            return []
    if isinstance(data, list):
        arr = data
    elif isinstance(data, dict):
        arr = data.get("domains", [])
    else:
        arr = []
    out = []
    for x in arr:
        if isinstance(x, (str, int, float)):
            out.append(_clean_domain(str(x)))
    return _dedup_keep_order(out)

def synth_targets_llm(desc: str, n: int, model: str, temperature: float, top_p: float,
                      max_retries: int, sleep_sec: float) -> List[str]:
    client = _get_openai_client()
    prompt = _llm_prompt(desc, n)
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=200,
                response_format={"type": "json_object"},  # encourages strict JSON
                messages=[{"role": "user", "content": prompt}],
            )
            content = (resp.choices[0].message.content or "").strip()
            domains = _parse_llm_domains(content)
            return domains[:n]
        except Exception as e:
            if attempt >= max_retries:
                # Too many attempts
                return []
            time.sleep(sleep_sec * attempt)
    return []

# Synthesis
def _make_safe_record(desc: str, targets: List[str]) -> Dict:
    targets = [_clean_domain(x) for x in targets][:5]
    return {"business_desc": desc, "targets": targets, "safety": "safe"}

def _make_unsafe_records(n: int) -> List[Dict]:
    rows = []
    for i in range(n):
        theme = random.choice(UNSAFE_THEMES)
        rows.append({
            "business_desc": f"{theme} website",
            "targets": [],
            "safety": "unsafe",
        })
    return rows

def main():
    cfg = load_config()
    s = cfg["dataset"]["raw"]
    random.seed(cfg.get("seed", 42))

    out_path = pathlib.Path(s["path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []

    # Mix generation method if hybrid
    def choose_backend() -> str:
        if s["backend"] == "hybrid":
            return "llm" if random.random() < float(s["hybrid_ratio"]) else "rule"
        return s["backend"]

    N = int(s.get("N", 3200))   # number of safe rows to generate
    for _ in range(N):
        ind = random.choice(INDUSTRIES)
        style = random.choice(STYLES)
        geo = random.choice(["in downtown area", "for freelancers", "for families", "subscription-based"])
        desc = f"{ind} {geo} ({style} vibe)"

        backend = choose_backend()
        if backend == "llm":
            targets = synth_targets_llm(
                desc, s["n_per_desc"], 
                s["llm_model"], 
                s["temperature"], 
                s["top_p"], 
                s["max_retries"], 
                s["sleep_sec"]
            )
            # fallback to rule if LLM failed
            if not targets:
                targets = synth_targets_rule(desc, s["n_per_desc"])
        else:
            targets = synth_targets_rule(desc, s["n_per_desc"])

        rows.append(_make_safe_record(desc, targets))

    # Add unsafe negatives
    unsafe_ratio = float(s.get("unsafe_multiplier", 0.1))
    n_unsafe = max(1, int(N * unsafe_ratio)) if unsafe_ratio > 0 else 0
    rows.extend(_make_unsafe_records(n_unsafe))

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[data_synth] backend={s['backend']} | rows={len(rows)} -> {out_path}")

if __name__ == "__main__":
    main()
