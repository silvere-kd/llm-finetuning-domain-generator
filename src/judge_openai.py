# src/judge_openai.py

"""OpenAI GPT-4 judge. Scores suggestions and returns details + aggregate."""
import os, re, json, time, pathlib, csv
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from templates.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_PROMPT_TEMP
from dotenv import load_dotenv

load_dotenv("/workspace/.env")


def _clamp(x: float)->float: x=float(x); return round(0 if x<0 else 1 if x>1 else x,4)

def judge(predictions_path: str, out_dir: str, model_name: str, weights: Dict[str,float]) -> Tuple[str,str]:
    """Read predictions.jsonl → ask judge → write details.jsonl & metrics.csv. Return paths."""
    client = OpenAI()
    items = [json.loads(l) for l in pathlib.Path(predictions_path).read_text(encoding="utf-8").splitlines()]
    outp = pathlib.Path(out_dir); outp.mkdir(parents=True, exist_ok=True)
    details_path = outp/"details.jsonl"
    metrics_path = outp/"metrics.csv"

    details_rows=[]
    metrics_rows=[]
    for rec in items:
        biz=rec.get("business_desc","")
        suggs=[str(x).strip().lower() for x in rec.get("suggestions",[]) if isinstance(x,(str,int,float))]
        if not suggs:
            details_rows.append({"id":rec.get("id"),"business_desc":biz,"details":[]})
            metrics_rows.append({"id":rec.get("id"),"business_desc":biz[:200],"mean_overall":0.0})
            continue
        user = JUDGE_USER_PROMPT_TEMP.format(business=biz, suggestions=json.dumps(suggs,ensure_ascii=False),
                                weights=json.dumps(weights,ensure_ascii=False))
        # Simple retry
        for attempt in range(1,4):
            try:
                resp = client.chat.completions.create(
                    model=model_name, temperature=0.0, response_format={"type":"json_object"},
                    messages=[{"role":"system","content":JUDGE_SYSTEM_PROMPT},{"role":"user","content":user}]
                )
                content = (resp.choices[0].message.content or "").strip()
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    m = re.findall(r"\[.*?\]", content, flags=re.S)
                    data = json.loads(m[-1]) if m else []
                rows = data["results"] if isinstance(data,dict) and "results" in data else (data if isinstance(data,list) else [])
                det=[]
                for d in rows:
                    if not isinstance(d,dict): continue
                    dom=str(d.get("domain","")).lower().strip()
                    if not dom: continue
                    item={"domain":dom,
                          "relevance":_clamp(d.get("relevance",0.0)),
                          "memorability":_clamp(d.get("memorability",0.0)),
                          "readability":_clamp(d.get("readability",0.0)),
                          "safety":_clamp(d.get("safety",0.0))}
                    item["overall"]=_clamp(d.get("overall", weights["relevance"]*item["relevance"]
                                                             +weights["memorability"]*item["memorability"]
                                                             +weights["readability"]*item["readability"]
                                                             +weights["safety"]*item["safety"]))
                    det.append(item)
                details_rows.append({"id":rec.get("id"),"business_desc":biz,"details":det})
                mean = round(sum(x["overall"] for x in det)/len(det),4) if det else 0.0
                metrics_rows.append({"id":rec.get("id"),"business_desc":biz[:200],"mean_overall":mean})
                break
            except Exception:
                if attempt==3:
                    details_rows.append({"id":rec.get("id"),"business_desc":biz,"details":[]})
                    metrics_rows.append({"id":rec.get("id"),"business_desc":biz[:200],"mean_overall":0.0})
                time.sleep(1.2*attempt)

    with details_path.open("w",encoding="utf-8") as f:
        for r in details_rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    with metrics_path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["id","business_desc","mean_overall"]); w.writeheader()
        for r in metrics_rows: w.writerow(r)

    return str(details_path), str(metrics_path)
