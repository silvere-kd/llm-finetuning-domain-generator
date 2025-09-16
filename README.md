## LLM-finetuning for domain namaes suggestions


### **How to run**

- Create venv + install libs

```
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

- Run baseline pipeline

```
uv run src/pipeline_baseline.py
```

- Run QLoRA finetuning and compare its performances with baseline

```
uv run src/pipeline_finetune.py --engine hf
```


### **How to interprete results**

- **Win rate > 50%** and **avg delta > 0** → finetuning is helpful.

- If wins are concentrated on specific prompt types (check ``comparison.csv``), target those domains with more data to amplify gains.

- If **readability delta** is negative in ``dimensions.json``, SFT data might be too permissive (tweak rules/augmentations).