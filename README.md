## LLM-finetuning for domain namaes suggestions


### How to run

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
uv run src/pipeline_finetune.py
```