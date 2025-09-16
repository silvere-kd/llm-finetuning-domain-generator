# src/run_manager.py

"""
RunManager: creates a unique run folder, stamps git/env/seeds, mirrors config,
and (optionally) logs to MLflow & Weights & Biases (W&B).

Usage:
    from run_manager import RunManager

    with RunManager(cfg, run_name_suffix="baseline") as rm:
        rm.log_params({"phase": "baseline"})
        # ... your pipeline work ...
        rm.log_metrics({"mean_overall": 0.8123})
        rm.log_artifact("outputs/baseline/predictions.jsonl")
        rm.log_dir("outputs/baseline_eval_openai", artifact_path="eval")
        rm.finish()  # optional; called automatically on context exit
"""

from __future__ import annotations
import os, sys, json, time, socket, shutil, hashlib, platform, subprocess, pathlib, random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Optional deps
try:
    import mlflow
except Exception:
    mlflow = None

try:
    import wandb
except Exception:
    wandb = None

import torch

# helpers 
def _utc_ts() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())

def _git_info() -> Dict[str, Any]:
    def run(cmd):
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return ""
    return {
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_is_dirty": bool(run(["git", "status", "--porcelain"])),
        "git_remote": run(["git", "config", "--get", "remote.origin.url"]),
    }

def _env_info() -> Dict[str, Any]:
    cuda = torch.version.cuda if hasattr(torch, "version") else None
    return {
        "python": sys.version.replace("\n"," "),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cuda_version": cuda,
        "torch_version": torch.__version__,
    }

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _hash_if_exists(path: Optional[str]) -> Optional[str]:
    if not path or not pathlib.Path(path).exists(): return None
    return _sha256(path)

def _copy_file(src: str, dst: str):
    pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def _dump_json(path: str, obj: Any):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# RunManager 
@dataclass
class RunManager:
    cfg: Dict[str, Any]
    run_name_suffix: str = "run"
    base_dir: str = "outputs/runs"
    run_dir: pathlib.Path = field(init=False)
    run_id: str = field(init=False)
    use_mlflow: bool = field(init=False, default=False)
    use_wandb: bool = field(init=False, default=False)
    _mlflow_active: bool = field(init=False, default=False)
    _wandb_active: bool = field(init=False, default=False)

    def __post_init__(self):
        ts = _utc_ts()
        safe_suffix = self.run_name_suffix.replace(" ", "-")
        self.run_id = f"{ts}_{safe_suffix}"
        self.run_dir = pathlib.Path(self.base_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Persist config snapshot
        cfg_path = self.run_dir / "config.yaml"
        try:
            # if cfg came from yaml.safe_load(...) and you still have the file path in cfg["_cfg_path"], you can copy it.
            import yaml
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.cfg, f, sort_keys=False, allow_unicode=True)
        except Exception:
            _dump_json(str(self.run_dir / "config.json"), self.cfg)

        # Save code / env fingerprints
        _dump_json(str(self.run_dir / "git_info.json"), _git_info())
        _dump_json(str(self.run_dir / "env_info.json"), _env_info())

        # Save requirements freeze if available
        try:
            freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], timeout=30).decode()
            (self.run_dir / "requirements.freeze.txt").write_text(freeze, encoding="utf-8")
        except Exception:
            pass

        # Data hashes (if present in cfg)
        data_hashes = {
            "train_path": _hash_if_exists(self.cfg.get("dataset", {}).get("processed", {}).get("train_path")),
            "val_path": _hash_if_exists(self.cfg.get("dataset", {}).get("processed", {}).get("val_path")),
        }
        _dump_json(str(self.run_dir / "data_hashes.json"), data_hashes)

        # Seeds
        seed = int(self.cfg.get("seed", 42))
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except Exception:
            pass
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        (self.run_dir / "seeds.txt").write_text(str(seed), encoding="utf-8")

        # Tracking backends
        trk = self.cfg.get("tracking", {})
        self.use_mlflow = bool(trk.get("use_mlflow", False)) and (mlflow is not None)
        self.use_wandb = bool(trk.get("use_wandb", False)) and (wandb is not None)

        # MLflow init
        if self.use_mlflow:
            mlflow.set_tracking_uri(trk.get("mlflow_tracking_uri", "file:mlruns"))
            mlflow.set_experiment(trk.get("mlflow_experiment", "domain-llm"))
            self._mlflow_active = True
            mlflow.start_run(run_name=self.run_id)
            mlflow.log_params({
                "run_id": self.run_id,
                "run_name_suffix": self.run_name_suffix,
                **{k: v for k, v in self.cfg.get("train", {}).items()},
                "model_name": self.cfg.get("model_name"),
            })

        # W&B init
        if self.use_wandb:
            wandb.login(key=os.environ.get("WANDB_API_KEY", None)) if os.environ.get("WANDB_API_KEY") else None
            wb = wandb.init(
                project=trk.get("wandb_project", "domain-llm"),
                entity=trk.get("wandb_entity", None),
                name=self.run_id,
                notes=trk.get("notes", ""),
                tags=trk.get("tags", []),
                config=self.cfg,
                reinit=True,
                mode=trk.get("wandb_mode", "online"),  # or "offline"
            )
            self._wandb_active = wb is not None

        # Write a tiny REPRODUCE.md
        cmd = " ".join([shutil.which('python') or "python", *sys.argv])
        (self.run_dir / "REPRODUCE.md").write_text(f"# Reproduce\n\n```\n{cmd}\n```\n", encoding="utf-8")

    # logging API 
    def log_params(self, params: Dict[str, Any]):
        if self._mlflow_active:
            mlflow.log_params(params)
        if self._wandb_active:
            wandb.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if self._mlflow_active:
            mlflow.log_metrics(metrics, step=step)
        if self._wandb_active:
            wandb.log(metrics, step=step)

        # Append to local metrics log
        line = {"step": step, **metrics, "ts": _utc_ts()}
        with open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")

    def log_artifact(self, path: str, artifact_path: Optional[str] = None):
        p = pathlib.Path(path)
        if not p.exists(): return
        # copy into run folder for immutability
        dst = self.run_dir / (artifact_path or "") / p.name
        _copy_file(str(p), str(dst))

        if self._mlflow_active:
            mlflow.log_artifact(str(p), artifact_path=artifact_path)
        if self._wandb_active and p.is_file():
            wandb.save(str(p), base_path=str(p.parent))

    def log_dir(self, directory: str, artifact_path: Optional[str] = None, patterns: Optional[list[str]] = None):
        d = pathlib.Path(directory)
        if not d.exists(): return
        # shallow copy
        dst_dir = self.run_dir / (artifact_path or "") / d.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for item in d.iterdir():
            if item.is_file():
                if patterns and not any(item.name.endswith(ext) for ext in patterns):
                    continue
                _copy_file(str(item), str(dst_dir / item.name))
        # log to backends
        if self._mlflow_active:
            mlflow.log_artifacts(str(d), artifact_path=artifact_path)
        if self._wandb_active:
            # W&B prefers files; directories are okay when using wandb.save with base_path
            for item in d.rglob("*"):
                if item.is_file():
                    wandb.save(str(item), base_path=str(d))

    def finish(self):
        if self._wandb_active:
            wandb.finish()
            self._wandb_active = False
        if self._mlflow_active:
            mlflow.end_run()
            self._mlflow_active = False

    # Context manager
    def __enter__(self) -> "RunManager":
        return self

    def __exit__(self, exc_type, exc, tb):
        # Capture exception state
        if exc:
            self.log_params({"run_status": "failed", "exception": str(exc)})
        else:
            self.log_params({"run_status": "success"})
        self.finish()
