"""Cross-cutting helpers: seeding, device, checkpoints, and optional MLflow.

Everything here is device-agnostic. ``get_device`` returns CUDA when available
(Databricks GPU cluster) and CPU otherwise (local smoke-test); the training loop
keys AMP off the same result. MLflow logging degrades to a no-op when the package
is absent or disabled, so the code runs identically on a laptop and on Databricks.
"""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(path: str | Path, model, cfg, extra: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "backbone": cfg.backbone,
        "img_size": cfg.img_size,
        "config": cfg.__dict__,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, model, map_location=None):
    ckpt = torch.load(path, map_location=map_location or "cpu")
    model.load_state_dict(ckpt["model_state"])
    return ckpt


# --------------------------------------------------------------------------- #
# Optional MLflow (native on Databricks)
# --------------------------------------------------------------------------- #
class _NullLogger:
    """No-op logger used when MLflow is unavailable or disabled."""

    def log_params(self, *_a, **_k): ...
    def log_metrics(self, *_a, **_k): ...
    def log_artifact(self, *_a, **_k): ...
    def set_tag(self, *_a, **_k): ...
    def end(self): ...


class MLflowLogger:
    def __init__(self, experiment: str):
        import mlflow  # imported lazily

        self._mlflow = mlflow
        mlflow.set_experiment(experiment)
        self._run = mlflow.start_run()

    def log_params(self, params: dict):
        # mlflow rejects non-scalar params; stringify dicts/None.
        clean = {k: ("" if v is None else v) for k, v in params.items()
                 if not isinstance(v, (dict, list))}
        self._mlflow.log_params(clean)

    def log_metrics(self, metrics: dict, step: int | None = None):
        scalar = {k: float(v) for k, v in metrics.items()
                  if isinstance(v, (int, float)) and not _isnan(v)}
        self._mlflow.log_metrics(scalar, step=step)

    def log_artifact(self, path):
        self._mlflow.log_artifact(str(path))

    def set_tag(self, k, v):
        self._mlflow.set_tag(k, v)

    def end(self):
        self._mlflow.end_run()


def _isnan(x) -> bool:
    try:
        return np.isnan(x)
    except TypeError:
        return False


def get_logger(cfg):
    """Return an MLflow logger if enabled+available, else a no-op logger."""
    if not getattr(cfg, "mlflow", False):
        return _NullLogger()
    try:
        return MLflowLogger(cfg.mlflow_experiment)
    except Exception as e:  # noqa: BLE001
        print(f"[mlflow] disabled ({e}); using no-op logger.")
        return _NullLogger()
