"""Typed configuration for the melanoma baseline.

A single ``Config`` dataclass drives every script. Values come from a YAML file
(``configs/baseline.yaml``) and may be overridden on the command line. Keeping all
knobs here means the data root, image size, and backbone are parameters — never
hardcoded — so the same code runs on a local CPU smoke-test and on a Databricks
GPU cluster.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# Default backbone -> input size, per the doctrine (B3 ~300, B4 ~380).
_BACKBONE_IMG_SIZE = {
    "tf_efficientnet_b0": 224,
    "tf_efficientnet_b3": 300,
    "tf_efficientnet_b4": 380,
}


@dataclass
class Config:
    # ---- Data ----
    data_root: str = "image_splits_MELandBEN"
    # On-disk split directory names -> canonical split label.
    train_dir: str = "Training"
    val_dir: str = "validate"
    test_dir: str = "Testing"
    pos_class_dir: str = "MEL"  # positive (melanoma) folder name
    neg_class_dir: str = "Benign"
    reports_dir: str = "reports"
    manifest_path: str = "reports/manifest.csv"

    # ---- Model ----
    backbone: str = "tf_efficientnet_b3"
    pretrained: bool = True
    img_size: int | None = None  # None -> inferred from backbone

    # ---- Training ----
    epochs: int = 15
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    num_workers: int = 4
    loss: str = "focal"  # "focal" | "weighted_ce"
    focal_gamma: float = 2.0
    focal_alpha: float | None = None  # None -> derive from class balance

    # ---- Eval ----
    tta: bool = False
    beta: float = 2.0  # F-beta; 2 == recall-weighted F2

    # ---- Bookkeeping ----
    seed: int = 42
    subset: int | None = None  # cap images/class for CPU smoke-tests
    checkpoint_dir: str = "checkpoints"
    mlflow: bool = False
    mlflow_experiment: str = "melanoma-baseline"

    # ---- Derived (not from YAML) ----
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.img_size is None:
            self.img_size = _BACKBONE_IMG_SIZE.get(self.backbone, 300)

    # -- IO helpers --
    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> "Config":
        raw: dict[str, Any] = {}
        if path is not None and Path(path).exists():
            with open(path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        raw.update({k: v for k, v in overrides.items() if v is not None})
        valid = {f.name for f in fields(cls)}
        known = {k: v for k, v in raw.items() if k in valid}
        unknown = {k: v for k, v in raw.items() if k not in valid}
        cfg = cls(**known)
        cfg.extra.update(unknown)
        return cfg

    # -- Convenience paths --
    @property
    def reports(self) -> Path:
        return Path(self.reports_dir)

    @property
    def figures_dir(self) -> Path:
        return self.reports / "figures"

    def split_dir(self, split: str) -> Path:
        mapping = {"train": self.train_dir, "val": self.val_dir, "test": self.test_dir}
        return Path(self.data_root) / mapping[split]
