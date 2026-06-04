"""Manifest-driven dataset.

``MelanomaDataset`` reads rows from the audit manifest (never the directory tree),
so the split, label, and source for every image are fixed and reproducible. It
returns the transformed image, a float label, and the source string — the source is
carried through to enable the doctrine's per-source metric breakdown at eval time.
"""

from __future__ import annotations

from pathlib import Path

# Import torch BEFORE cv2: on Windows, OpenCV's bundled DLLs can shadow torch's
# (c10.dll init fails with WinError 1114) if cv2 loads first.
import torch
from torch.utils.data import Dataset

import cv2
import numpy as np
import pandas as pd

from .manifest import load_manifest

# cv2 reads faster than PIL and plays nicely with Albumentations (numpy HWC).
cv2.setNumThreads(0)  # avoid oversubscription inside DataLoader workers


class MelanomaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def _read_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR or None
        if img is None:  # fallback for formats cv2 may miss (some bmp/png)
            from PIL import Image
            img = np.array(Image.open(path).convert("RGB"))
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = self._read_image(row["path"])
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return img, label, row["source"]


def split_frame(cfg, split: str) -> pd.DataFrame:
    """Return the manifest rows for one split, optionally subsampled per class.

    ``cfg.subset`` caps the number of images *per class* — used to keep CPU
    smoke-tests fast while preserving class balance.
    """
    df = load_manifest(cfg)
    df = df[(df["split"] == split) & (~df["corrupt"])].copy()
    if cfg.subset:
        df = (
            df.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), cfg.subset), random_state=cfg.seed))
            .reset_index(drop=True)
        )
    return df


def class_counts(df: pd.DataFrame) -> tuple[int, int]:
    """Return (n_negative, n_positive) for class-weight / alpha derivation."""
    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    return n_neg, n_pos
