"""Metrics, F2 threshold tuning, and the per-source breakdown.

The doctrine is emphatic: never report accuracy alone, always optimize F2 (recall
weighted 2x), tune the decision threshold on validation (never assume 0.5), and
break every metric down **per source / per domain** because mixed-source data hides
large gaps behind a single headline number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_prob, threshold: float = 0.5, beta: float = 2.0) -> dict:
    """Full metric dict at a given threshold. AUC uses probabilities directly."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # specificity = recall of the negative class
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # AUC undefined when only one class present (e.g. a tiny source slice).
    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")

    return {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": fbeta_score(y_true, y_pred, beta=1.0, zero_division=0),
        f"f{int(beta)}": fbeta_score(y_true, y_pred, beta=beta, zero_division=0),
        "auc": auc,
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "threshold": threshold,
    }


def tune_threshold(y_true, y_prob, beta: float = 2.0,
                   grid: np.ndarray | None = None) -> tuple[float, float]:
    """Pick the threshold that maximizes F-beta on (validation) data.

    Returns (best_threshold, best_fbeta).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    best_t, best_score = 0.5, -1.0
    for t in grid:
        score = fbeta_score(y_true, (y_prob >= t).astype(int), beta=beta, zero_division=0)
        if score > best_score:
            best_t, best_score = float(t), float(score)
    return best_t, best_score


def per_source_report(sources, y_true, y_prob, threshold: float, beta: float = 2.0,
                      domains=None) -> pd.DataFrame:
    """Metrics broken down by source (and by domain if provided), plus 'overall'.

    Sorted with the coarse domain rows first, then individual sources by size.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    sources = np.asarray(sources)

    rows: list[dict] = []

    def add(name: str, kind: str, mask: np.ndarray) -> None:
        if mask.sum() == 0:
            return
        m = compute_metrics(y_true[mask], y_prob[mask], threshold, beta)
        m = {"group": name, "kind": kind, **m}
        rows.append(m)

    add("overall", "overall", np.ones(len(y_true), dtype=bool))
    if domains is not None:
        domains = np.asarray(domains)
        for d in sorted(pd.unique(domains)):
            add(d, "domain", domains == d)
    for s in sorted(pd.unique(sources)):
        add(s, "source", sources == s)

    df = pd.DataFrame(rows)
    # nice column order
    cols = ["group", "kind", "n", "n_pos", "recall", "precision", "specificity",
            "f1", f"f{int(beta)}", "auc", "balanced_acc", "threshold"]
    return df[[c for c in cols if c in df.columns]]
