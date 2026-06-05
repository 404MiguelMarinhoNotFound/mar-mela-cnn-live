"""Evaluate a diverse ensemble on the test split with per-source breakdown.

    py scripts/05_ensemble_eval.py --config configs/baseline.yaml
    py scripts/05_ensemble_eval.py --config configs/baseline.yaml \
        --checkpoints checkpoints/tf_efficientnet_b3_seed42_best.pt \
                      checkpoints/tf_efficientnet_b4_seed42_best.pt

Loads several trained checkpoints, runs each **with TTA**, averages their
probabilities (unweighted mean), and tunes the decision threshold on the
*ensemble-averaged validation* probabilities — never a naive 0.5, and never an
average of per-model thresholds (the ensemble's score distribution differs).
Prints overall + per-domain + per-source metrics, flags the best single model
for comparison, and writes ``reports/ensemble_test_metrics.csv``.
"""

from __future__ import annotations

import argparse
import sys
from glob import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import torch first: on Windows it must load before numpy/pandas/cv2 or its
# c10.dll fails to initialize (WinError 1114).
import torch  # noqa: E402
import numpy as np
import pandas as pd

from melanoma.config import Config
from melanoma.data.dataset import MelanomaDataset, split_frame
from melanoma.data.manifest import source_domain
from melanoma.data.transforms import eval_transforms
from melanoma.engine import evaluate
from melanoma.metrics import per_source_report, tune_threshold
from melanoma.models.backbone import build_model
from melanoma.utils import get_device, load_checkpoint


def _infer_one(ckpt_path: str, cfg: Config, device):
    """Load one checkpoint and return (val_probs, val_labels, test_probs,
    test_labels, test_sources) using that model's own backbone/img_size + TTA."""
    meta = torch.load(ckpt_path, map_location="cpu")
    # Each model is built from its OWN stored backbone/img_size (B3=300 vs B4=380),
    # so the input resolution always matches what it was trained at.
    cfg.backbone = meta.get("backbone", cfg.backbone)
    cfg.img_size = meta.get("img_size", cfg.img_size)
    model = build_model(cfg).to(device)
    load_checkpoint(ckpt_path, model, map_location=device)

    val_ds = MelanomaDataset(split_frame(cfg, "val"), eval_transforms(cfg.img_size))
    test_ds = MelanomaDataset(split_frame(cfg, "test"), eval_transforms(cfg.img_size))
    v_probs, v_labels, _ = evaluate(model, val_ds, cfg, device, tta=True)
    t_probs, t_labels, t_sources = evaluate(model, test_ds, cfg, device, tta=True)
    return v_probs, v_labels, t_probs, t_labels, t_sources


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a melanoma ensemble on test")
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--checkpoints", nargs="+", default=None,
                   help="checkpoint paths; default globs checkpoints/*_seed*_best.pt")
    p.add_argument("--data-root", default=None)
    args = p.parse_args()

    cfg = Config.from_yaml(args.config, data_root=args.data_root)
    device = get_device()

    ckpts = args.checkpoints or sorted(
        glob(str(Path(cfg.checkpoint_dir) / "*_seed*_best.pt")))
    if not ckpts:
        raise SystemExit(
            f"No checkpoints found in {cfg.checkpoint_dir} matching *_seed*_best.pt. "
            "Train some first (scripts/02_train.py --seed N) or pass --checkpoints.")
    print(f"Ensembling {len(ckpts)} model(s) on device={device}:")
    for c in ckpts:
        print(f"  - {c}")

    # Accumulate per-model probabilities; keep labels/sources from the first model
    # and assert every other model lines up (split_frame is deterministic and
    # evaluate uses shuffle=False, so index alignment is guaranteed if the splits
    # match — the assert catches a stale manifest or mismatched config).
    val_acc = test_acc = None
    ref_val_labels = ref_test_labels = ref_test_sources = None
    per_model_val_f2: list[tuple[str, float]] = []

    for ckpt in ckpts:
        v_probs, v_labels, t_probs, t_labels, t_sources = _infer_one(ckpt, cfg, device)

        if ref_val_labels is None:
            ref_val_labels, ref_test_labels, ref_test_sources = (
                v_labels, t_labels, t_sources)
        else:
            if not (np.array_equal(v_labels, ref_val_labels)
                    and np.array_equal(t_labels, ref_test_labels)
                    and np.array_equal(t_sources, ref_test_sources)):
                raise SystemExit(
                    f"Label/source ordering for {ckpt} does not match the first "
                    "model. Ensure all models use the same splits/manifest/config.")

        val_acc = v_probs if val_acc is None else val_acc + v_probs
        test_acc = t_probs if test_acc is None else test_acc + t_probs

        # Each model's own val-F2 (TTA), for the single-vs-ensemble comparison.
        _, v_f2 = tune_threshold(v_labels, v_probs, beta=cfg.beta)
        per_model_val_f2.append((Path(ckpt).name, v_f2))

    n = len(ckpts)
    mean_val = val_acc / n
    mean_test = test_acc / n

    # Threshold tuned on the ENSEMBLE-AVERAGED val probabilities.
    threshold, ens_val_f2 = tune_threshold(ref_val_labels, mean_val, beta=cfg.beta)
    beta_key = f"f{int(cfg.beta)}"

    print(f"\nPer-model val {beta_key} (TTA):")
    for name, f2 in per_model_val_f2:
        print(f"  {f2:.4f}  {name}")
    best_name, best_f2 = max(per_model_val_f2, key=lambda x: x[1])
    print(f"  best single: {best_f2:.4f} ({best_name})")
    print(f"  ENSEMBLE val {beta_key}: {ens_val_f2:.4f}  "
          f"({ens_val_f2 - best_f2:+.4f} vs best single)")
    print(f"\nDecision threshold (F{int(cfg.beta)}-optimal on ensemble val): "
          f"{threshold:.3f}")

    domains = [source_domain(s) for s in ref_test_sources]
    report = per_source_report(ref_test_sources, ref_test_labels, mean_test,
                               threshold, beta=cfg.beta, domains=domains)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n=== Ensemble test metrics (val-tuned threshold, TTA) ===")
    print(report.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    out = Path(cfg.reports_dir) / "ensemble_test_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print("\nReminder (doctrine sec.7): judge the model by the per-source/per-domain "
          "table - dermoscopic sources will outscore clinical ones.")


if __name__ == "__main__":
    main()
