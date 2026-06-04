"""Evaluate a trained checkpoint on the test split with per-source breakdown.

    py scripts/03_evaluate.py --config configs/baseline.yaml

Uses the threshold tuned on validation (stored in the checkpoint, or re-tuned on
the val split if missing) — never a naive 0.5. Prints overall + per-domain +
per-source F1/F2/recall/precision/AUC/balanced-acc and writes them to
``reports/test_metrics.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import torch first: on Windows it must load before numpy/pandas/cv2 or its
# c10.dll fails to initialize (WinError 1114).
import torch  # noqa: E402
import pandas as pd

from melanoma.config import Config
from melanoma.data.dataset import MelanomaDataset, split_frame
from melanoma.data.manifest import source_domain
from melanoma.data.transforms import eval_transforms
from melanoma.engine import evaluate
from melanoma.metrics import per_source_report, tune_threshold
from melanoma.models.backbone import build_model
from melanoma.utils import get_device, load_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate melanoma baseline on test")
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--backbone", default=None, help="locate default checkpoint")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--data-root", default=None)
    args = p.parse_args()

    cfg = Config.from_yaml(args.config, data_root=args.data_root, backbone=args.backbone)
    device = get_device()
    ckpt_path = args.checkpoint or str(
        Path(cfg.checkpoint_dir) / f"{cfg.backbone}_best.pt")

    # Build the model from the checkpoint's own backbone/img_size so it always
    # matches what was trained, regardless of the config default.
    meta = torch.load(ckpt_path, map_location="cpu")
    cfg.backbone = meta.get("backbone", cfg.backbone)
    cfg.img_size = meta.get("img_size", cfg.img_size)
    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(ckpt_path, model, map_location=device)
    print(f"Loaded {ckpt_path} (epoch {ckpt.get('epoch', '?')})")

    # Threshold: prefer the val-tuned value stored at train time.
    threshold = ckpt.get("threshold")
    if threshold is None:
        print("No stored threshold; re-tuning on validation split...")
        val_ds = MelanomaDataset(split_frame(cfg, "val"), eval_transforms(cfg.img_size))
        v_probs, v_labels, _ = evaluate(model, val_ds, cfg, device, tta=args.tta)
        threshold, _ = tune_threshold(v_labels, v_probs, beta=cfg.beta)
    print(f"Decision threshold (F{int(cfg.beta)}-optimal on val): {threshold:.3f}")

    # Test inference.
    test_ds = MelanomaDataset(split_frame(cfg, "test"), eval_transforms(cfg.img_size))
    probs, labels, sources = evaluate(model, test_ds, cfg, device, tta=args.tta)
    domains = [source_domain(s) for s in sources]

    report = per_source_report(sources, labels, probs, threshold,
                               beta=cfg.beta, domains=domains)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n=== Test metrics (val-tuned threshold) ===")
    print(report.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    out = Path(cfg.reports_dir) / "test_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print("\nReminder (doctrine sec.7): judge the model by the per-source/per-domain "
          "table - dermoscopic sources will outscore clinical ones.")


if __name__ == "__main__":
    main()
