"""Train the single-model EfficientNet baseline.

    py scripts/02_train.py --config configs/baseline.yaml            # full run (GPU)
    py scripts/02_train.py --config configs/baseline.yaml --subset 200 --epochs 1

Checkpoints the best model by **validation F2** to
``<checkpoint_dir>/<backbone>_best.pt`` and records the val-tuned threshold inside it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from melanoma.config import Config
from melanoma.data.dataset import MelanomaDataset, class_counts, split_frame
from melanoma.data.transforms import eval_transforms, train_transforms
from melanoma.engine import evaluate, make_scheduler, train_one_epoch
from melanoma.losses import build_loss
from melanoma.metrics import compute_metrics, tune_threshold
from melanoma.models.backbone import build_model
from melanoma.utils import get_device, get_logger, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train melanoma baseline")
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--data-root", default=None)
    p.add_argument("--backbone", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--subset", type=int, default=None,
                   help="cap images per class (CPU smoke-tests)")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--mlflow", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    overrides = dict(
        data_root=args.data_root, backbone=args.backbone, epochs=args.epochs,
        batch_size=args.batch_size, subset=args.subset, num_workers=args.num_workers,
        mlflow=True if args.mlflow else None,
    )
    cfg = Config.from_yaml(args.config, **overrides)
    set_seed(cfg.seed)
    device = get_device()
    print(f"device={device} backbone={cfg.backbone} img_size={cfg.img_size} "
          f"epochs={cfg.epochs} batch={cfg.batch_size} subset={cfg.subset}")

    # --- data ---
    train_df = split_frame(cfg, "train")
    val_df = split_frame(cfg, "val")
    n_neg, n_pos = class_counts(train_df)
    print(f"train: {len(train_df)} (neg={n_neg} pos={n_pos}) | val: {len(val_df)}")

    train_ds = MelanomaDataset(train_df, train_transforms(cfg.img_size))
    val_ds = MelanomaDataset(val_df, eval_transforms(cfg.img_size))
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=device.type == "cuda", drop_last=True,
    )

    # --- model / loss / optim ---
    model = build_model(cfg).to(device)
    criterion = build_loss(cfg, n_neg, n_pos)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = make_scheduler(optimizer, cfg, steps_per_epoch=max(1, len(train_loader)))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    logger = get_logger(cfg)
    logger.log_params({k: v for k, v in cfg.__dict__.items() if k != "extra"})

    ckpt_path = Path(cfg.checkpoint_dir) / f"{cfg.backbone}_best.pt"
    beta_key = f"f{int(cfg.beta)}"
    best_f2 = -1.0

    for epoch in range(cfg.epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer,
                               scheduler, device, scaler)
        probs, labels, _ = evaluate(model, val_ds, cfg, device, tta=False)
        thr, _ = tune_threshold(labels, probs, beta=cfg.beta)
        m = compute_metrics(labels, probs, threshold=thr, beta=cfg.beta)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch+1:02d}/{cfg.epochs} | loss {loss:.4f} | lr {lr_now:.2e} "
              f"| val {beta_key} {m[beta_key]:.4f} recall {m['recall']:.4f} "
              f"AUC {m['auc']:.4f} thr {thr:.2f}")
        logger.log_metrics({"train_loss": loss, "lr": lr_now,
                            f"val_{beta_key}": m[beta_key], "val_recall": m["recall"],
                            "val_auc": m["auc"], "val_threshold": thr}, step=epoch)

        if m[beta_key] > best_f2:
            best_f2 = m[beta_key]
            save_checkpoint(ckpt_path, model, cfg,
                            extra={"threshold": thr, "val_metrics": m, "epoch": epoch})
            print(f"  -> new best val {beta_key}={best_f2:.4f}; saved {ckpt_path}")

    logger.log_metrics({f"best_val_{beta_key}": best_f2})
    logger.log_artifact(ckpt_path)
    logger.end()
    print(f"\nDone. Best val {beta_key}={best_f2:.4f}. Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
