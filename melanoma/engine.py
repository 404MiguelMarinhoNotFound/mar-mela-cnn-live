"""Training and evaluation loops — device-agnostic, AMP on CUDA only.

Implements the doctrine's training setup: AdamW, cosine-annealing LR with a 1-epoch
warm-up, mixed precision when a GPU is present. ``evaluate`` returns raw
probabilities, labels, and sources so threshold tuning and the per-source report can
run downstream without re-inferring.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.transforms import tta_transforms


def make_scheduler(optimizer, cfg, steps_per_epoch: int):
    """Linear warm-up for ``warmup_epochs`` then cosine decay to ~0."""
    warmup_steps = max(1, cfg.warmup_epochs * steps_per_epoch)
    total_steps = max(warmup_steps + 1, cfg.epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # doctrine: warm-up LR = 0.1x initial, ramping to 1x
            return 0.1 + 0.9 * (step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device,
                    scaler=None) -> float:
    model.train()
    use_amp = scaler is not None and device.type == "cuda"
    running, n = 0.0, 0
    for images, labels, _src in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(images).view(-1)
            loss = criterion(logits, labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()
        bs = images.size(0)
        running += loss.item() * bs
        n += bs
    return running / max(1, n)


@torch.no_grad()
def evaluate(model, dataset, cfg, device, tta: bool = False, batch_size=None):
    """Run inference over a dataset; return (probs, labels, sources).

    With ``tta=True`` the dataset's own transform is ignored and several TTA views
    are averaged per image.
    """
    model.eval()
    bs = batch_size or cfg.batch_size

    if not tta:
        loader = DataLoader(dataset, batch_size=bs, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=device.type == "cuda")
        probs, labels, sources = [], [], []
        for images, lbls, src in loader:
            images = images.to(device, non_blocking=True)
            p = torch.sigmoid(model(images).view(-1))
            probs.append(p.cpu().numpy())
            labels.append(lbls.numpy())
            sources.extend(src)
        return np.concatenate(probs), np.concatenate(labels), np.array(sources)

    # TTA: average probabilities across views. Swap in each view transform.
    from copy import copy

    views = tta_transforms(cfg.img_size)
    acc_probs = None
    labels_out, sources_out = None, None
    for view in views:
        ds = copy(dataset)
        ds.transform = view
        loader = DataLoader(ds, batch_size=bs, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=device.type == "cuda")
        probs, labels, sources = [], [], []
        for images, lbls, src in loader:
            images = images.to(device, non_blocking=True)
            p = torch.sigmoid(model(images).view(-1))
            probs.append(p.cpu().numpy())
            labels.append(lbls.numpy())
            sources.extend(src)
        probs = np.concatenate(probs)
        acc_probs = probs if acc_probs is None else acc_probs + probs
        labels_out = np.concatenate(labels)
        sources_out = np.array(sources)
    return acc_probs / len(views), labels_out, sources_out
