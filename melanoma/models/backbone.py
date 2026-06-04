"""Backbone factory.

Transfer-learning recipe per the doctrine: take an ImageNet-pretrained EfficientNet
(via ``timm``), replace the classifier with a single-logit binary head, fine-tune.
Returns the model plus the name of the final conv stage so Grad-CAM can hook it
without hardcoding layer paths.
"""

from __future__ import annotations

import timm
import torch.nn as nn


def build_model(cfg) -> nn.Module:
    """Create an EfficientNet (or any timm model) with a 1-logit head."""
    model = timm.create_model(
        cfg.backbone,
        pretrained=cfg.pretrained,
        num_classes=1,  # single logit -> BCE/focal
    )
    return model


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Best-effort last-conv layer for Grad-CAM across timm EfficientNet variants.

    EfficientNets expose ``conv_head`` (the final 1x1 expansion) which is the
    conventional Grad-CAM target. Fall back to the last ``blocks`` stage, then to
    the last Conv2d found.
    """
    if hasattr(model, "conv_head"):
        return model.conv_head
    if hasattr(model, "blocks"):
        return model.blocks[-1]
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM target.")
    return last_conv
