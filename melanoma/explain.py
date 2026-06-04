"""Grad-CAM explainability (doctrine §8).

A dependency-free Grad-CAM: forward/backward hooks on the backbone's final conv
stage produce a class-discriminative heatmap, letting clinicians confirm the network
attends to the lesion rather than hair, ink, or rulers. No retraining required.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .models.backbone import gradcam_target_layer

_MEAN = np.array([0.485, 0.456, 0.406])
_STD = np.array([0.229, 0.224, 0.225])


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model.eval()
        self.target = target_layer or gradcam_target_layer(model)
        self._activations = None
        self._gradients = None
        self.target.register_forward_hook(self._save_activation)
        self.target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, out):
        self._activations = out.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def __call__(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Return a [H,W] heatmap in [0,1] for a single CHW input tensor."""
        device = next(self.model.parameters()).device
        x = image_tensor.unsqueeze(0).to(device)
        logit = self.model(x).view(-1)
        self.model.zero_grad(set_to_none=True)
        logit.backward()

        grads = self._gradients          # [1, C, h, w]
        acts = self._activations         # [1, C, h, w]
        weights = grads.mean(dim=(2, 3), keepdim=True)  # GAP over spatial dims
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))  # [1,1,h,w]
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    """CHW normalized tensor -> HWC uint8 RGB image for overlaying."""
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * _STD + _MEAN)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def overlay_heatmap(image_tensor: torch.Tensor, cam: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    """Blend a Grad-CAM heatmap over the (denormalized) image. Returns RGB uint8."""
    base = denormalize(image_tensor)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (alpha * heat + (1 - alpha) * base).astype(np.uint8)
