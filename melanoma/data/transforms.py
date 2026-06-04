"""Albumentations augmentation stacks.

The train stack mirrors the SIIM-ISIC 2020 winning recipe (doctrine §3/§6):
geometric flips/rotations, brightness/contrast jitter, a blur family, noise, the
distortion family, **CLAHE** (normalizes the wildly different exposure of phone vs
dermoscopic images), hue/sat shifts, and **Cutout** (CoarseDropout). Val/test only
resize + normalize. A light TTA stack provides flip/transpose variants averaged at
inference.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet normalization (backbones are ImageNet-pretrained).
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def train_transforms(img_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # ShiftScaleRotate is deprecated in albumentations 2.x; Affine is the
        # supported equivalent.
        A.Affine(translate_percent=(-0.06, 0.06), scale=(0.9, 1.1),
                 rotate=(-30, 30), p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
        ], p=0.3),
        # albumentations 2.x: std_range is a fraction of the 0-255 range.
        A.GaussNoise(std_range=(0.04, 0.2), p=0.3),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3),
            A.ElasticTransform(alpha=1, sigma=50),
        ], p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.4),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                             val_shift_limit=10, p=0.4),
        # albumentations 2.x CoarseDropout API (== Cutout). Fractional hole sizes.
        A.CoarseDropout(num_holes_range=(1, 1),
                        hole_height_range=(0.1, 0.15),
                        hole_width_range=(0.1, 0.15), p=0.4),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


def eval_transforms(img_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


def tta_transforms(img_size: int) -> list[A.Compose]:
    """A small bank of test-time-augmentation views (incl. the identity view)."""
    base = [A.Resize(img_size, img_size)]
    tail = [A.Normalize(mean=_MEAN, std=_STD), ToTensorV2()]
    variants = [
        [],                                   # identity
        [A.HorizontalFlip(p=1.0)],
        [A.VerticalFlip(p=1.0)],
        [A.Transpose(p=1.0)],
    ]
    return [A.Compose(base + v + tail) for v in variants]
