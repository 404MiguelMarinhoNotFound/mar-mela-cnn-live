"""Generate Grad-CAM overlay panels for a sample of test images (doctrine §8).

    py scripts/04_gradcam.py --config configs/baseline.yaml --per-source 3

Samples images stratified by source so you can audit whether the network attends to
the lesion across dermoscopic *and* clinical domains. Writes a panel grid to
``reports/figures/gradcam_panels.png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import torch first: on Windows it must load before numpy/matplotlib/cv2 or its
# c10.dll fails to initialize (WinError 1114).
import torch  # noqa: E402
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from melanoma.config import Config
from melanoma.data.dataset import MelanomaDataset, split_frame
from melanoma.data.transforms import eval_transforms
from melanoma.explain import GradCAM, denormalize, overlay_heatmap
from melanoma.models.backbone import build_model
from melanoma.utils import get_device, load_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description="Grad-CAM panels")
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--per-source", type=int, default=2,
                   help="images sampled per source")
    p.add_argument("--backbone", default=None, help="locate default checkpoint")
    p.add_argument("--data-root", default=None)
    args = p.parse_args()

    cfg = Config.from_yaml(args.config, data_root=args.data_root, backbone=args.backbone)
    device = get_device()
    ckpt_path = args.checkpoint or str(
        Path(cfg.checkpoint_dir) / f"{cfg.backbone}_best.pt")

    # Match the trained backbone/img_size stored in the checkpoint.
    meta = torch.load(ckpt_path, map_location="cpu")
    cfg.backbone = meta.get("backbone", cfg.backbone)
    cfg.img_size = meta.get("img_size", cfg.img_size)
    model = build_model(cfg).to(device)
    load_checkpoint(ckpt_path, model, map_location=device)
    cam_engine = GradCAM(model)

    # Stratified sample by source from the test split.
    test_df = split_frame(cfg, "test")
    sample = (
        test_df.groupby("source", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), args.per_source), random_state=cfg.seed))
        .reset_index(drop=True)
    )
    ds = MelanomaDataset(sample, eval_transforms(cfg.img_size))

    n = len(sample)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for i in range(n):
        image, label, source = ds[i]
        cam = cam_engine(image)
        with torch.no_grad():
            prob = torch.sigmoid(model(image.unsqueeze(0).to(device)).view(-1)).item()

        axes[i, 0].imshow(denormalize(image))
        axes[i, 0].set_title(f"{source} | label={int(label.item())}", fontsize=9)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(overlay_heatmap(image, cam))
        axes[i, 1].set_title(f"Grad-CAM | p(mel)={prob:.2f}", fontsize=9)
        axes[i, 1].axis("off")

    out = cfg.figures_dir / "gradcam_panels.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Wrote {out}  ({n} images)")


if __name__ == "__main__":
    main()
