"""Step 0: build the image manifest and run the data audit.

    py scripts/01_audit_data.py --data-root image_splits_MELandBEN

Outputs:
    reports/manifest.csv        one row per image (drives all later stages)
    reports/figures/*.png       class balance, source composition, resolution
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script (no install needed).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from melanoma.config import Config
from melanoma.data import audit, manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Build manifest + audit data")
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--data-root", default=None, help="override data_root")
    p.add_argument("--no-dims", action="store_true",
                   help="skip per-image dimension probing (faster)")
    p.add_argument("--rebuild", action="store_true",
                   help="rebuild manifest even if it exists")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config, data_root=args.data_root)
    print(f"data_root   : {cfg.data_root}")
    print(f"manifest    : {cfg.manifest_path}")

    manifest_path = Path(cfg.manifest_path)
    if args.rebuild or not manifest_path.exists():
        print("Building manifest (probing image dimensions)..."
              if not args.no_dims else "Building manifest (paths only)...")
        df = manifest.build_manifest(cfg, probe_dims=not args.no_dims)
        print(f"Wrote {len(df)} rows -> {manifest_path}")
    else:
        df = manifest.load_manifest(cfg)
        print(f"Loaded existing manifest: {len(df)} rows "
              f"(use --rebuild to regenerate)")

    audit.run_audit(cfg, df)


if __name__ == "__main__":
    main()
