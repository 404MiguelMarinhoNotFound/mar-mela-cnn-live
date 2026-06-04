"""Build the image manifest that drives the whole pipeline.

Because the train/val/test splits are pre-made on disk, the manifest is our single
source of truth: one row per image with its split, class label, inferred source,
dimensions, and a corrupt flag. Every later stage (Dataset, audit, evaluation)
reads this CSV instead of re-walking the directory tree.

Source inference is filename-based. The dataset blends several public corpora whose
naming conventions are distinct (verified by inspecting the files):

    ISIC_*               -> isic          (dermoscopic)
    *Dermatoscopic*      -> dermatoscopic (dermoscopic)
    IMD*                 -> ph2           (dermoscopic)
    Dermis *             -> dermis
    *HB*                 -> hb
    *VB*                 -> vb
    SSM_*                -> ssm
    <condition>-*        -> dermnet       (clinical/phone close-ups)
    (anything else)      -> other

Each source also maps to a coarse ``domain`` (dermoscopic vs clinical), which is
what the doctrine's per-source metric breakdown ultimately cares about.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from PIL import Image

# DermNet-style clinical close-ups are named "<condition>-<n>.jpg".
_CLINICAL_CONDITION_RE = re.compile(
    r"^(rosacea|eczema|dermatofibroma|stasis|allergic|seborrheic|seborrhoeic|"
    r"lentigo|malignant-melanoma|psoriasis|melanoma|nevus|basal-cell|squamous|"
    r"actinic|keratosis|wart|tinea|acne|vitiligo)",
    re.IGNORECASE,
)

# source -> coarse domain. "other" is a genuine grab-bag (hashed-ID corpora plus
# DSC*/IMG_* camera shots), so it gets its own 'unknown' domain rather than being
# forced into 'clinical' — keeps the dermoscopic-vs-clinical comparison honest.
_DERMOSCOPIC_SOURCES = {"isic", "dermatoscopic", "ph2", "dermis", "ssm"}
_CLINICAL_SOURCES = {"dermnet", "hb", "vb"}


def infer_source(filename: str) -> str:
    """Return a fine-grained source label for an image filename."""
    name = Path(filename).name
    stem = Path(name).stem
    upper = name.upper()

    if upper.startswith("ISIC_"):
        return "isic"
    if "DERMATOSCOPIC" in upper:
        return "dermatoscopic"
    if upper.startswith("IMD"):
        return "ph2"
    if stem.lower().startswith("dermis"):
        return "dermis"
    if upper.startswith("SSM_"):
        return "ssm"
    if _CLINICAL_CONDITION_RE.match(stem):
        return "dermnet"
    # HB / VB tags appear as suffixes (e.g. 000057HB.jpg) or anywhere in the name.
    if "HB" in upper:
        return "hb"
    if "VB" in upper:
        return "vb"
    return "other"


def source_domain(source: str) -> str:
    """Map a fine source to a coarse domain: dermoscopic / clinical / unknown."""
    if source in _DERMOSCOPIC_SOURCES:
        return "dermoscopic"
    if source in _CLINICAL_SOURCES:
        return "clinical"
    return "unknown"


def _iter_split_classes(cfg) -> list[tuple[str, str, int, Path]]:
    """Yield (split, class_dir, label, path) for every on-disk class folder."""
    out: list[tuple[str, str, int, Path]] = []
    for split in ("train", "val", "test"):
        split_dir = cfg.split_dir(split)
        for class_dir, label in ((cfg.pos_class_dir, 1), (cfg.neg_class_dir, 0)):
            out.append((split, class_dir, label, split_dir / class_dir))
    return out


def build_manifest(cfg, probe_dims: bool = True) -> pd.DataFrame:
    """Walk the split/class folders and build (and persist) the manifest.

    Parameters
    ----------
    cfg : Config
    probe_dims : if True, open each image to record width/height/mode and detect
        unreadable files. Set False for a faster path-only manifest.

    Returns the DataFrame and writes it to ``cfg.manifest_path``.
    """
    rows: list[dict] = []
    for split, class_dir, label, folder in _iter_split_classes(cfg):
        if not folder.exists():
            raise FileNotFoundError(f"Expected data folder missing: {folder}")
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            source = infer_source(path.name)
            row = {
                "path": str(path),
                "filename": path.name,
                "split": split,
                "class_dir": class_dir,
                "label": label,
                "source": source,
                "domain": source_domain(source),
                "format": path.suffix.lower().lstrip("."),
                "width": None,
                "height": None,
                "mode": None,
                "corrupt": False,
            }
            if probe_dims:
                try:
                    with Image.open(path) as im:
                        row["width"], row["height"] = im.size
                        row["mode"] = im.mode
                        im.verify()  # cheap integrity check
                except Exception:  # noqa: BLE001 - any decode error => corrupt
                    row["corrupt"] = True
            rows.append(row)

    df = pd.DataFrame(rows)
    out_path = Path(cfg.manifest_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def load_manifest(cfg) -> pd.DataFrame:
    """Load an existing manifest, building it on demand if absent."""
    path = Path(cfg.manifest_path)
    if not path.exists():
        return build_manifest(cfg)
    return pd.read_csv(path)
