# CLAUDE.md — mar-mela-cnn-live

Project context and commands for Claude Code sessions.

## Project summary

Binary melanoma vs. benign classifier. ~15,225 mixed-source images (ISIC
dermoscopic + clinical/phone). Optimizes **F2 (recall-weighted)**. Single
EfficientNet backbone (transfer learning), focal loss, Albumentations augmentation
stack, per-source metric breakdown, Grad-CAM explainability.
Full design rationale: `melanoma_cnn_doctrine_v2.md`.

## Layout

```
melanoma/           importable package
  config.py         dataclass Config + YAML load/merge
  data/
    manifest.py     infer_source(), build_manifest() -> manifest CSV
    audit.py        EDA: class balance, source composition, leakage check
    dataset.py      MelanomaDataset (manifest-driven, torch Dataset)
    transforms.py   Albumentations stacks: train / eval / TTA
  models/
    backbone.py     timm EfficientNet, ImageNet weights, single-logit head
  losses.py         BinaryFocalLoss(gamma=2) + weighted-CE fallback
  metrics.py        F2, tune_threshold(), per_source_report()
  engine.py         train_one_epoch / evaluate (AMP on CUDA, device-agnostic)
  explain.py        Grad-CAM (hook-based, no retraining)
  utils.py          set_seed, get_device, checkpoint I/O, MLflow logger

scripts/
  01_audit_data.py  -> reports/manifest.csv + reports/figures/
  02_train.py       train; checkpoints best val-F2 model
  03_evaluate.py    test eval, val-tuned threshold, per-source table
  04_gradcam.py     Grad-CAM overlay panels per source

configs/baseline.yaml   all knobs (backbone, lr, focal_gamma, subset, …)
```

## Common commands

```bash
# Install deps (CPU local). On Databricks: install via cluster libs.
py -m pip install -r requirements.txt

# Step 0: build manifest + audit (always run first after new data)
py scripts/01_audit_data.py --data-root image_splits_MELandBEN
py scripts/01_audit_data.py --rebuild   # force regenerate

# CPU smoke-test (subset=40 per class, 1 epoch, B0 backbone)
py scripts/02_train.py --config configs/baseline.yaml \
  --backbone tf_efficientnet_b0 --subset 40 --epochs 1 \
  --batch-size 8 --num-workers 0

# Full train (run on Databricks GPU; data_root -> /Volumes/workspace/mar-mela/images)
py scripts/02_train.py --config configs/baseline.yaml

# Evaluate on test split (per-source/per-domain F2 table)
py scripts/03_evaluate.py --config configs/baseline.yaml

# Grad-CAM panels (2 images per source, stratified)
py scripts/04_gradcam.py --config configs/baseline.yaml
```

## Key design decisions

- **Metric:** F2 (β=2), not accuracy. Decision threshold tuned on val, never 0.5.
- **Loss:** BinaryFocalLoss (γ=2, α derived from train class balance).
- **Source domains:** `dermoscopic` (ISIC, PH2, dermis, SSM) / `clinical` (DermNet, HB, VB) / `unknown` (hashed-ID corpus + camera shots). Always report per-domain.
- **Import order on Windows:** `torch` must import before `numpy`/`pandas`/`cv2` (WinError 1114 DLL conflict). All scripts enforce this.
- **Device:** `get_device()` returns CUDA if available, else CPU. AMP and pin_memory are gated on CUDA — same code runs locally and on Databricks.
- **Manifest:** every stage reads `reports/manifest.csv` (built by `01_audit_data.py`). Never re-walk the directory tree from scripts.

## Honest performance expectations (doctrine §9)

| Setting | Expected AUC / F2 |
|---|---|
| In-distribution dermoscopy (ISIC) | AUC 0.92–0.96 |
| HAM10000-style, single source | F1 ~0.87 |
| **Mixed-source, externally validated** | **materially lower — judge per-source** |
| Clinical / phone-only subset | AUC ~0.72 |

## Environment notes

- Python 3.11.2 (`py` launcher on Windows).
- `torch 2.12+cpu` locally; Databricks cluster provides CUDA build.
- `numpy` must stay `<2` (torch + pandas ABI); `opencv-python-headless<4.12` for numpy-1.x compat.
- `mlflow` optional: set `mlflow: true` in config when on Databricks; no-op locally.

## Git / repo

- Remote: `https://github.com/404MiguelMarinhoNotFound/mar-mela-cnn-live.git`
- **Never commit:** `image_splits_MELandBEN/`, `reports/`, `checkpoints/`, `.claude/` (all in `.gitignore`).

---

## TODO

### Databricks — data already uploaded

The training images are live at:

```
Unity Catalog volume: workspace.mar-mela.images
DBFS path:            /Volumes/workspace/mar-mela/images
```

Structure on the volume mirrors the local folder:

```
/Volumes/workspace/mar-mela/images/
  Training/  {Benign, MEL}   # 12,166 images
  Testing/   {Benign, MEL}   #  1,520 images
  validate/  {Benign, MEL}   #  1,539 images
```

- [ ] **Create Databricks notebook** (`notebooks/train_baseline.py` or `.ipynb`) that
  sets `data_root = "/Volumes/workspace/mar-mela/images"` and calls `02_train.py`
  (or imports `melanoma` directly). Attach to a GPU cluster (e.g. g4dn / A10).
- [ ] **Run full training** on Databricks GPU — B3 or B4 backbone, 15 epochs, batch 32.
- [ ] **Rebuild manifest on cluster** (`01_audit_data.py --data-root /Volumes/workspace/mar-mela/images`)
  before training — the local `reports/manifest.csv` has local Windows paths.
- [ ] **Evaluate** with `03_evaluate.py`; compare per-domain F2 vs. doctrine §9 ladder.
- [ ] **Log to MLflow** — set `mlflow: true` in config; Databricks MLflow is natively available.
- [ ] **Optional next v1 improvements:** TTA (`tta: true`), try B4 backbone, small 3-model ensemble.
- [ ] **v2 roadmap:** multi-class diagnosis target, metadata fusion, U-Net ROI, SHAP.
