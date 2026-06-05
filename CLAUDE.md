# CLAUDE.md — mar-mela-cnn-live

Project context and commands for Claude Code sessions.

## Project summary

Binary melanoma vs. benign classifier. ~15,225 mixed-source images (ISIC
dermoscopic + clinical/phone). Optimizes **F2 (recall-weighted)**. Diverse
EfficientNet ensemble (3× B3 + 2× B4, varied seeds), focal loss, Albumentations
augmentation stack, TTA, per-source metric breakdown, Grad-CAM explainability.
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
  01_audit_data.py    -> reports/manifest.csv + reports/figures/
  02_train.py         train one model; --seed flag for ensemble diversity;
                      checkpoints best val-F2 to {backbone}_seed{seed}_best.pt
  03_evaluate.py      single-model test eval, val-tuned threshold, per-source table
  04_gradcam.py       Grad-CAM overlay panels per source
  05_ensemble_eval.py ensemble eval: average TTA probs across checkpoints,
                      re-tune threshold on ensemble val probs, per-source table
  run_ensemble.ps1    overnight runner: trains all 5 models then runs ensemble eval

configs/baseline.yaml   all knobs (backbone, lr, focal_gamma, subset, …)
docs/superpowers/specs/ design specs
```

## Common commands

```bash
# Install deps (CPU local). On Databricks: install via cluster libs.
py -m pip install -r requirements.txt

# IMPORTANT — local GPU setup (Windows, RTX 2070 Super / CUDA 12.3):
# torch must be the CUDA build, not +cpu. Reinstall if needed:
py -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
py -m pip install "numpy<2" "typing-extensions>=4.14.1" "opencv-python-headless<4.12"

# Step 0: build manifest + audit (always run first after new data)
py scripts/01_audit_data.py --data-root image_splits_MELandBEN
py scripts/01_audit_data.py --rebuild   # force regenerate

# CPU smoke-test (subset=40 per class, 1 epoch, B0 backbone)
py scripts/02_train.py --config configs/baseline.yaml \
  --backbone tf_efficientnet_b0 --subset 40 --epochs 1 \
  --batch-size 8 --num-workers 0 --seed 42

# Train one model (GPU)
py scripts/02_train.py --config configs/baseline.yaml --seed 42
# -> checkpoint: checkpoints/tf_efficientnet_b3_seed42_best.pt

# Run the full overnight diverse ensemble (5 models, 30 epochs each, ~7 h)
powershell -ExecutionPolicy Bypass -File scripts\run_ensemble.ps1
# -> checkpoints/*_seed*_best.pt  |  reports/train_*.log

# Ensemble evaluation (TTA + averaged probs + per-domain report)
py scripts/05_ensemble_eval.py --config configs/baseline.yaml
# -> reports/ensemble_test_metrics.csv

# Single-model evaluate on test split
py scripts/03_evaluate.py --config configs/baseline.yaml

# Grad-CAM panels (2 images per source, stratified)
py scripts/04_gradcam.py --config configs/baseline.yaml
```

## Key design decisions

- **Metric:** F2 (β=2), not accuracy. Decision threshold tuned on val, never 0.5.
- **Loss:** BinaryFocalLoss (γ=2, α derived from train class balance).
- **Ensemble:** 5 models (3× B3 @ 300px + 2× B4 @ 380px, seeds 42–44/42–43). Unweighted
  probability averaging; threshold re-tuned on *ensemble-averaged* val probs.
- **TTA:** 4-view bank (identity, H-flip, V-flip, transpose) averaged at inference.
- **Source domains:** `dermoscopic` (ISIC, PH2, dermis, SSM) / `clinical` (DermNet, HB, VB) / `unknown` (hashed-ID corpus + camera shots). Always report per-domain.
- **Import order on Windows:** `torch` must import before `numpy`/`pandas`/`cv2` (WinError 1114 DLL conflict). All scripts enforce this.
- **Device:** `get_device()` returns CUDA if available, else CPU. AMP and pin_memory are gated on CUDA — same code runs locally and on Databricks.
- **Manifest:** every stage reads `reports/manifest.csv` (built by `01_audit_data.py`). Never re-walk the directory tree from scripts.

## Achieved performance (v1 ensemble, local RTX 2070 Super)

5-model ensemble (3× B3 + 2× B4, 30 epochs, TTA), threshold=0.29 tuned on ensemble val probs.

| Domain / Source | F2 | AUC | Recall |
|---|---|---|---|
| **Overall** | **0.905** | **0.956** | 0.974 |
| Dermoscopic | 0.904 | 0.951 | 0.979 |
| Clinical | 0.866 | 0.968 | 0.935 |
| Unknown | 0.941 | 0.973 | 0.978 |
| ISIC (source) | 0.899 | 0.953 | 0.980 |
| DermNet (source) | 0.958 | 1.000 | 1.000 |
| HB (source) | 0.812 | 0.933 | 0.929 |

Per-model best val F2 (TTA): B4 seed42 **0.9203** · B3 seed44 0.9186 · B4 seed43 0.9179
· B3 seed42 0.9160 · B3 seed43 0.9140. Ensemble val F2: **0.9224** (+0.0021 vs best single).

## Honest performance expectations (doctrine §9)

| Setting | Expected AUC / F2 |
|---|---|
| In-distribution dermoscopy (ISIC) | AUC 0.92–0.96 |
| HAM10000-style, single source | F1 ~0.87 |
| **Mixed-source, externally validated** | **materially lower — judge per-source** |
| Clinical / phone-only subset | AUC ~0.72 |

## Environment notes

- Python 3.11.2 (`py` launcher on Windows).
- **Local GPU:** `torch 2.5.1+cu121` (CUDA 12.1 build, compatible with driver CUDA 12.3).
  RTX 2070 Super (8 GB). B3 batch 32 fits; B4 needs batch 16.
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
- [ ] **Run ensemble training** on Databricks GPU — use `run_ensemble.ps1` as a template
  (adapt to bash/notebook); rebuild manifest first (local `reports/manifest.csv` has Windows paths).
- [ ] **Log to MLflow** — set `mlflow: true` in config; Databricks MLflow is natively available.
- [ ] **v2 roadmap:** multi-class diagnosis target, metadata fusion, U-Net ROI, SHAP.
