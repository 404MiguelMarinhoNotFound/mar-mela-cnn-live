# mar-mela-cnn-live

Melanoma vs. benign image classifier — a single-model EfficientNet baseline that
optimizes **F2 (recall-weighted)**, built per [`melanoma_cnn_doctrine_v2.md`](melanoma_cnn_doctrine_v2.md).

The dataset is **mixed-source** (ISIC dermoscopy + clinical/phone images), which
the doctrine identifies as the hardest problem. Accordingly the pipeline audits
the data by source, stratifies metrics **per source**, tunes the decision
threshold for F2, and ships Grad-CAM so you can confirm the network looks at the
lesion — not hair, ink, or rulers.

> Single-model baseline (v1). Ensemble, metadata fusion, and U-Net ROI are out of
> scope but the layout leaves room for them.

## Data layout (not committed — see `.gitignore`)

```
image_splits_MELandBEN/
  Training/  {Benign, MEL}   # 12,166 images
  Testing/   {Benign, MEL}   #  1,520 images
  validate/  {Benign, MEL}   #  1,539 images
```

## Project layout

```
melanoma/            # importable package
  config.py          # dataclass Config + YAML load
  data/              # manifest, audit/EDA, dataset, augmentation transforms
  models/backbone.py # timm EfficientNet, ImageNet weights, binary head
  losses.py          # FocalLoss (gamma=2) + class-weighted CE
  metrics.py         # F2, per-source report, F2 threshold tuning
  engine.py          # train/eval loops (device-agnostic, AMP on CUDA)
  explain.py         # Grad-CAM
  utils.py           # seed, device, checkpoints, optional MLflow
scripts/             # 01_audit_data  02_train  03_evaluate  04_gradcam
configs/baseline.yaml
```

## Quickstart

```bash
py -m pip install -r requirements.txt

# Step 0 — audit data, build the manifest that drives everything downstream
py scripts/01_audit_data.py --data-root image_splits_MELandBEN

# Smoke-test the whole loop on CPU (tiny subset)
py scripts/02_train.py --config configs/baseline.yaml --subset 200 --epochs 1

# Real training happens on a Databricks GPU cluster (device auto-detected)
py scripts/02_train.py --config configs/baseline.yaml

# Evaluate on the test split with the val-tuned threshold; per-source breakdown
py scripts/03_evaluate.py --config configs/baseline.yaml

# Grad-CAM panels
py scripts/04_gradcam.py --config configs/baseline.yaml
```

## Honest expectations

Per the doctrine's generalization reality-check: dermoscopic sources (ISIC, PH2)
will score far higher than clinical/phone images. Judge the model by the
**per-source F2 table**, not a single headline number.
