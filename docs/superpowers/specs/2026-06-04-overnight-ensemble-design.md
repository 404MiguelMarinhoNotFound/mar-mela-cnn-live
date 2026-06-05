# Overnight Diverse Ensemble — Design

**Date:** 2026-06-04
**Goal:** Maximize melanoma-vs-benign **F2** (per doctrine, judged per-domain) using
free overnight GPU compute on an RTX 2070 Super (8 GB), via a diverse model ensemble.

## Context

- Data (from `reports/manifest.csv`): train 12,166 (neg 6,359 / pos 5,807, ~52/48),
  val 1,539, test 1,520. 8 sources across 3 domains (dermoscopic / clinical / unknown).
- Augmentation stack (`transforms.py`) already mirrors the SIIM-ISIC 2020 winning recipe,
  and a 4-view TTA bank exists. Gains tonight therefore come from **model diversity**,
  not more augmentation.
- Roadmap (CLAUDE.md) already lists "small 3-model ensemble" and "try B4 backbone" as the
  intended v1 improvements — this design executes exactly that.

## Models (5 total)

| # | Backbone          | img | batch | seed | epochs |
|---|-------------------|-----|-------|------|--------|
| 1 | tf_efficientnet_b3 | 300 | 32    | 42   | 30     |
| 2 | tf_efficientnet_b3 | 300 | 32    | 43   | 30     |
| 3 | tf_efficientnet_b3 | 300 | 32    | 44   | 30     |
| 4 | tf_efficientnet_b4 | 380 | 16    | 42   | 30     |
| 5 | tf_efficientnet_b4 | 380 | 16    | 43   | 30     |

Shared hyperparameters (current `baseline.yaml` defaults, unchanged):
focal loss γ=2 (α auto from class balance), AdamW lr 2e-4, weight_decay 1e-4,
cosine schedule + 1-epoch warmup, AMP on CUDA. Each model checkpoints its
**best-by-val-F2** epoch; threshold tuned on val every epoch (never 0.5).

Rough budget on a 2070 Super at 30 epochs: B3 ~45 min/model, B4 ~135 min/model →
~6.5–7 h training + eval. Comfortable overnight; headroom to add seeds later.

## Code changes

### 1. `scripts/02_train.py` — per-model checkpoints
- Add `--seed` CLI flag overriding `cfg.seed`.
- Checkpoint filename becomes `{backbone}_seed{seed}_best.pt` (was `{backbone}_best.pt`)
  so the 5 runs do not overwrite each other.
- No change to training logic, metrics, or checkpoint contents (still stores
  `threshold`, `val_metrics`, `epoch`, `backbone`, `img_size`).

### 2. `scripts/05_ensemble_eval.py` (new) — ensemble evaluation
- Accept multiple checkpoints: `--checkpoints a.pt b.pt ...` (or default-glob
  `checkpoints/*_seed*_best.pt`).
- For each model: build from its own stored `backbone`/`img_size`, run **with TTA**
  on both val and test splits, collect probabilities.
- **Ensemble probability = unweighted mean** across the 5 models (per image).
- **Decision threshold:** re-tune on the *ensemble-averaged val probabilities* for
  F2 (not an average of per-model thresholds — the ensemble's distribution differs).
- Report overall + per-domain + per-source F1/F2/recall/precision/AUC/balanced-acc
  via the existing `per_source_report`, written to `reports/ensemble_test_metrics.csv`.
- Also print each single model's val-F2 and the **best single model** for comparison,
  so we can see whether the ensemble actually beats the best member.

### 3. `scripts/run_ensemble.ps1` (new) — overnight runner
- Trains the 5 models sequentially via `py scripts/02_train.py ... --seed N`.
- Continues to the next model if one fails (logs the failure); tees each run's
  stdout to `reports/train_<backbone>_seed<seed>.log`.
- After training, runs `scripts/05_ensemble_eval.py` over all produced checkpoints.
- Designed to be launched once and left unattended.

## Design decisions / rationale

- **Unweighted mean of probabilities.** Simple, robust, and standard; weighting by
  val-F2 is a possible later refinement but adds tuning risk for marginal gain (YAGNI).
- **Re-tune threshold on ensemble val probs.** The averaged score distribution is not
  the same as any single model's, so its F2-optimal cut differs — re-tuning is correct.
- **TTA on at ensemble eval.** Near-free recall bump; the bank already exists.
- **B3×3 + B4×2 mix** rather than 5× one backbone: architecture + seed diversity
  decorrelates errors more than seed diversity alone, which is what makes ensembling pay.

## Out of scope (explicitly not doing tonight)

- Hyperparameter sweep (approach C) — diversity ensembling buys more per unit effort.
- Weighted/stacked ensembling, k-fold CV, new augmentations, model-architecture changes.
- Databricks (per user instruction, local GPU only).

## Success criteria

- All 5 checkpoints produced without manual intervention.
- `reports/ensemble_test_metrics.csv` written with a per-domain table.
- Ensemble F2 compared against the best single model; dermoscopic domain expected
  strongest (doctrine §9), clinical/phone weakest — judged per-domain, not on the
  headline number.
