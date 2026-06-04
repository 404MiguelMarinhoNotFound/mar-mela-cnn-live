# Melanoma Detection CNN — Architecture Doctrine (v2, Expanded)

**Scope:** Binary classifier (melanoma vs. benign), ~15k mixed-source images (dermoscopic ISIC + clinical + phone), roughly balanced, transfer-learning backbone, optimizing **F1 / F2 (recall-weighted)**.

**What changed in v2:** Grounded in *large-scale* datasets (ISIC 2019 ≈ 25,331 images; SIIM-ISIC 2020 ≈ 33k; HAM10000 = 10,015; combined challenge sets up to 58,457), added the full architecture internals of the backbones and the actual challenge-winning pipeline, and added a hard reality-check section on generalization failure from *Lancet Digital Health*, *Annals of Oncology*, and *Nature Medicine*–adjacent stress-test studies. A note on bias appears at the end of each major section.

> **On the "small data, biased results" concern:** You were right. The 95–98% headline numbers in v1 mostly came from small, single-dataset, dermoscopic-only studies. v2 leans on the large public challenges and on external-validation studies that expose what happens when those models meet real-world images. The numbers are lower and more honest.

---

## 0. Guiding Principles

1. **Recall is the north star; F2 keeps it honest.** A missed melanoma costs far more than a false alarm. Optimize F2 (recall weighted 2× over precision), not accuracy.
2. **The mixed-source domain is your hardest problem, not the architecture.** The largest external-validation study to date found that **source institution itself influenced classification errors**, and clinically unusual lesions cratered accuracy (pigmented nevi 83% correct vs. non-pigmented 35%). ([Lancet Digital Health, 2022](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(22)00021-8/fulltext))
3. **Transfer learning is the consensus default.** The standard, challenge-winning recipe is: take an ImageNet-pretrained CNN, replace the final layer, fine-tune. ([SIIM-ISIC 2020 winning solution, Ha et al.](https://arxiv.org/abs/2010.05351))
4. **Ensembling + heavy augmentation, not a single exotic model, wins on large data.** The 2020 winner was an ensemble of **18 models**; diversity of backbone and input size was the explicit key. ([Ha et al.](https://arxiv.org/abs/2010.05351))
5. **Explainability is part of the model.** Grad-CAM/SHAP let you confirm the network looks at the lesion, not at ink, hair, or rulers — known failure modes (below).

---

## 1. The Large-Scale Datasets (ground truth for benchmarking)

| Dataset | Size | Classes | Melanoma ratio | Notes |
|---|---|---|---|---|
| **HAM10000** | 10,015 | 7 | ~11% MEL; **>66% is `nv`** | Severe imbalance; the de-facto teaching benchmark. ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2772528621000340)) |
| **ISIC 2019** | 25,331 | 8 (+1 unknown at test) | minority | Test set contains an **unknown class not in training** — a deliberate OOD trap. ([Gessert et al.](https://arxiv.org/pdf/1910.03910)) |
| **SIIM-ISIC 2020** | ~33,000 | binary (9-class diagnosis available) | **1.76% malignant** | Extreme imbalance; AUC is unstable as a result. ([Ha et al.](https://ar5iv.labs.arxiv.org/html/2010.05351)) |
| **Combined 2018+2019+2020** | up to 58,457 | — | 2019 subset has 17.85% positives | Combining years stabilizes the metric. ([Ha et al.](https://ar5iv.labs.arxiv.org/html/2010.05351)) |

**Doctrine for your 15k mixed set:** treat HAM10000 / ISIC 2019 as your *pretraining-adjacent* and *external-validation* corpora. Your set is closer in size to HAM10000 than to the full challenge data, so the HAM10000 results below are your most honest yardstick.

**Bias note:** All of these are dermatologically curated and dermoscopy-heavy. Your phone/clinical images are *under-represented* in every public benchmark — this is precisely the gap Section 7 addresses.

---

## 2. Backbone Architectures — Internals

### 2.1 EfficientNet (primary recommendation)
**Core idea — compound scaling.** Rather than scaling depth, width, or resolution independently, EfficientNet scales all three together with a fixed compound coefficient (Tan & Le, 2019). The building block is the **MBConv** (mobile inverted bottleneck) with depthwise-separable convolutions and built-in **squeeze-and-excitation (SE)** channel attention. This is why it extracts richer fine-grained features per parameter than ResNet.

**Evidence on skin lesions:**
- On HAM10000 with ImageNet transfer learning, **EfficientNet-B4 was the best of B0–B7**, reaching **87% F1 and 87.91% top-1 accuracy** on the imbalanced 7-class task. ([Multiclass Skin Cancer Classification using EfficientNets, ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2772528621000340))
- EfficientNet-B0 beat ResNet-50 on HAM10000 (macro/micro AUC 0.93/0.97) by extracting richer fine-grained features. ([Review, The SAI](https://thesai.org/Downloads/Volume12No10/Paper_60-Skin_Lesions_Classification_and_Segmentation.pdf))
- EfficientNetB3 was the chosen backbone for a SIIM-ISIC 2019/2020 study validated against **170 dermatologists**. ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12192895/))

**Your default:** EfficientNet-B3 or B4. B4 is the sweet spot in the literature for ~10–25k dermoscopic images.

### 2.2 ResNet (residual learning baseline)
**Core idea — residual skip connections.** Each block learns a residual `F(x)` added to its input `x` (`H(x) = F(x) + x`), so gradients flow through identity shortcuts and very deep nets train without vanishing gradients. Reliable, well-understood, a fair baseline — but parameter-hungry and prone to overfit on small medical sets.

**Evidence:** ResNet-50 reached 79.95% (ISIC 2016) → 81.57% (2017) → 89.28% (2018) accuracy across challenge years; consistently a notch below EfficientNet and DenseNet in head-to-heads (EfficientNet-B7 88.41% vs. ResNet-50 83.42%). ([IIETA](https://www.iieta.org/download/file/fid/138535); [Comparative analysis](https://www.researchgate.net/publication/383542189))

### 2.3 DenseNet (feature-reuse baseline)
**Core idea — dense connectivity.** Every layer receives the feature maps of *all* preceding layers (concatenation, not addition). This maximizes feature reuse and is parameter-efficient: a 250-layer DenseNet has only ~15.3M parameters. Strong on dermoscopy but heavy to train.

**Evidence:** DenseNet-201 hit 81.27% (ISIC 2016) and 88.70% (ISIC 2018) — but **collapsed to 73.44% on ISIC 2017**, the canonical warning that single-dataset performance does not transfer. ([IIETA](https://www.iieta.org/download/file/fid/138535))

### 2.4 SE-ResNeXt / ResNeSt (ensemble diversifiers)
The 2020 winner deliberately added **SE-ResNeXt-101** (grouped convolutions + SE attention) and **ResNeSt-101** (split-attention) alongside EfficientNets purely for *model diversity* in the ensemble. ([Ha et al.](https://ar5iv.labs.arxiv.org/html/2010.05351))

### 2.5 ConvNeXt (modern strong baseline)
A 38-architecture benchmark across 10 families on HAM10000 (frozen pretrained weights) found **ConvNeXtXLarge achieved the best raw performance**, with EfficientNet best on performance-per-parameter. ([MDPI Electronics 2025](https://www.mdpi.com/2079-9292/14/14/2785))

**Bias note:** Most internal comparisons fix training conditions but still report single-dataset test accuracy. Trust the *relative* ranking (EfficientNet ≈ ConvNeXt > DenseNet > ResNet for this domain) more than the absolute numbers.

---

## 3. The Reference Pipeline — SIIM-ISIC 2020 Winning Solution (study in depth)

This is the single most instructive large-data artifact for your project. Full internals, from Ha, Liu & Liu (2020):

**Validation strategy (the real lesson).** With only **1.76% positives in 33k images**, 5-fold CV AUC was wildly unstable. Their fix: **combine 2018+2019+2020 data** (2019 alone has 17.85% positives) and CV on the combined set (`cv_all`). Metric stability ranked `cv_all > cv_2020 > private_LB > public_LB` (std devs 0.0012 / 0.0043 / 0.0060 / 0.0093). They **ignored the public leaderboard entirely** — its correlation with CV was ~0 for single models.

**Target choice.** Instead of the binary benign/malignant label, they trained on the **9-class diagnosis** (NV, MEL, BCC, BKL, AK, SCC, VASC, DF, Unknown) and took the MEL-class probability at inference. The richer target gave the model more signal. One model used a 4-class target for diversity.

**Metadata fusion architecture.** Some models concatenated **14 patient/image metadata features** (sex, approx age, 10 one-hot anatomical-site features, image byte-size, and `n_images` per patient) with CNN features. Metadata passed through **two FC layers (512→128, or 128→32)** before concatenation with the CNN embedding, then a final FC layer. Image-only models scored higher individually, but metadata models added ensemble diversity.

**Augmentation stack (Albumentations).** Transpose, Flip, Rotate, RandomBrightness, RandomContrast, MotionBlur, MedianBlur, GaussianBlur, GaussNoise, OpticalDistortion, GridDistortion, ElasticTransform, **CLAHE**, HueSaturationValue, ShiftScaleRotate, and **Cutout**.

**Training setup.** Cosine-annealing LR with **1 warm-up epoch** (warm-up LR = 0.1× initial), **15 epochs** typical, initial LR 1e-4 to 3e-4, **batch size 64**, mixed precision on V100s.

**Ensemble.** 18 models spanning EfficientNet **B3–B7**, SE-ResNeXt-101, ResNeSt-101; input sizes **384 / 448 / 512 / 576 / 640 / 768 / 896**; combined by **averaging probability *ranks*** (each model's outputs mapped to uniform [0,1] before averaging).

**Result:** **0.9600 AUC cross-validation, 0.9490 AUC private leaderboard** (single models ~0.92–0.94; the ensemble added ~0.01–0.02 AUC).

**What you steal from this for 15k mixed images:**
- Build a **stable, combined-source validation set** before tuning anything.
- Consider a **multi-class diagnosis target** even though your deliverable is binary.
- **Fuse metadata** if you have any (site, age, sex).
- Use the **CLAHE + Cutout + geometric/blur** augmentation stack as your starting point.
- A **small ensemble (3–5 models, varied input size)** is a cheap, reliable +AUC.

---

## 4. Class Imbalance & Loss Function

### Doctrine
Even with a "roughly balanced" 15k set, do not default to plain cross-entropy — real inference skews melanoma-minority, and you optimize F2.

### Evidence (large-data)
- **Focal loss** down-weights easy negatives to emphasize the hard minority class; validated against CE and weighted-CE on ISIC 2018 in a CNN→ViT hybrid. ([Diagnostics/MDPI 2022](https://www.mdpi.com/2075-4418/13/1/72))
- **SMOTE** minority oversampling + median-filter preprocessing improved benign-nevus vs. malignant-melanoma classification across ResNet50 / EfficientNet-B0 / Inception-V3 / Inception-ResNet-V2 on HAM10000+ISIC2019. ([ResearchGate / GRA framework](https://www.researchgate.net/figure/Class-distribution-in-the-HAM10000-and-ISIC-2019-datasets_tbl2_359064577))
- The 2020 winner sidestepped imbalance partly via the **multi-class target trick** and combined-year data rather than loss reweighting alone. ([Ha et al.](https://arxiv.org/abs/2010.05351))

### Knobs
- Loss: **Focal loss** (tune γ ≈ 2) or class-weighted CE fallback.
- **SMOTE / minority oversampling** at the data layer.
- **Threshold tuning** on validation to maximize F2 — never assume 0.5.

---

## 5. Preprocessing & Segmentation

### Doctrine
Standardize aggressively across your three sources; consider a U-Net ROI stage.

### Evidence
- A two-step **U-Net (trained on HAM10000) → classifier** pipeline segmented SIIM-ISIC 2020 lesions before classification, the U-Net reaching **96.03% validation accuracy** for masking. ([Stanford CS230 / mcilwain](https://tom-mcilwain.github.io/melanoma.html))
- Standard first step across 34 reviewed studies: standardize **size, resolution, and color balance**. ([Systematic review, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11727731/))
- **CLAHE** (contrast-limited adaptive histogram equalization) appears in the winning augmentation stack — useful for normalizing the wildly different exposure of phone vs. dermoscopic images. ([Ha et al.](https://ar5iv.labs.arxiv.org/html/2010.05351))

**Bias note:** Segmentation models are themselves trained on dermoscopy; a U-Net trained only on ISIC may segment phone images poorly. Validate the mask quality on your phone subset before trusting it.

---

## 6. Data Augmentation

### Doctrine
Mandatory — closes the domain gap *and* regularizes. Use TTA at inference.

### Evidence
- Augmenting HAM10000 from 10,015→45,756 and ISIC2019 from 25,331→174,132 images let a fine-tuned 5-layer CNN reach **97.88% / 98.67% accuracy** (single-dataset — see caveat). ([Data Augmentation in HAM10000 and ISIC 2019](https://www.academia.edu/107025321/))
- **Test-Time Augmentation** lifted balanced accuracy to **97.58% on ISIC 2019**, matching far heavier ViT approaches. ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11766406/))
- Augmentation is "essential" for EfficientNet generalization across **real-world imaging conditions that vary significantly** — your exact case. ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12192895/))

---

## 7. The Generalization Reality Check (the part most papers hide)

This section exists because of your "results are biased" instinct — which the largest validation studies confirm.

- **Source institution biases predictions.** In the ISIC 2019 grand-challenge validation, the originating institution influenced classification errors, and **47.1% of out-of-distribution images were misclassified as malignant** by the top 25 algorithms — a flood of unnecessary biopsies if deployed. Experts beat algorithms badly on unseen categories (**26% vs 6% correct**). ([Lancet Digital Health, 2022](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(22)00021-8/fulltext))
- **Real-world artifacts break CNNs.** Surgical ink markings reduce melanoma specificity; changes in zoom, brightness, contrast, and a simple vertical flip alter the predictions of a dermatologist-level CNN. Stress-testing across 7 datasets revealed gaps in discrimination, calibration, and robustness. ([Stress testing, PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7820258/))
- **Clinical (non-dermoscopic) images are genuinely harder.** A *de novo* CNN on clinical close-up melanoma images reached only **AUC 0.72 vs. dermatologists' 0.81** — and was outperformed by the doctors. Your phone/clinical subset will behave like this, not like the 0.95 dermoscopy numbers. ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8476836/))
- **But CNNs genuinely can match/beat experts on in-distribution dermoscopy.** Inception-v4 outperformed most of **58 dermatologists**; the foundational Inception-v3 work fine-tuned on **129,450 images** reached dermatologist-level classification. The capability is real — *within distribution.* ([Annals of Oncology 2018](https://www.annalsofoncology.org/article/S0923-7534(19)34105-5/fulltext); [Nature/Esteva, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8382232/))
- **Human + CNN beats either alone.** Fusion reached **89% sensitivity** vs. dermatologists 66% / CNN 86.1%; a market-approved CNN + dermatologists hit **100% sensitivity, 86.4% accuracy**. Frame your tool as decision *support*, not replacement. ([Multi-dataset ViT evaluation, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12607522/))

**Doctrine:** Your honest target on a *mixed, externally-validated* set is well below the 0.95+ dermoscopy ceiling. Budget for it, stratify your test metrics by image source, and report per-source F2.

---

## 8. Explainability (XAI)

### Doctrine
Ship **Grad-CAM heatmaps with every prediction**; prefer/compare SHAP for faithfulness; use t-SNE for latent-space sanity checks.

### Evidence
- **Grad-CAM** produces class-discriminative maps without retraining, letting clinicians verify predictions rest on lesion features, not spurious correlations. ([arXiv 2026](https://arxiv.org/html/2601.00964v1))
- A faithfulness comparison of LIME, SHAP, attention maps, and Grad-CAM on an Inception-ResNetV2 skin-lesion CNN found **SHAP most faithful**. ([Frontiers in Physiology 2026](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2026.1717517/full))
- A HAM10000 multi-stage model with Grad-CAM reached **94.8% accuracy, 91.9% macro-F1, 0.957 AUC** while staying interpretable — a realistic, well-validated bar. ([ResearchGate](https://www.researchgate.net/figure/EfficientNet-B4-model-performance-on-HAM10000-dataset-a-multiclass-classification-AK_fig6_372139502))

---

## 9. Validation Protocol

### Doctrine
- **Stratified split** by class *and by image source*. Common literature split: 72/8/20 (HAM10000) or 60/10/30. ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2772528621000340); [Diagnostics/MDPI 2024](https://www.mdpi.com/2075-4418/14/19/2242))
- **Combined-source CV** for metric stability — the single biggest lesson from the 2020 winner.
- **External validation is mandatory** — hold out an entire source (e.g., all phone images, or all of ISIC 2019) and report on it separately.
- **Report:** F1, F2, recall/sensitivity, precision, AUC, balanced accuracy, **and per-source breakdowns**. Never accuracy alone.

### Honest benchmark ladder for your project
| Setting | Realistic AUC/F1 you should expect |
|---|---|
| In-distribution dermoscopy (ISIC-like) | AUC 0.92–0.96, F1 ~0.87 (EfficientNet-B4) |
| HAM10000 7-class, interpretable model | ~94.8% acc, 0.957 AUC, 0.919 macro-F1 |
| **Mixed-source, externally validated** | **Materially lower — plan for it** |
| Clinical/phone-only subset | AUC ~0.72 (de novo CNN vs. derm 0.81) |

---

## 10. Recommended Build Sequence

1. **Assemble & stratify** all three sources; build a stable combined-source validation split *first*.
2. **Standardize** size/resolution/color; apply CLAHE; strip artifacts (hair, ink).
3. **(Optional) U-Net ROI segmentation** — but validate masks on the phone subset.
4. **Baseline:** EfficientNet-B3/B4, ImageNet (or noisy-student) weights, fine-tuned; focal loss; consider a 9-class diagnosis target with MEL-prob readout.
5. **Augment** with the Albumentations stack (CLAHE, Cutout, geometric, blur); add TTA at inference.
6. **Fuse metadata** (age, sex, site) via 2 FC layers if available.
7. **Tune decision threshold** on validation for max F2.
8. **Benchmark** vs. DenseNet-201 and ResNet-50 under identical conditions.
9. **Small ensemble (3–5 models, varied input sizes, rank-averaged)** for a reliable lift.
10. **Add Grad-CAM/SHAP**; audit attention regions per source.
11. **External validation** on a held-out source + ISIC 2019; report F1/F2/recall/AUC/balanced-acc **per source**.
12. **Escalate** to CNN-ViT hybrid / attention modules only if a measured gap justifies it.

---

## References

**Large-scale challenges & winning pipelines**
1. Ha, Liu, Liu — *Identifying Melanoma Images using EfficientNet Ensemble: Winning Solution to the SIIM-ISIC Melanoma Classification Challenge*, arXiv:2010.05351, 2020. https://arxiv.org/abs/2010.05351 · full text: https://ar5iv.labs.arxiv.org/html/2010.05351
2. Gessert et al. — *Skin Lesion Classification Using Ensembles of Multi-Resolution EfficientNets with Meta Data* (ISIC 2019), arXiv:1910.03910. https://arxiv.org/pdf/1910.03910
3. Multiclass Skin Cancer Classification using EfficientNets (HAM10000, B0–B7) — *ScienceDirect*, 2021. https://www.sciencedirect.com/science/article/pii/S2772528621000340
4. Skin lesion classification & Prediction by Data Augmentation in HAM10000 and ISIC 2019 — 2022. https://www.academia.edu/107025321/

**External validation & generalization (the reality check)**
5. *Validation of AI prediction models for skin cancer diagnosis: the 2019 ISIC Grand Challenge* — **The Lancet Digital Health**, 2022. https://www.thelancet.com/journals/landig/article/PIIS2589-7500(22)00021-8/fulltext
6. *Stress testing reveals gaps in clinic readiness of image-based diagnostic AI models* — *npj Digital Medicine* / PMC, 2021. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7820258/
7. *Discrimination Between Invasive and In Situ Melanomas Using Clinical Close-Up Images and a De Novo CNN* — PMC. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8476836/
8. Haenssle et al. — *Man against machine: CNN vs 58 dermatologists* — **Annals of Oncology**, 2018. https://www.annalsofoncology.org/article/S0923-7534(19)34105-5/fulltext
9. Esteva et al. — *Dermatologist-level classification of skin cancer with deep neural networks* (129,450 images) — *Nature* / PMC. https://pmc.ncbi.nlm.nih.gov/articles/PMC8382232/
10. *Clinical Application of Vision Transformers for Melanoma Classification: A Multi-Dataset Evaluation* (human+CNN fusion) — PMC. https://pmc.ncbi.nlm.nih.gov/articles/PMC12607522/

**Architecture comparisons & components**
11. Deep Learning Approaches for Skin Lesion Detection (38-architecture benchmark) — *Electronics (MDPI)*, 2025. https://www.mdpi.com/2079-9292/14/14/2785
12. Accurate Deep Learning Algorithms for Skin Lesion Classification (ResNet/DenseNet across ISIC 2016–2018) — *IIETA*. https://www.iieta.org/download/file/fid/138535
13. Comparative Analysis of EfficientNet and ResNet Models — 2024. https://www.researchgate.net/publication/383542189
14. Skin Lesions Classification and Segmentation: A Review — *IJACSA/The SAI*, 2021. https://thesai.org/Downloads/Volume12No10/Paper_60-Skin_Lesions_Classification_and_Segmentation.pdf
15. Enhancing skin lesion classification: a CNN approach with human baseline (EfficientNetB3, 170 dermatologists) — PMC. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12192895/

**Imbalance, segmentation, augmentation, XAI**
16. A Deep CNN Transformer Hybrid Model Using Focal Loss — *Diagnostics (MDPI)*, 2022. https://www.mdpi.com/2075-4418/13/1/72
17. Two-step U-Net → classifier on SIIM-ISIC 2020 (Stanford CS230) — https://tom-mcilwain.github.io/melanoma.html
18. Hybrid Deep Learning Framework (U-Net → Inception-ResNet-v2 → ViT) — *Diagnostics (MDPI)*, 2024. https://www.mdpi.com/2075-4418/14/19/2242
19. Test Time Augmentation and Explainable AI (ISIC 2019, TTA + Grad-CAM + t-SNE) — PMC. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11766406/
20. Multimodal skin lesion classification (XAI faithfulness: SHAP > Grad-CAM) — *Frontiers in Physiology*, 2026. https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2026.1717517/full
21. Systematic literature review of ML/DL for melanoma (34 studies) — PMC, 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11727731/

> **Standing caveat:** Single-dataset accuracies of 95–98% are common in the literature and almost never survive external, mixed-source validation. Treat your own per-source, externally-validated F2 as the only number that matters.
