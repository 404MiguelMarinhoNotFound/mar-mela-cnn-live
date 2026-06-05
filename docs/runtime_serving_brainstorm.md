# Runtime serving brainstorm — melanoma classifier on a Vercel website

> Status: **brainstorm / architecture only** (no UI yet). Goal: let a user upload or
> capture a live photo of a mole on a simple web UI and get a melanoma-vs-benign
> readout, with the trained model reachable at runtime.

## 0. TL;DR

- **Vercel cannot run this model directly.** Serverless/Fluid functions have **no GPU**,
  a **~250 MB bundle limit**, and short default durations. The full PyTorch stack
  (torch alone is ~800 MB+) and the 5-model ensemble do not fit.
- So Vercel plays **frontend + thin API/orchestration**. The model runs in one of:
  - **A. External scale-to-zero GPU inference service** (Modal / Replicate / HF
    Endpoints / Cloud Run-GPU) — *keeps full fidelity + Grad-CAM*. **Recommended.**
  - **B. Shrunk ONNX model on Vercel CPU** (Fluid compute) — no external infra, but
    forces dropping the ensemble/TTA → accuracy regression.
  - **C. In-browser inference** (ONNX Runtime Web / WebGPU) — zero inference cost,
    image never leaves the device (privacy win for medical), but heavy on mobile and
    Grad-CAM is hard.
- **Recommended path:** Phase 1 ship a single quantized model (B fast demo), Phase 2
  move to **A** (full ensemble on Modal/Replicate behind a Vercel `/api/predict`
  route) for the documented F2 ≈ 0.905 performance + Grad-CAM overlays.

## 1. What we're actually serving (the constraint that drives everything)

From `CLAUDE.md` / doctrine, the production artifact is:

| Property | Value | Serving implication |
|---|---|---|
| Architecture | 5-model ensemble: 3× EfficientNet-B3 @ 300px + 2× B4 @ 380px | ~70M params total; too big for Vercel bundle |
| Inference | TTA 4-view (identity, H-flip, V-flip, transpose) averaged | **20 forward passes/image** (5 models × 4 views) |
| Runtime | PyTorch + timm | torch wheel alone >> 250 MB function limit |
| Decision | threshold **0.29** tuned on ensemble val probs (NOT 0.5) | must be applied server-side; return prob *and* label |
| Explainability | Grad-CAM overlay per image | needs the model graph + hooks → server-side, not trivial in browser |
| Metric posture | optimizes **F2** (recall-weighted); clinical/phone subset AUC ~0.72 | phone uploads are out-of-distribution → surface uncertainty |
| Checkpoints | `checkpoints/*_seed*_best.pt`, **gitignored, not in repo** | artifacts must be hosted (HF Hub / S3 / Modal volume), not bundled |

Two hard truths:
1. **A live phone photo is the model's weakest domain.** The doctrine reports clinical/
   phone-only AUC ~0.72 vs dermoscopic 0.92–0.96. Whatever we build must communicate
   uncertainty, not a confident "diagnosis."
2. **Preprocessing parity is non-negotiable.** Inference must replicate the exact
   training transform: `A.Resize(img_size, img_size)` (300 for B3, 380 for B4) +
   ImageNet normalize. A silent mismatch (wrong resize, BGR/RGB swap, wrong norm)
   degrades accuracy with no error. This must be ported verbatim wherever inference runs.

## 2. Why not "just run it on Vercel"

| Vercel limit | Reality | Verdict for this model |
|---|---|---|
| Function bundle size | ~250 MB unzipped | torch + 5 checkpoints (~280 MB weights fp32 alone) → **over** |
| GPU | none on Functions/Fluid | 20× B3/B4 forward passes on CPU = multiple seconds → slow |
| Max duration | 10s default (Hobby); up to 60–300s on paid/Fluid | borderline for full CPU ensemble |
| Memory | up to ~3–4 GB | loadable, but CPU latency still poor |
| Cold start | yes | big Python/torch import = slow first hit |

Vercel *is* the right place for: the UI, request validation, rate-limiting / abuse
controls (Firewall, BotID), auth, calling the model, and rendering results.

## 3. Option A — Vercel frontend + external GPU inference (recommended)

```
[ Browser ]                 [ Vercel ]                         [ Inference service ]
 camera /        --HTTPS-->  Next.js UI                         Modal / Replicate /
 file upload                 /api/predict  --HTTPS+token-->     HF Endpoint / Cloud Run-GPU
                             - validate img                      - load 5 checkpoints once
                             - rate limit                        - TTA ensemble (existing
                             - forward image                       `melanoma` package code)
                             <----------- JSON {prob,label,        - Grad-CAM overlay
                                          gradcam_png} <---------- - apply threshold 0.29
```

- **Keeps the existing Python code unchanged** — wrap `melanoma.engine.evaluate` /
  the ensemble TTA logic in a `predict(image_bytes)` function on the host.
- **Host candidates (scale-to-zero, pay-per-second):**
  - **Modal** — wrap the `melanoma` package, mount checkpoints on a Modal Volume, load
    once per warm container. Easiest to lift existing code. Scale to zero.
  - **Replicate** — Cog container, public/private model, simple HTTP. Good DX.
  - **Hugging Face Inference Endpoints** — host checkpoints on the Hub, autoscaling.
  - **Cloud Run (GPU) / RunPod / Baseten** — more control, more ops.
- **Pros:** full fidelity (F2 ≈ 0.905 preserved), Grad-CAM works, no model surgery,
  one place owns the weights.
- **Cons:** GPU cost + cold starts on the inference side; an extra service to operate;
  end-to-end latency = network + 20 forward passes (~1–3 s warm, more on cold).
- **Vercel pieces:** `/api/predict` Route Handler (Node runtime) proxies the image,
  enforces payload cap + content-type, rate-limits (Upstash/Vercel KV counter), injects
  the inference token from env vars, normalizes the JSON response. Optionally Fluid
  compute + `maxDuration` bump so the proxy can wait out a cold inference start.

## 4. Option B — Shrunk ONNX on Vercel CPU (Fluid compute, no external infra)

- Export model(s) to **ONNX**, run with `onnxruntime` in a Python function under `/api`.
- **Must shrink to fit:** the 5-model fp32 ensemble (~280 MB) exceeds the bundle limit.
  Realistic only with **int8 quantization** + **fewer models** (e.g. single best **B4
  seed42**, val-F2 0.9203) and likely **dropping TTA**.
- **Pros:** everything on Vercel, one deploy, no GPU bill.
- **Cons:** sacrifices the ensemble + TTA → measurable F2 drop; CPU latency near the
  duration ceiling; Grad-CAM via onnxruntime is awkward. A degraded-but-self-contained
  option.

## 5. Option C — In-browser inference (ONNX Runtime Web / WebGPU / TF.js)

- Convert to ONNX, ship to the client, run with `onnxruntime-web` (WASM fallback, WebGPU
  when available). Vercel just serves static assets + the model file.
- **Pros:** **image never leaves the device** (strong privacy story for a medical demo),
  zero inference cost, infinite scale, no cold starts.
- **Cons:** client downloads tens of MB of weights (mitigate: quantize + HTTP cache +
  one model, not five); WebGPU not universal, WASM path is slow on mobile; Grad-CAM is
  hard client-side; the full 5-model ensemble is too heavy for phones → reduce to 1.
- **Best when:** privacy-first "instant on-device screening" framing is the priority and
  a single-model accuracy trade is acceptable.

## 6. Option D — Hybrid / tiered (longer-term)

- A small quantized model **in-browser** for an instant preview, with an optional
  "get full second opinion" button that POSTs to the **Option A** GPU ensemble for the
  high-fidelity result + Grad-CAM. Best of both: instant + private by default, full
  power on demand.

## 7. Recommendation & phased rollout

1. **Phase 1 — fast demo:** single best model (B4 seed42) → ONNX (int8), served either
   in-browser (Option C, privacy) or on Vercel CPU (Option B). Proves the UX + the
   preprocessing-parity pipeline end to end with low cost.
2. **Phase 2 — production fidelity:** stand up **Option A** (full 5-model ensemble + TTA
   + Grad-CAM on Modal/Replicate) behind a Vercel `/api/predict` route. Swap the frontend
   from the Phase-1 endpoint to this one — UI is unchanged.
3. **Phase 3 — hybrid (optional):** add the on-device preview + "full second opinion"
   tiering from Option D.

### Concrete recommended stack (Phase 2 target)

- **Frontend:** Next.js (App Router) on Vercel. Capture via `getUserMedia` (live camera,
  mobile-friendly) + file-upload fallback. Client-side resize/compress before upload.
- **API:** Next.js Route Handler `/api/predict` (Node). Validates content-type + size
  cap, rate-limits, forwards to inference, returns
  `{ probability, label, threshold: 0.29, domain_hint, gradcam_png_b64 }`.
- **Inference:** Modal (or Replicate) Python function importing the existing `melanoma`
  package; checkpoints on a Modal Volume / HF Hub; load once per warm container; run TTA
  ensemble + Grad-CAM; apply threshold server-side.
- **State:** **no image persistence** (ephemeral). Optional Upstash/Vercel KV for
  rate-limit counters. Secrets (inference URL + token) in Vercel env vars.

## 8. Cross-cutting concerns (especially for a medical tool)

- **Not a diagnosis.** Prominent disclaimer: screening/educational aid, "consult a
  dermatologist." Aligns with doctrine §9 honesty posture.
- **Out-of-distribution input.** Phone photos ≈ clinical/unknown domain (AUC ~0.72).
  Return a domain/uncertainty hint, not just a binary verdict. Consider rejecting
  blurry / low-light / non-skin images (a cheap quality gate before inference).
- **Threshold semantics.** Always apply the val-tuned **0.29** server-side and return
  both the raw probability and the F2-oriented decision; never imply 0.5.
- **Privacy / data handling.** Default to ephemeral, no storage. If using Option A,
  state clearly that the image is sent to a processing service; in Option C it never
  leaves the device.
- **Abuse / cost control.** Payload size cap, image-type allowlist, rate limiting,
  Vercel Firewall/BotID. GPU inference is metered — guard it.
- **Cold-start UX.** Show a "warming up" state; set `maxDuration` high enough on the
  proxy to survive an inference cold start.
- **Preprocessing parity (repeat, because it's the #1 silent failure mode):** port the
  exact `eval_transforms` (Resize to backbone size + ImageNet normalize, RGB) to
  wherever inference runs.
- **Artifact hosting.** Checkpoints are gitignored and not in the repo — Phase 2 needs
  them uploaded to the inference host's storage (Modal Volume / HF Hub / S3).

## 8a. Free hosting — deep dive ("free + decent runtimes")

The hard part: **free**, **decent latency**, and **a heavy CV ensemble** pull against
each other. There's a trade-off triangle — you can pick two cleanly, and engineer
around the third:

```
        FREE GPU (fast)                 ALWAYS-ON (no cold start)
        but cold starts / quotas         but CPU-only
        - HF ZeroGPU (~3.5 min/day)      - Oracle Always Free A1 (4 ARM, 24GB)
        - Modal ($30/mo credits)         - (Cloud Run = scale-to-zero → cold start)
                       \                 /
                        \               /
                   ON-DEVICE (free, infinite scale, no cold start)
                   but single model, client compute
                   - in-browser ONNX Runtime Web / WebGPU
```

### The lever that makes free CPU viable: shrink the model

Free hosting almost always means **CPU**. The 5-model fp32 + TTA ensemble (20 forward
passes) is too slow there. Convert to **ONNX** and pick a point on this curve:

| Config | ~Latency on 2–4 CPU cores | F2 vs full ensemble |
|---|---|---|
| Full 5-model + TTA ×4 (20 passes) | ~3–8 s | baseline (0.905) |
| 2–3 models, no TTA | ~1–3 s | small drop |
| **Single B4 (seed42), int8, no TTA** | **~0.2–0.8 s** | modest drop (single val-F2 0.92) |

Single int8 model is the sweet spot for a free demo: sub-second, fits any free tier,
Grad-CAM still works. Keep the full ensemble for a later paid/GPU tier.

### Ranked free options for THIS project

**1. Oracle Cloud Always Free A1 VM — best "free + always-on + decent latency".** ⭐
- 4 ARM cores, 24 GB RAM, **never expires** (not a trial/credit). Run a persistent
  `FastAPI + onnxruntime` server → **no cold start**, consistent latency.
- 24 GB RAM holds all 5 ONNX models comfortably; run single-model sub-second or a
  reduced ensemble in 1–3 s. Grad-CAM works.
- Put **Cloudflare Tunnel (free)** or **Caddy (auto-HTTPS)** in front → free TLS, hides
  the VM IP, gives a stable hostname for the Vercel `/api/predict` route to call.
- Cons: most setup/ops of the free options; A1 capacity is frequently "out of capacity"
  in single-AD regions (mitigate: pick a 3-AD region e.g. Ashburn/London, or flip to
  PAYG with a card — keeps the always-free resources, just guarantees provisioning).

**2. Hugging Face Space (free CPU Basic) — best zero-ops free option.**
- 2 vCPU / 16 GB, public HTTPS endpoint out of the box, `git push` to deploy. Wrap the
  `melanoma` package in FastAPI (Docker Space) or Gradio (exposes a REST API too).
- Cons: only 2 vCPU (slower → lean toward single int8 model); **free Spaces sleep when
  idle → ~30–60 s cold start** on the first hit after inactivity (UX: "warming up").
- ZeroGPU is *not* a fit for a backend API here: ~3.5 min/day free quota + Gradio-only.

**3. Modal ($30/mo free credits) — best free-ish GPU latency, bursty traffic.**
- True scale-to-zero GPU, sub-second-ish cold starts, runs the **full ensemble fast**.
  $30/mo covers a lot at demo volume (A10G ≈ $1.10/hr, billed per-ms of actual run) →
  effectively free for low traffic. Lifts the existing Python code with minimal change.
- Cons: it's credits, not free-forever; needs a card; cost grows if traffic does.

**4. In-browser ONNX / WebGPU — only genuinely $0, no-account, no-cold-start, ∞ scale.**
- Vercel serves a static ONNX model; inference runs on the user's device. **Image never
  leaves the device** (privacy win). WebGPU is fast on modern phones/laptops; WASM
  fallback is slower. Single model; Grad-CAM is hard client-side.

**Also-ran:** Google Cloud Run free tier (CPU, scale-to-zero, generous free request
allowance) — viable for an ONNX server, but scale-to-zero means cold starts, so Oracle's
always-on A1 is strictly better for "decent runtime" at the same $0.

### Recommended free architecture

```
[ Browser ]  --->  [ Vercel: Next.js UI + /api/predict proxy ]  --->  [ free model host ]
 camera / upload     validate · rate-limit · forward image            Oracle A1 VM:
                     return {prob,label,0.29,gradcam}                  FastAPI + onnxruntime
                                                                       (single int8 B4, no cold start)
                                                                       behind Cloudflare Tunnel (HTTPS)
```

- **Primary:** Oracle A1 + FastAPI + single int8 ONNX model, fronted by Cloudflare Tunnel.
  $0 forever, no cold start, sub-second, Grad-CAM intact.
- **If you want zero server ops instead:** HF CPU Space (accept idle cold starts).
- **If you later want the full-ensemble fidelity:** swap the proxy target to Modal — UI
  and `/api/predict` contract stay identical.

## 9. Open questions for the user

1. **Privacy vs fidelity:** is "image never leaves device" (Option C, single model) more
   important than top accuracy + Grad-CAM (Option A, full ensemble)?
2. **Budget:** OK to run a metered GPU endpoint (Modal/Replicate), or must it be
   $0-infra (in-browser / Vercel CPU only)?
3. **Grad-CAM required in v1?** It strongly favors a server path (Option A/B).
4. **Where do the trained checkpoints live now**, and can we upload them to a model host?
5. **Expected traffic / scale** — demo-only vs public launch — changes the cost calculus.
