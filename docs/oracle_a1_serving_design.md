# Serving design — Oracle A1 (always-free) + single int8 model

> Chosen path from `runtime_serving_brainstorm.md`: **free + decent runtime**.
> Host = **Oracle Cloud Always Free A1** VM (4 ARM OCPU, 24 GB RAM, never expires).
> Model = **single best model, ONNX int8** (instead of the 5-model ensemble).
> Still architecture/plan — no UI yet.

## 1. Topology

```
[ Browser ]                [ Vercel ]                    [ Oracle A1 VM (always-on) ]
 camera / file   --HTTPS--> Next.js UI                    Cloudflare Tunnel (free TLS,
 upload                     /api/predict (proxy)  --TLS-->  hides VM IP, stable hostname)
                            - size/type guard               |
                            - rate limit                    v
                            - inject shared token          FastAPI (uvicorn, systemd)
                            - normalize JSON               - onnxruntime: int8 B4 (fast path)
                            <--- {prob,label,0.29, ...} -- - torch (lazy): Grad-CAM only
```

- **Vercel never sees the VM IP** — it calls a Cloudflare Tunnel hostname. The tunnel
  gives free HTTPS and DDoS shielding, and means no inbound ports/certs on the VM.
- **Always-on** → no cold start. The single model loads once at boot and stays resident.

## 2. Why single-model + the Grad-CAM nuance (important)

We're dropping the ensemble + TTA for speed/footprint. Use the **best single model,
B4 seed42** (val-F2 ≈ 0.9203 per `CLAUDE.md`), at **380×380**.

**Grad-CAM needs autograd — ONNX int8 inference can't produce it.** Two clean ways to
keep explainability on this always-on box (24 GB RAM fits both):

- **Recommended — two-path server:** int8 **ONNX** for the fast probability (~0.2–0.8 s);
  keep a **CPU PyTorch** copy of the same model resident and compute Grad-CAM **only when
  requested** (existing `melanoma.explain.GradCAM`). Best of both.
- **Simpler alt:** skip ONNX entirely and run the **single PyTorch model on CPU**. One
  B4 forward on 4 ARM cores is already ~0.5–1 s, and Grad-CAM is free. On an *always-on*
  box, int8's main wins (cold-start, memory) don't apply — so this is a legitimate
  default. Choose ONNX int8 if you want minimal deps / max throughput.

> Decision to confirm: is Grad-CAM in v1 worth keeping a torch copy resident? If yes →
> two-path. If "probability only, fastest/smallest" → pure int8 ONNX, drop Grad-CAM.

## 3. Offline model-export pipeline (one-time, run where the checkpoint lives)

Checkpoints are gitignored and produced on the training box / Databricks, so this runs
there, then ships the artifact to the VM.

1. **Load** `tf_efficientnet_b4_seed42_best.pt` via `melanoma.models.backbone.build_model`
   (single-logit head). `img_size = 380` (from `config._BACKBONE_IMG_SIZE`).
2. **Export ONNX:** `torch.onnx.export(model, dummy(1,3,380,380), opset=17)`, fixed
   batch 1 (sigmoid applied server-side, or bake it in). Output = one logit.
3. **Quantize int8:** start with **dynamic quantization** (`onnxruntime.quantization
   .quantize_dynamic`) — no calibration data needed. If accuracy drops too far, use
   **static quantization** calibrated on a small `validate/` subset.
4. **Parity check (do not skip):** run ONNX-int8 vs the PyTorch model across `validate/`,
   compare probabilities + F2. **Re-tune the decision threshold on the int8 val probs** —
   the 0.29 from the *ensemble* won't be exactly right for a quantized single model.
   Record the new threshold in config.
5. **Ship** the `.onnx` (~10–20 MB int8) to the VM (scp / object storage / or commit a
   release artifact). Plus the PyTorch `.pt` if using the two-path Grad-CAM design.

## 4. Preprocessing parity (the #1 silent failure mode)

The server must reproduce `melanoma.data.transforms.eval_transforms` **exactly**:

- Decode → **RGB** (not BGR), resize to **380×380** (`A.Resize`),
- normalize with ImageNet stats: `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`,
- channels-first `CHW`, `float32`, batch dim → `(1,3,380,380)`.

Implement with Pillow + NumPy (no torch needed for the ONNX fast path → tiny deps).
A wrong resize / channel order / missing normalize degrades silently with no error.

## 5. FastAPI server contract

```
GET  /health           -> 200 {"status":"ok","model":"b4_seed42_int8","ver":"..."}
POST /predict          -> body: multipart image OR {image_b64}
                          headers: X-Api-Token: <shared secret>
                          resp: {
                            "probability": 0.83,        # sigmoid(logit)
                            "label": "melanoma",        # prob >= threshold
                            "threshold": 0.31,           # int8-retuned, from §3.4
                            "domain_hint": "clinical",   # phone photos = OOD, low-conf zone
                            "gradcam_png_b64": "..."     # optional, two-path only
                          }
```

- **Model loaded once at process start** (module global), reused across requests.
- **Token check** first — reject without the shared secret (cheap abuse gate).
- **No image persistence** — process in memory, discard. Medical privacy default.
- **Input guards:** content-type allowlist (jpeg/png/webp), max bytes (e.g. 6 MB),
  min resolution; optionally a cheap "is this skin / not blurry" gate before inference.
- **Grad-CAM** computed lazily (only if `?explain=1`) to keep the default path fast.

## 6. Vercel side (`/api/predict` proxy)

- Next.js Route Handler (Node runtime). Validates content-type + size, rate-limits
  (Upstash/Vercel KV counter keyed by IP), injects `X-Api-Token` from a Vercel env var,
  forwards to the Cloudflare Tunnel hostname, normalizes/forwards the JSON.
- Keeps the VM token server-side (never in the browser). Bump `maxDuration` a little to
  absorb the occasional slow request; no cold start to wait on since the VM is always-on.
- CORS: only the Vercel app origin may call the proxy; the VM only trusts the token.

## 7. VM operations

- **Process:** uvicorn behind systemd (`Restart=always`) **or** a small Docker container
  with `--restart unless-stopped`. Either survives reboots and crashes.
- **Reverse entry:** `cloudflared tunnel` as a systemd service → free HTTPS hostname,
  no open inbound ports, no Let's Encrypt to manage. (Caddy auto-HTTPS is the alt if you
  prefer a real domain + open 443.)
- **Health/uptime:** `/health` + an external pinger (e.g. a Vercel Cron or UptimeRobot)
  to catch a dead process. Always-free = no auto-healing, so monitor it.
- **Deps footprint:** ONNX fast path needs only `onnxruntime`, `pillow`, `numpy`,
  `fastapi`, `uvicorn` — light on ARM. Add `torch` (CPU, ARM wheel) only for the Grad-CAM
  path. Build/verify the ARM (aarch64) wheels — A1 is ARM, not x86.

## 8. Latency budget (single int8 B4, always-on)

| Stage | ~Time |
|---|---|
| Browser → Vercel → Tunnel → VM (network) | 50–200 ms |
| Decode + preprocess (Pillow/NumPy) | 10–40 ms |
| ONNX int8 forward (1×, 4 ARM cores) | **200–800 ms** |
| Grad-CAM (only if requested, torch CPU) | +0.3–1 s |
| **Total (no Grad-CAM)** | **~0.5–1.2 s** |

Consistent because there's **no cold start** — the model is always warm.

## 9. Risks / caveats

- **A1 capacity:** Ampere A1 is often "out of capacity" in single-AD regions. Mitigate:
  provision in a 3-AD region (Ashburn/London), retry on a script, or flip the account to
  PAYG (keeps the always-free resources) to guarantee provisioning.
- **No managed scaling/healing:** one VM = single point of failure; fine for a demo, add
  monitoring + auto-restart. For real traffic, fall back to Modal (full ensemble, GPU).
- **Accuracy trade:** single int8 model < the 0.905 F2 ensemble. Re-measure F2 on
  `validate/` after quantization and publish the honest number (doctrine §9).
- **OOD phone photos:** still the weak domain (clinical AUC ~0.72) — surface uncertainty,
  keep the "screening aid, not a diagnosis" disclaimer.

## 10. Build checklist (when we proceed)

- [ ] Export B4 seed42 → ONNX → int8; parity-check + **re-tune threshold** on int8 val probs.
- [ ] FastAPI server: `/health`, `/predict`, preprocessing parity, token gate, no persistence.
- [ ] (Optional) two-path Grad-CAM via resident CPU torch model.
- [ ] Provision Oracle A1, install ARM deps, systemd/Docker + `cloudflared` tunnel.
- [ ] Vercel `/api/predict` proxy: validation, rate limit, token injection, env var.
- [ ] Health pinger + honest per-domain accuracy note in the UI.
