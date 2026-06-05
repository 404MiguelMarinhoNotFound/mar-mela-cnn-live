# Setup runbook — Oracle A1 + Cloudflare Tunnel + Vercel

> Companion to `oracle_a1_serving_design.md`. This is the **ordered, do-this-now**
> checklist to stand up the free always-on inference host. Commands assume the
> Ubuntu 24.04 **aarch64** (ARM) image on an Oracle Always Free A1 VM.
> Verified against Oracle + Cloudflare docs, June 2026 (links at bottom).

## Phase 0 — Accounts you'll need (all free)

- [ ] **Oracle Cloud** account (requires a card for identity check — temporary ~$1
      authorization hold, not charged; Always Free stays free).
- [ ] **Cloudflare** account (free plan) — for the Tunnel that fronts the VM with HTTPS.
- [ ] **A domain on Cloudflare** *(decision — see Phase 5)*. A named tunnel needs a
      hostname; a quick tunnel gives a random URL that changes on restart (bad for a
      fixed Vercel env var). Cheapest path to a **stable** endpoint = put any cheap
      domain on Cloudflare's free DNS.
- [ ] **Vercel** account + the GitHub repo connected (for the UI + `/api/predict` later).

## Phase 1 — Create the Oracle account

1. Sign up at cloud.oracle.com → "Start for free". Verify identity with a card.
2. **Pick your Home Region carefully — it is permanent and Always Free resources are
   locked to it.** Choose a large, multi-AD region for ARM capacity:
   **US East (Ashburn)** or **UK South (London)** are the usual good picks.

## Phase 2 — Provision the A1 ARM VM

Console → **Compute → Instances → Create instance**:

1. **Image:** "Canonical Ubuntu 24.04 **Minimal aarch64**" (aarch64 = required for ARM).
2. **Shape:** change to **Ampere (ARM-based) → VM.Standard.A1.Flex**. Slide to the full
   **4 OCPU / 24 GB RAM** (the whole always-free allowance). Prefer **On-demand capacity**.
3. **SSH keys:** "Generate a key pair for me" → **download the private key NOW**
   (unrecoverable later). Or paste your own public key.
4. **Networking:** create a new VCN + public subnet; **toggle ON "Assign a public IPv4
   address"**.
5. **Boot volume (optional):** custom size up to **200 GB** (free quota).
6. Create.

### If you hit "Out of capacity" (common for A1)

- Retry across **Availability Domains** (AD-1/2/3) in the Placement section.
- Retry at off-peak times, or script the create-instance call on a loop.
- Surest fix: **upgrade the account to Pay-As-You-Go** (add card). You keep the Always
  Free resources and aren't charged while within free limits — PAYG just grants priority
  to ARM hardware.

### If no public IP appeared

Instance → **Attached VNICs → Primary VNIC → IP Addresses → Edit primary private IP →
assign Ephemeral Public IP**. Copy the IP.

## Phase 3 — First login + base setup

```bash
# from your machine (chmod 600 the key on macOS/Linux)
ssh -i /path/to/private.key ubuntu@<PUBLIC_IP>

# on the VM
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-venv python3-pip

# Oracle Ubuntu images block inbound by default at the OS firewall.
# Cloudflare Tunnel is OUTBOUND-ONLY, so you do NOT need to open 80/443 inbound.
# (Only open ports if you choose the Caddy/public-IP route instead of a tunnel.)
```

> Security note: with a Cloudflare Tunnel you can leave **all inbound ports closed** —
> the VM only makes an outbound connection to Cloudflare on port 7844. That's the secure
> default; skip the `iptables` port-opening unless you deliberately go public-IP + Caddy.

## Phase 4 — Deploy the inference server

1. Copy the exported model artifact(s) to the VM (`scp` the `.onnx`, plus the `.pt` if
   keeping the two-path Grad-CAM design — see design doc §2–3).
2. App layout on the VM, e.g. `/opt/mela/`:
   ```bash
   python3 -m venv /opt/mela/venv
   /opt/mela/venv/bin/pip install fastapi uvicorn[standard] onnxruntime pillow numpy
   # add torch (CPU aarch64 wheel) ONLY if serving Grad-CAM
   ```
3. Run the FastAPI app (`/predict`, `/health` per design doc §5) under **systemd** so it
   survives reboots/crashes:
   ```ini
   # /etc/systemd/system/mela.service
   [Service]
   ExecStart=/opt/mela/venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
   WorkingDirectory=/opt/mela
   Restart=always
   User=ubuntu
   [Install]
   WantedBy=multi-user.target
   ```
   ```bash
   sudo systemctl enable --now mela
   curl localhost:8000/health   # expect {"status":"ok",...}
   ```
   (Bind to `127.0.0.1` — only the tunnel needs to reach it.)

## Phase 5 — Cloudflare Tunnel (free HTTPS, stable hostname)

Decision: **named tunnel (recommended)** vs quick tunnel.

- **Quick tunnel** (`cloudflared tunnel --url http://localhost:8000`): zero setup, but
  the `*.trycloudflare.com` URL is **random and changes on restart** → only for testing.
- **Named tunnel** (recommended): stable `api.yourdomain.com`, survives restarts. Needs
  a domain on your Cloudflare account.

Named tunnel steps:
```bash
# install cloudflared (ARM64)
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64 \
  -o cloudflared && sudo install cloudflared /usr/local/bin/

cloudflared tunnel login                 # opens a browser auth; pick your domain
cloudflared tunnel create mela           # creates tunnel + credentials json
cloudflared tunnel route dns mela api.yourdomain.com
```
```yaml
# ~/.cloudflared/config.yml
tunnel: <TUNNEL_ID>
credentials-file: /home/ubuntu/.cloudflared/<TUNNEL_ID>.json
ingress:
  - hostname: api.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
```
```bash
sudo cloudflared service install        # run the tunnel as a systemd service
curl https://api.yourdomain.com/health  # HTTPS works, no open inbound ports
```

> Verify the VM can reach Cloudflare on port **7844** outbound (default-open on Oracle).

## Phase 6 — Wire Vercel `/api/predict`

- [ ] Add Vercel env vars: `INFERENCE_URL=https://api.yourdomain.com`,
      `INFERENCE_TOKEN=<shared secret>` (also set the same token on the VM server).
- [ ] `/api/predict` Route Handler: validate content-type + size, rate-limit, inject
      `X-Api-Token`, forward the image to `INFERENCE_URL/predict`, return normalized JSON
      (contract in design doc §5–6). Token stays server-side, never in the browser.

## Phase 7 — Verify end-to-end + keep it alive

- [ ] `curl https://api.yourdomain.com/health` → ok.
- [ ] POST a test image (with token) → sensible `{probability,label,threshold}`.
- [ ] Call the Vercel `/api/predict` from the deployed app → same result.
- [ ] **Uptime:** add an external pinger (UptimeRobot free, or a Vercel Cron hitting
      `/health`) — always-free VMs have no auto-healing, so monitor the process.
- [ ] Reboot the VM once and confirm `mela` + `cloudflared` come back automatically.

## One-screen checklist

```
[ ] Oracle account, home region = multi-AD (Ashburn/London)
[ ] A1 Flex VM: Ubuntu 24.04 aarch64, 4 OCPU / 24 GB, public IP, key downloaded
[ ] (capacity issues?) retry ADs / off-peak / upgrade to PAYG
[ ] SSH in, apt update, python venv
[ ] scp model artifact(s); pip install onnxruntime+fastapi (+torch if Grad-CAM)
[ ] systemd: mela.service Restart=always, bound to 127.0.0.1:8000
[ ] Cloudflare: domain added; named tunnel -> api.yourdomain.com -> :8000; service install
[ ] Vercel: env vars + /api/predict proxy with shared token
[ ] verify health + predict end-to-end; add uptime pinger; test reboot recovery
```

## Decisions still open before/while doing this

1. **Domain for the tunnel** — buy/point a cheap domain to Cloudflare (stable hostname),
   or accept the throwaway quick-tunnel URL for an early test only?
2. **Grad-CAM in v1?** — determines whether we also install torch on the VM (design §2).
3. **Region** — confirm Ashburn vs London (latency to your users vs ARM availability).

## Sources

- Oracle A1 Always Free setup guide (Medium, Vinojan V.)
- Oracle Cloud Free Tier breakdown (fullmetalbrackets)
- Cloudflare Tunnel setup docs (developers.cloudflare.com/tunnel)
