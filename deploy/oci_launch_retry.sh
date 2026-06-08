#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# oci_launch_retry.sh
#
# Keep retrying a VM.Standard.A1.Flex (Ampere/ARM) launch until OCI capacity
# appears in your home region. Designed to beat the "Out of host capacity"
# error in single-AD regions like Paris (eu-paris-1).
#
# RUN THIS IN ORACLE CLOUD SHELL — the OCI CLI is pre-installed and
# pre-authenticated there, so there is NOTHING to set up (no API keys).
#   Console -> top-right terminal icon ">_" -> Cloud Shell.
#
# It auto-discovers your availability domain, a public subnet, and the latest
# Ubuntu 24.04 aarch64 image. The only thing you MUST provide is your
# compartment (root/tenancy) OCID.
# ---------------------------------------------------------------------------
set -uo pipefail

### ---- CONFIG (edit COMPARTMENT_ID; the rest have sane defaults) ----
# Root/tenancy compartment OCID. Find it: Console -> Profile (top-right) ->
# Tenancy -> copy the OCID (starts with "ocid1.tenancy...").
COMPARTMENT_ID="${COMPARTMENT_ID:-}"

# Start SMALL to beat capacity; resize up to 4/24 later (Edit instance + reboot).
OCPUS="${OCPUS:-1}"
MEM_GB="${MEM_GB:-6}"

DISPLAY_NAME="${DISPLAY_NAME:-mela-a1}"
SHAPE="VM.Standard.A1.Flex"
SLEEP="${SLEEP:-60}"          # seconds between attempts

# Optional overrides — auto-discovered if left blank:
AD_NAME="${AD_NAME:-}"
SUBNET_ID="${SUBNET_ID:-}"
IMAGE_ID="${IMAGE_ID:-}"
SSH_PUB="${SSH_PUB:-$HOME/.ssh/id_rsa.pub}"
### ------------------------------------------------------------------

die(){ echo "ERROR: $*" >&2; exit 1; }

[ -n "$COMPARTMENT_ID" ] || die "Set COMPARTMENT_ID to your tenancy/root OCID (see comment above)."

# SSH key: generate one in Cloud Shell if missing (you'll need the private key
# ~/.ssh/id_rsa to log in later — download it from Cloud Shell afterward).
if [ ! -f "$SSH_PUB" ]; then
  echo "No SSH key at $SSH_PUB — generating one (id_rsa / id_rsa.pub)..."
  ssh-keygen -t rsa -b 4096 -f "${SSH_PUB%.pub}" -N "" || die "ssh-keygen failed"
fi

# Don't create duplicates if an instance with this name already exists.
existing=$(oci compute instance list --compartment-id "$COMPARTMENT_ID" \
  --query "data[?\"display-name\"=='$DISPLAY_NAME' && \"lifecycle-state\"!='TERMINATED']|[0].id" \
  --raw-output 2>/dev/null || true)
if [ -n "${existing:-}" ] && [ "$existing" != "null" ]; then
  die "An instance named '$DISPLAY_NAME' already exists ($existing). Aborting to avoid duplicates."
fi

# Auto-discover availability domain (single-AD region -> the only one).
if [ -z "$AD_NAME" ]; then
  AD_NAME=$(oci iam availability-domain list --compartment-id "$COMPARTMENT_ID" \
    --query 'data[0].name' --raw-output) || die "could not list availability domains"
fi

# Auto-discover a PUBLIC subnet (prohibit-public-ip-on-vnic == false).
if [ -z "$SUBNET_ID" ]; then
  SUBNET_ID=$(oci network subnet list --compartment-id "$COMPARTMENT_ID" \
    --query 'data[?"prohibit-public-ip-on-vnic"==`false`]|[0].id' --raw-output) \
    || die "could not list subnets"
  { [ -n "$SUBNET_ID" ] && [ "$SUBNET_ID" != "null" ]; } \
    || die "No public subnet found. Create one (VCN wizard) or set SUBNET_ID."
fi

# Auto-discover newest Ubuntu 24.04 aarch64 image valid for this shape.
if [ -z "$IMAGE_ID" ]; then
  IMAGE_ID=$(oci compute image list --compartment-id "$COMPARTMENT_ID" \
    --operating-system "Canonical Ubuntu" --operating-system-version "24.04" \
    --shape "$SHAPE" --sort-by TIMECREATED --sort-order DESC \
    --query 'data[?contains("display-name", `aarch64`)]|[0].id' --raw-output) \
    || die "could not list images"
  { [ -n "$IMAGE_ID" ] && [ "$IMAGE_ID" != "null" ]; } \
    || die "No Ubuntu 24.04 aarch64 image found. Set IMAGE_ID manually."
fi

echo "=== Launch config ==="
echo "  compartment : $COMPARTMENT_ID"
echo "  AD          : $AD_NAME"
echo "  subnet      : $SUBNET_ID"
echo "  image       : $IMAGE_ID"
echo "  shape       : $SHAPE  ($OCPUS OCPU / ${MEM_GB} GB)"
echo "  ssh pub key : $SSH_PUB"
echo "  retry every : ${SLEEP}s"
echo "  name        : $DISPLAY_NAME"
echo "====================="
echo "Keep this Cloud Shell tab OPEN. Ctrl-C to stop."

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "[$(date '+%H:%M:%S')] attempt #$attempt ..."
  out=$(oci compute instance launch \
    --availability-domain "$AD_NAME" \
    --compartment-id "$COMPARTMENT_ID" \
    --shape "$SHAPE" \
    --shape-config "{\"ocpus\":$OCPUS,\"memoryInGBs\":$MEM_GB}" \
    --image-id "$IMAGE_ID" \
    --subnet-id "$SUBNET_ID" \
    --assign-public-ip true \
    --display-name "$DISPLAY_NAME" \
    --ssh-authorized-keys-file "$SSH_PUB" 2>&1)
  rc=$?

  if [ $rc -eq 0 ]; then
    echo "============================================================"
    echo "SUCCESS — instance is provisioning!"
    echo "$out" | grep -iE '"id"|"display-name"|"lifecycle-state"' | head -5
    echo "Get its public IP once RUNNING:"
    echo "  oci compute instance list-vnics --instance-id <INSTANCE_OCID> --query 'data[0].\"public-ip\"' --raw-output"
    echo "============================================================"
    break
  fi

  if echo "$out" | grep -qiE 'out of (host )?capacity|capacity|TooManyRequests|429|500'; then
    echo "  out of capacity — retrying in ${SLEEP}s"
  else
    echo "  NON-capacity error — stopping so you can read it:"
    echo "$out"
    exit 1
  fi
  sleep "$SLEEP"
done
