# Windows setup — hunt for Oracle A1 capacity from your PC

Run the ARM-instance retry loop natively in **PowerShell** (no WSL/Git Bash).
Leave your PC awake and it will grab an `VM.Standard.A1.Flex` the moment
capacity frees up in Paris.

## Step 1 — Get your OCIDs (copy these from the console)

- **Tenancy OCID:** Console → profile icon (top-right) → **Tenancy** → copy OCID
  (`ocid1.tenancy.oc1..…`). This is also your compartment.
- **User OCID:** profile icon → **My profile** → copy OCID (`ocid1.user.oc1..…`).
- **Region:** `eu-paris-1`.

## Step 2 — Install the OCI CLI (PowerShell)

Open **PowerShell** (normal user is fine) and run:

```powershell
Set-ExecutionPolicy RemoteSigned -Scope Process -Force
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocol]::Tls12
Invoke-WebRequest https://raw.githubusercontent.com/oracle/oci-cli/master/scripts/install/install.ps1 -OutFile install.ps1
.\install.ps1 -AcceptAllDefaults
```

**Close and reopen PowerShell** afterward (so the new PATH takes effect), then verify:

```powershell
oci --version
```

## Step 3 — Authenticate (`oci setup config`)

```powershell
oci setup config
```

Answer the prompts:
- **config location:** accept default (`C:\Users\<you>\.oci\config`).
- **user OCID:** paste from Step 1.
- **tenancy OCID:** paste from Step 1.
- **region:** `eu-paris-1`.
- **generate a new API signing key?** → **Y** (accept default paths/passphrase = blank).

It prints where it wrote the **public key** (e.g. `~/.oci/oci_api_key_public.pem`).

## Step 4 — Add the API public key to Oracle

Console → profile icon → **My profile → API keys → Add API key → Paste public key**.
Open the public-key file from Step 3 and paste its full contents. Save.

Test it works:

```powershell
oci iam region list --output table
```

If that returns a table, auth is good.

## Step 5 — Make sure OpenSSH client is available

The script generates the SSH key you'll use to log into the VM. Check:

```powershell
ssh-keygen --help
```

If "not recognized": **Settings → Apps → Optional features → Add → OpenSSH Client → Install.**

## Step 6 — Keep the PC awake while it hunts

```powershell
powercfg /change standby-timeout-ac 0   # never sleep on AC power
```

(or Settings → System → Power → Sleep → Never). The loop dies if the PC sleeps.

## Step 7 — Run the retry loop

```powershell
# get the script
Invoke-WebRequest https://raw.githubusercontent.com/404MiguelMarinhoNotFound/mar-mela-cnn-live/claude/melanoma-detection-vercel-8Vm7k/deploy/oci_launch_retry.ps1 -OutFile oci_launch_retry.ps1

# set your tenancy OCID and run
$env:COMPARTMENT_ID = "ocid1.tenancy.oc1..PASTE_YOURS"
.\oci_launch_retry.ps1
```

It auto-finds your AD / public subnet / Ubuntu 24.04 aarch64 image, then loops
every 60 s. When capacity appears you'll see **`SUCCESS - instance is
provisioning!`** in green. Leave the window open until then.

- Bigger box (lower odds): `.\oci_launch_retry.ps1 -Ocpus 4 -MemGb 24`
- Default is **1 OCPU / 6 GB** on purpose — far better odds; resize up later.

## Step 8 — After success

- Find the **public IP**: Console → Compute → Instances → `mela-a1`.
- Your login key is `C:\Users\<you>\.ssh\id_rsa`. Connect:
  ```powershell
  ssh -i $env:USERPROFILE\.ssh\id_rsa ubuntu@<PUBLIC_IP>
  ```
- Then we move on to installing the inference server (see
  `../docs/oracle_a1_setup_runbook.md`, Phase 3+).

## Troubleshooting

- **`oci` not recognized:** reopen PowerShell; if still missing, add
  `C:\Users\<you>\bin` to PATH.
- **Auth errors (NotAuthenticated):** the API public key isn't added yet (Step 4),
  or the fingerprint/region in `~/.oci/config` is wrong.
- **Script prints a NON-capacity error:** it stops on purpose — read it / send it over.
- **Want it unattended without leaving your PC on:** use the GitHub Actions cron
  instead (`.github/workflows/oci-a1-retry.yml`).
