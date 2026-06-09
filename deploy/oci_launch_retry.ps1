<#
  oci_launch_retry.ps1  —  Windows / PowerShell version

  Keep retrying a VM.Standard.A1.Flex (Ampere/ARM) launch until OCI capacity
  appears, to beat the "Out of host capacity" error in single-AD regions
  (e.g. Paris / eu-paris-1).

  PREREQS (one-time, see deploy/WINDOWS_SETUP.md):
    1. OCI CLI for Windows installed.
    2. `oci setup config` completed + API public key added in the console.

  USAGE:
    $env:COMPARTMENT_ID = "ocid1.tenancy.oc1..xxxx"
    .\deploy\oci_launch_retry.ps1
    # bigger box (lower odds):   .\deploy\oci_launch_retry.ps1 -Ocpus 4 -MemGb 24
    # single attempt (no loop):  .\deploy\oci_launch_retry.ps1 -Once
#>
[CmdletBinding()]
param(
  [string]$CompartmentId = $env:COMPARTMENT_ID,
  [int]$Ocpus = 1,
  [int]$MemGb = 6,
  [string]$DisplayName = "mela-a1",
  [int]$SleepSec = 120,
  [switch]$Once,
  [string]$AdName = "",
  [string]$SubnetId = "",
  [string]$ImageId = "",
  [string]$SshPub = "$env:USERPROFILE\.ssh\id_rsa.pub"
)

# NOTE: keep this 'Continue', not 'Stop'. The OCI CLI writes its ServiceError
# (incl. the expected "Out of host capacity") to stderr; under 'Stop' PowerShell
# turns that into a terminating NativeCommandError and kills the retry loop.
$ErrorActionPreference = "Continue"
function Die($m){ Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }

if (-not $CompartmentId) { Die "Set -CompartmentId or `$env:COMPARTMENT_ID to your tenancy/root OCID." }
if (-not (Get-Command oci -ErrorAction SilentlyContinue)) {
  Die "OCI CLI not found. Install it and run 'oci setup config' (see deploy/WINDOWS_SETUP.md)."
}
if (-not (Get-Command ssh-keygen -ErrorAction SilentlyContinue)) {
  Die "ssh-keygen not found. Enable Windows 'OpenSSH Client' (Settings > Apps > Optional features)."
}

# Run oci and return parsed .data (null on error/empty).
function Oci-Data([string[]]$ociArgs) {
  $raw = & oci @ociArgs 2>&1 | Out-String
  if ($LASTEXITCODE -ne 0) { return $null }
  try { return ($raw | ConvertFrom-Json).data } catch { return $null }
}

# SSH key for logging into the instance later (kept on THIS machine).
if (-not (Test-Path $SshPub)) {
  Write-Host "No SSH key at $SshPub - generating one..."
  $priv = $SshPub -replace '\.pub$',''
  $dir = Split-Path $priv
  if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
  ssh-keygen -t rsa -b 4096 -f $priv -N '""' | Out-Null
}

# Don't create duplicates.
$inst = Oci-Data @('compute','instance','list','--compartment-id',$CompartmentId,'--output','json')
$dup  = $inst | Where-Object { $_.'display-name' -eq $DisplayName -and $_.'lifecycle-state' -ne 'TERMINATED' } | Select-Object -First 1
if ($dup) { Write-Host "Instance '$DisplayName' already exists ($($dup.id)). Nothing to do."; exit 0 }

# Auto-discover availability domain (single-AD region -> the only one).
if (-not $AdName) {
  $ad = Oci-Data @('iam','availability-domain','list','--compartment-id',$CompartmentId,'--output','json')
  $AdName = ($ad | Select-Object -First 1).name
  if (-not $AdName) { Die "could not list availability domains (check auth)." }
}

# Auto-discover a PUBLIC subnet.
if (-not $SubnetId) {
  $sn = Oci-Data @('network','subnet','list','--compartment-id',$CompartmentId,'--output','json')
  $pub = $sn | Where-Object { -not $_.'prohibit-public-ip-on-vnic' } | Select-Object -First 1
  if (-not $pub) { Die "no public subnet found. Create one (VCN wizard) or pass -SubnetId." }
  $SubnetId = $pub.id
}

# Auto-discover newest Ubuntu 24.04 aarch64 image valid for this shape.
if (-not $ImageId) {
  $im = Oci-Data @('compute','image','list','--compartment-id',$CompartmentId,
    '--operating-system','Canonical Ubuntu','--operating-system-version','24.04',
    '--shape','VM.Standard.A1.Flex','--sort-by','TIMECREATED','--sort-order','DESC','--output','json')
  $img = $im | Where-Object { $_.'display-name' -match 'aarch64' } | Select-Object -First 1
  if (-not $img) { Die "no Ubuntu 24.04 aarch64 image found. Pass -ImageId." }
  $ImageId = $img.id
}

# shape-config via temp file (avoids PowerShell/native JSON-quoting issues).
$scFile = Join-Path $env:TEMP "oci_shapeconfig.json"
('{"ocpus":' + $Ocpus + ',"memoryInGBs":' + $MemGb + '}') | Set-Content -Path $scFile -NoNewline -Encoding ascii
$scUri = "file://" + ($scFile -replace '\\','/')

Write-Host "=== Launch config ==="
Write-Host "  compartment : $CompartmentId"
Write-Host "  AD          : $AdName"
Write-Host "  subnet      : $SubnetId"
Write-Host "  image       : $ImageId"
Write-Host "  shape       : VM.Standard.A1.Flex ($Ocpus OCPU / $MemGb GB)"
Write-Host "  ssh pub key : $SshPub"
Write-Host "  name        : $DisplayName"
Write-Host "====================="

function Try-Launch {
  $out = (& oci compute instance launch `
    --availability-domain $AdName `
    --compartment-id $CompartmentId `
    --shape "VM.Standard.A1.Flex" `
    --shape-config $scUri `
    --image-id $ImageId `
    --subnet-id $SubnetId `
    --assign-public-ip true `
    --display-name $DisplayName `
    --ssh-authorized-keys-file $SshPub 2>&1 | Out-String)
  $code = $LASTEXITCODE
  if ($code -eq 0) {
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "SUCCESS - instance is provisioning!" -ForegroundColor Green
    Write-Host $out
    Write-Host "Get its public IP once RUNNING (Console > Compute > Instances, or):"
    Write-Host "  oci compute instance list-vnics --instance-id <OCID> --query 'data[0].\""public-ip\""' --raw-output"
    Write-Host "============================================================" -ForegroundColor Green
    return 0
  }
  # FATAL — config/quota problems that retrying will never fix: stop and show.
  if ($out -match '(?i)NotAuthenticated|NotAuthorized|LimitExceeded|QuotaExceeded|CannotParseRequest|InvalidParameter|MissingParameter|InvalidImage|shape.*not (valid|compatible)') {
    Write-Host "FATAL error - stopping:" -ForegroundColor Red; Write-Host $out
    return 1
  }
  # Everything else (out of capacity, TooManyRequests, timeouts, 5xx) — retry.
  $msg = ($out | Select-String -Pattern '"(code|message)":\s*"[^"]*"' -AllMatches).Matches.Value -join '  '
  if ($msg) { Write-Host "  ($msg)" -ForegroundColor DarkGray }
  else      { Write-Host "  (transient error - retrying)" -ForegroundColor DarkGray }
  return 2
}

if ($Once) {
  $rc = Try-Launch
  if ($rc -eq 1) { exit 1 } else { exit 0 }
}

Write-Host "Hunting capacity - keep this window OPEN and the PC AWAKE. Ctrl+C to stop." -ForegroundColor Cyan
$n = 0
while ($true) {
  $n++
  Write-Host ("[{0}] attempt #{1}..." -f (Get-Date -Format HH:mm:ss), $n)
  $rc = Try-Launch
  if ($rc -eq 0) { break }
  if ($rc -eq 1) { exit 1 }   # real (non-capacity) error
  Start-Sleep -Seconds $SleepSec
}
