# Overnight diverse-ensemble runner (Windows / PowerShell).
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_ensemble.ps1
#
# Trains 5 models sequentially (3x B3 + 2x B4, varied seeds), each saving its own
# best-val-F2 checkpoint, then runs the ensemble evaluation. A single model failure
# (e.g. a transient OOM) is logged and does not abort the night. Each run's stdout
# is teed to reports\train_<backbone>_seed<seed>.log.

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path "reports" | Out-Null

# backbone, batch_size, seed
$models = @(
    @{ bb = "tf_efficientnet_b3"; batch = 32; seed = 42 },
    @{ bb = "tf_efficientnet_b3"; batch = 32; seed = 43 },
    @{ bb = "tf_efficientnet_b3"; batch = 32; seed = 44 },
    @{ bb = "tf_efficientnet_b4"; batch = 16; seed = 42 },
    @{ bb = "tf_efficientnet_b4"; batch = 16; seed = 43 }
)

$epochs = 30
$start = Get-Date
Write-Host "=== Ensemble training started $start ($($models.Count) models, $epochs epochs each) ==="

foreach ($m in $models) {
    $tag = "$($m.bb)_seed$($m.seed)"
    $log = "reports\train_$tag.log"
    Write-Host "`n--- Training $tag (batch=$($m.batch)) -> $log ---"
    try {
        py scripts\02_train.py --config configs\baseline.yaml `
            --backbone $m.bb --batch-size $m.batch --seed $m.seed --epochs $epochs `
            2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "$tag exited with code $LASTEXITCODE (continuing to next model)."
        }
    } catch {
        Write-Warning "$tag failed: $_ (continuing to next model)."
    }
}

Write-Host "`n=== Training done. Running ensemble evaluation ==="
try {
    py scripts\05_ensemble_eval.py --config configs\baseline.yaml `
        2>&1 | Tee-Object -FilePath "reports\ensemble_eval.log"
} catch {
    Write-Warning "Ensemble eval failed: $_"
}

$end = Get-Date
Write-Host "`n=== All done $end (elapsed $([math]::Round(($end - $start).TotalHours, 2)) h) ==="
Write-Host "Checkpoints: checkpoints\*_seed*_best.pt | Report: reports\ensemble_test_metrics.csv"
