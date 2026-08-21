param(
  [string]$Python = "python",
  [switch]$RequireGpu
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$CudaAvailable = & $Python -c "import torch; print('1' if torch.cuda.is_available() else '0')"
if ($LASTEXITCODE -ne 0) {
  throw "Unable to probe PyTorch CUDA support."
}
$Device = if ($CudaAvailable.Trim() -eq "1") { "cuda" } else { "cpu" }
if ($RequireGpu -and $Device -ne "cuda") {
  throw "GPU calibration was required, but CUDA is unavailable."
}

$OutputDir = Join-Path $ProjectRoot "experiment_results\v13_multiscale_v2_threshold_calibration"
& $Python (Join-Path $ProjectRoot "calibrate_v13_charge_thresholds.py") `
  --output-dir $OutputDir `
  --episodes 6 `
  --max-steps 4000 `
  --sequence-stride 10 `
  --batch-size 128 `
  --seed 22313 `
  --device $Device
if ($LASTEXITCODE -ne 0) {
  throw "V13 threshold calibration failed."
}

Write-Host "V13 threshold calibration finished. Do not run confirmation until reviewed."
