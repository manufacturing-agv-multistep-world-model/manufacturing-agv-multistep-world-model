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
  throw "GPU confirmation was required, but CUDA is unavailable."
}

$CalibrationPath = Join-Path $ProjectRoot "experiment_results\v13_multiscale_v2_threshold_calibration\calibrated_thresholds.json"
if (-not (Test-Path -LiteralPath $CalibrationPath)) {
  throw "Missing frozen calibration file: ${CalibrationPath}"
}
$Calibration = Get-Content -Raw -LiteralPath $CalibrationPath | ConvertFrom-Json
if ([int]$Calibration.calibration_seed -ne 22313) {
  throw "Unexpected calibration seed; confirmation is locked to calibration seed 22313."
}

foreach ($Seed in @(42, 43, 44)) {
  $ModelPath = Join-Path $ProjectRoot "world_model_runs\pi_gwm_multistep_v13_multiscale_v2_seed${Seed}\physics_graph_world_model_multistep.pt"
  $OutputDir = Join-Path $ProjectRoot "experiment_results\world_model_multistep_v13_v2_confirm_model_seed${Seed}"
  $ModelProperty = $Calibration.models.PSObject.Properties["$Seed"]
  if ($null -eq $ModelProperty) {
    throw "Calibration threshold is missing for model seed ${Seed}."
  }
  $Threshold = [double]$ModelProperty.Value.threshold
  Write-Host "Confirming model seed ${Seed} with frozen threshold ${Threshold}."
  & $Python (Join-Path $ProjectRoot "diagnose_world_model_multistep.py") `
    --model-path $ModelPath `
    --output-dir $OutputDir `
    --horizons 1,3,5,10 `
    --episodes 8 `
    --max-steps 4000 `
    --batch-size 128 `
    --sequence-stride 10 `
    --exploration-rate 0.35 `
    --agv-count 3 `
    --scenario rush `
    --capacity-mode stress `
    --seed 23313 `
    --future-risk-threshold $Threshold `
    --device $Device
  if ($LASTEXITCODE -ne 0) {
    throw "Calibrated V13 confirmation failed for model seed ${Seed}."
  }
}

$Report = Join-Path $ProjectRoot "experiment_results\v13_multiscale_v2_calibrated_confirmation_audit.md"
& $Python (Join-Path $ProjectRoot "analyze_v13_calibrated_confirmation.py") `
  --results-root (Join-Path $ProjectRoot "experiment_results") `
  --calibration $CalibrationPath `
  --report $Report
if ($LASTEXITCODE -ne 0) {
  throw "Calibrated V13 confirmation audit failed."
}

Write-Host "Calibrated V13 independent confirmation finished."
