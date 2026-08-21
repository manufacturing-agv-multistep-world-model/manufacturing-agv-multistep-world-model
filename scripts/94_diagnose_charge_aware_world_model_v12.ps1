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
  throw "GPU diagnostics were required, but CUDA is unavailable."
}

foreach ($Seed in @(42, 43, 44)) {
  $ModelPath = Join-Path $ProjectRoot "world_model_runs\pi_gwm_multistep_v12_charge_seed${Seed}\physics_graph_world_model_multistep.pt"
  $OutputDir = Join-Path $ProjectRoot "experiment_results\world_model_multistep_v12_charge_seed${Seed}"
  if (-not (Test-Path -LiteralPath $ModelPath)) {
    throw "Missing charge-aware V12 checkpoint for seed ${Seed}: ${ModelPath}"
  }
  & $Python (Join-Path $ProjectRoot "diagnose_world_model_multistep.py") `
    --model-path $ModelPath `
    --output-dir $OutputDir `
    --horizons 1,3,5,10 `
    --episodes 4 `
    --max-steps 4000 `
    --batch-size 128 `
    --sequence-stride 10 `
    --exploration-rate 0.35 `
    --agv-count 3 `
    --scenario rush `
    --capacity-mode stress `
    --seed 19412 `
    --device $Device
  if ($LASTEXITCODE -ne 0) {
    throw "Charge-aware V12 diagnostics failed for seed ${Seed}."
  }
}

$Report = Join-Path $ProjectRoot "experiment_results\v12_charge_open_loop_diagnostic_audit.md"
& $Python (Join-Path $ProjectRoot "analyze_v12_open_loop_diagnostics.py") `
  --results-root (Join-Path $ProjectRoot "experiment_results") `
  --report $Report
if ($LASTEXITCODE -ne 0) {
  throw "Charge-aware V12 diagnostic audit failed."
}

Write-Host "Charge-aware V12 diagnostics finished for all model seeds."
