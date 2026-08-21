param(
  [string]$Python = "python",
  [switch]$RequireGpu,
  [ValidateRange(1, 64)]
  [int]$CpuThreads = 4
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:OMP_NUM_THREADS = "$CpuThreads"
$env:MKL_NUM_THREADS = "$CpuThreads"
$env:OPENBLAS_NUM_THREADS = "$CpuThreads"
$Diagnostic = Join-Path $ProjectRoot "diagnose_counterfactual_world_model_v141.py"
$OutputDir = Join-Path $ProjectRoot "experiment_results\world_model_counterfactual_v141_confirmation_v1"
$Cache = Join-Path $ProjectRoot "experiment_results\world_model_counterfactual_v141_confirmation_v1\fresh_paired_samples_seed15100.pkl.gz"
$DeviceArgs = @("--device", "auto")
if ($RequireGpu) {
  $DeviceArgs += "--require-cuda"
}
$Checkpoints = @()
foreach ($Seed in @(42, 43, 44)) {
  $Checkpoint = Join-Path $ProjectRoot "world_model_runs\pi_gwm_counterfactual_v141_seed${Seed}\physics_graph_world_model_counterfactual.pt"
  if (-not (Test-Path -LiteralPath $Checkpoint)) {
    throw "Missing V14.1 checkpoint for seed ${Seed}: ${Checkpoint}"
  }
  $Checkpoints += @("--checkpoint", $Checkpoint)
}

& $Python $Diagnostic `
  @Checkpoints `
  --output-dir $OutputDir `
  --diagnostic-cache $Cache `
  --episodes 8 `
  --behavior-steps 4000 `
  --warmup-steps 1200 `
  --sample-stride 80 `
  --candidates-per-state 3 `
  --max-rollout-steps 500 `
  --data-seed 15100 `
  --bootstrap-replicates 2000 `
  --bootstrap-seed 15199 `
  --batch-size 256 `
  @DeviceArgs
if ($LASTEXITCODE -ne 0) {
  throw "V14.1 independent counterfactual confirmation failed."
}

Write-Host "V14.1 independent counterfactual confirmation finished: ${OutputDir}"
