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
$Runner = Join-Path $ProjectRoot "compare_counterfactual_architectures_v150.py"
$OutputDir = Join-Path $ProjectRoot "experiment_results\v150_graph_vs_flat_development_seed16400"
$Cache = Join-Path $OutputDir "development_paired_samples_seed16400.pkl.gz"
$DeviceArgs = @("--device", "auto")
if ($RequireGpu) {
  $DeviceArgs += "--require-cuda"
}
$Checkpoints = @()
foreach ($Seed in @(42, 43, 44)) {
  $Graph = Join-Path $ProjectRoot "world_model_runs\pi_gwm_counterfactual_v141_seed${Seed}\physics_graph_world_model_counterfactual.pt"
  $Flat = Join-Path $ProjectRoot "world_model_runs\flat_mlp_counterfactual_v150_seed${Seed}\flat_counterfactual_baseline.pt"
  if (-not (Test-Path -LiteralPath $Graph)) { throw "Missing V14.1 checkpoint: ${Graph}" }
  if (-not (Test-Path -LiteralPath $Flat)) { throw "Missing V15.0 checkpoint: ${Flat}" }
  $Checkpoints += @("--graph-checkpoint", $Graph, "--flat-checkpoint", $Flat)
}

& $Python $Runner `
  --phase development `
  @Checkpoints `
  --output-dir $OutputDir `
  --diagnostic-cache $Cache `
  --batch-size 256 `
  @DeviceArgs
if ($LASTEXITCODE -ne 0) {
  throw "V15.0 development integrity check failed."
}

Write-Host "V15.0 development run finished. It is implementation evidence only: ${OutputDir}"

