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
$OutputDir = Join-Path $ProjectRoot "experiment_results\v150_graph_vs_flat_confirmation_seed17400"
$Cache = Join-Path $OutputDir "untouched_paired_samples_seed17400.pkl.gz"
$Log = Join-Path $ProjectRoot "v150_architecture_confirmation_console.log"
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

$PythonArgs = @(
  $Runner,
  "--phase", "confirmation"
) + $Checkpoints + @(
  "--output-dir", $OutputDir,
  "--diagnostic-cache", $Cache,
  "--batch-size", "256"
) + $DeviceArgs

$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Python @PythonArgs 2>&1 | Tee-Object -FilePath $Log
$PythonExitCode = $LASTEXITCODE
$ErrorActionPreference = $PreviousErrorActionPreference
if ($PythonExitCode -ne 0) {
  Write-Error "V15.0 confirmation process exited with code ${PythonExitCode}. Full log: ${Log}"
  exit $PythonExitCode
}

Write-Host "V15.0 frozen architecture confirmation finished: ${OutputDir}"

