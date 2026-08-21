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
$Runner = Join-Path $ProjectRoot "compare_paired_vs_absolute_v151.py"
$OutputDir = Join-Path $ProjectRoot "experiment_results\v151_paired_vs_absolute_confirmation_seed18400"
$Cache = Join-Path $OutputDir "fresh_paired_samples_seed18400.pkl.gz"
$Log = Join-Path $ProjectRoot "v151_paired_formulation_confirmation_console.log"
$DeviceArgs = @("--device", "auto")
if ($RequireGpu) {
  $DeviceArgs += "--require-cuda"
}
$Checkpoints = @()
foreach ($Seed in @(42, 43, 44)) {
  $Paired = Join-Path $ProjectRoot "world_model_runs\pi_gwm_counterfactual_v141_seed${Seed}\physics_graph_world_model_counterfactual.pt"
  $Absolute = Join-Path $ProjectRoot "world_model_runs\pi_gwm_absolute_v151_seed${Seed}\absolute_outcome_graph_baseline.pt"
  if (-not (Test-Path -LiteralPath $Paired)) { throw "Missing paired checkpoint: ${Paired}" }
  if (-not (Test-Path -LiteralPath $Absolute)) { throw "Missing absolute checkpoint: ${Absolute}" }
  $Checkpoints += @("--paired-checkpoint", $Paired, "--absolute-checkpoint", $Absolute)
}

$PythonArgs = @($Runner) + $Checkpoints + @(
  "--output-dir", $OutputDir,
  "--diagnostic-cache", $Cache,
  "--batch-size", "256",
  "--parallel-episodes", "$CpuThreads"
) + $DeviceArgs

$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Python @PythonArgs 2>&1 | Tee-Object -FilePath $Log
$PythonExitCode = $LASTEXITCODE
$ErrorActionPreference = $PreviousErrorActionPreference
if ($PythonExitCode -ne 0) {
  Write-Error "V15.1 confirmation exited with code ${PythonExitCode}. Full log: ${Log}"
  exit $PythonExitCode
}

Write-Host "V15.1 paired-formulation confirmation finished: ${OutputDir}"
