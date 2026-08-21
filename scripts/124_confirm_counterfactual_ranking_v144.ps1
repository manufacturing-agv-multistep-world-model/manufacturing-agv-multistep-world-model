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
$Diagnostic = Join-Path $ProjectRoot "diagnose_counterfactual_ranking_v144.py"
$OutputDir = Join-Path $ProjectRoot "experiment_results\world_model_counterfactual_v144_ranking_confirmation_v1"
$Cache = Join-Path $OutputDir "fresh_paired_samples_seed15400.pkl.gz"
$Log = Join-Path $ProjectRoot "v144_ranking_confirmation_console.log"
$DeviceArgs = @("--device", "auto")
if ($RequireGpu) {
  $DeviceArgs += "--require-cuda"
}
$Checkpoints = @()
foreach ($Seed in @(42, 43, 44)) {
  $Checkpoint = Join-Path $ProjectRoot "world_model_runs\pi_gwm_counterfactual_v141_seed${Seed}\physics_graph_world_model_counterfactual.pt"
  if (-not (Test-Path -LiteralPath $Checkpoint)) {
    throw "Missing frozen V14.1 checkpoint for seed ${Seed}: ${Checkpoint}"
  }
  $Checkpoints += @("--checkpoint", $Checkpoint)
}

$PythonArgs = @(
  $Diagnostic
) + $Checkpoints + @(
  "--output-dir", $OutputDir,
  "--diagnostic-cache", $Cache,
  "--episodes", "12",
  "--behavior-steps", "4000",
  "--warmup-steps", "1200",
  "--sample-stride", "80",
  "--candidates-per-state", "3",
  "--max-rollout-steps", "500",
  "--data-seed", "15400",
  "--bootstrap-replicates", "5000",
  "--bootstrap-seed", "15499",
  "--batch-size", "256"
) + $DeviceArgs

$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Python @PythonArgs 2>&1 | Tee-Object -FilePath $Log
$PythonExitCode = $LASTEXITCODE
$ErrorActionPreference = $PreviousErrorActionPreference
if ($PythonExitCode -ne 0) {
  Write-Error "V14.4 Python process exited with code ${PythonExitCode}. Full log: ${Log}"
  exit $PythonExitCode
}

Write-Host "V14.4 ranking confirmation finished: ${OutputDir}"
