param(
  [string]$Python = "python",
  [switch]$RequireGpu,
  [ValidateRange(1, 64)]
  [int]$CpuThreads = 12,
  [ValidateRange(1, 12)]
  [int]$ParallelEpisodes = 6
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ThreadsPerWorker = [Math]::Max(1, [Math]::Floor($CpuThreads / $ParallelEpisodes))
$env:OMP_NUM_THREADS = "$ThreadsPerWorker"
$env:MKL_NUM_THREADS = "$ThreadsPerWorker"
$env:OPENBLAS_NUM_THREADS = "$ThreadsPerWorker"
$Diagnostic = Join-Path $ProjectRoot "diagnose_counterfactual_shadow_v145.py"
$OutputDir = Join-Path $ProjectRoot "experiment_results\world_model_counterfactual_v145_shadow_confirmation_parallel_v2"
$Cache = Join-Path $OutputDir "fresh_shadow_samples_seed15600_parallel_v2.pkl.gz"
$Log = Join-Path $ProjectRoot "v145_shadow_confirmation_console.log"
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
$PythonArgs = @($Diagnostic) + $Checkpoints + @(
  "--output-dir", $OutputDir,
  "--diagnostic-cache", $Cache,
  "--episodes", "12",
  "--behavior-steps", "4000",
  "--warmup-steps", "1200",
  "--sample-stride", "80",
  "--candidates-per-state", "3",
  "--max-rollout-steps", "500",
  "--data-seed", "15600",
  "--bootstrap-replicates", "5000",
  "--bootstrap-seed", "15699",
  "--batch-size", "256",
  "--parallel-episodes", "$ParallelEpisodes"
) + $DeviceArgs

$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Python @PythonArgs 2>&1 | Tee-Object -FilePath $Log
$PythonExitCode = $LASTEXITCODE
$ErrorActionPreference = $PreviousErrorActionPreference
if ($PythonExitCode -ne 0) {
  Write-Error "V14.5 Python process exited with code ${PythonExitCode}. Full log: ${Log}"
  exit $PythonExitCode
}

Write-Host "V14.5 shadow confirmation finished: ${OutputDir}"
