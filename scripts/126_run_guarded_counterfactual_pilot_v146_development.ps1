param(
  [string]$Python = "python",
  [switch]$RequireGpu,
  [ValidateRange(1, 32)]
  [int]$CpuThreads = 4
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:OMP_NUM_THREADS = "$CpuThreads"
$env:MKL_NUM_THREADS = "$CpuThreads"
$env:OPENBLAS_NUM_THREADS = "$CpuThreads"
$Runner = Join-Path $ProjectRoot "run_guarded_counterfactual_pilot_v146.py"
$OutputDir = Join-Path $ProjectRoot "experiment_results\v146_guarded_closed_loop_development_v1"
$Log = Join-Path $ProjectRoot "v146_guarded_closed_loop_development_console.log"
$DeviceArgs = @("--device", "auto")
if ($RequireGpu) {
  $DeviceArgs += "--require-cuda"
}
$PythonArgs = @(
  $Runner,
  "--phase", "development",
  "--hours", "1",
  "--env-seed-start", "15801",
  "--env-seed-count", "5",
  "--max-steps", "2200",
  "--cooldown-sec", "60",
  "--maximum-overrides", "12",
  "--output-dir", $OutputDir
) + $DeviceArgs

$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Python @PythonArgs 2>&1 | Tee-Object -FilePath $Log
$PythonExitCode = $LASTEXITCODE
$ErrorActionPreference = $PreviousErrorActionPreference
if ($PythonExitCode -ne 0) {
  Write-Error "V14.6 development process exited with code ${PythonExitCode}. Full log: ${Log}"
  exit $PythonExitCode
}

Write-Host "V14.6 guarded closed-loop development finished: ${OutputDir}"
