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
$Runner = Join-Path $ProjectRoot "run_horizon_isolated_pilot_v147.py"
$OutputDir = Join-Path $ProjectRoot "experiment_results\v149d_rush_baseline_4h_development_v1"
$Log = Join-Path $ProjectRoot "v149d_rush_baseline_4h_development_console.log"
$DeviceArgs = @("--device", "auto")
if ($RequireGpu) { $DeviceArgs += "--require-cuda" }

$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Python $Runner --study rush_baseline_four_hour --output-dir $OutputDir --max-steps 8000 @DeviceArgs 2>&1 | Tee-Object -FilePath $Log
$PythonExitCode = $LASTEXITCODE
$ErrorActionPreference = $PreviousErrorActionPreference
if ($PythonExitCode -ne 0) {
  Write-Error "V14.9-D rush-arrival baseline-capacity development exited with code ${PythonExitCode}. Full log: ${Log}"
  exit $PythonExitCode
}
Write-Host "V14.9-D rush-arrival baseline-capacity development finished: ${OutputDir}"
