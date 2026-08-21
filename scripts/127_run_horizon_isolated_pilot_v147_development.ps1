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
$OutputDir = Join-Path $ProjectRoot "experiment_results\v147_horizon_isolated_development_v1"
$Log = Join-Path $ProjectRoot "v147_horizon_isolated_development_console.log"
$DeviceArgs = @("--device", "auto")
if ($RequireGpu) { $DeviceArgs += "--require-cuda" }

$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Python $Runner --output-dir $OutputDir --max-steps 2200 @DeviceArgs 2>&1 | Tee-Object -FilePath $Log
$PythonExitCode = $LASTEXITCODE
$ErrorActionPreference = $PreviousErrorActionPreference
if ($PythonExitCode -ne 0) {
  Write-Error "V14.7 development process exited with code ${PythonExitCode}. Full log: ${Log}"
  exit $PythonExitCode
}
Write-Host "V14.7 horizon-isolated development finished: ${OutputDir}"
