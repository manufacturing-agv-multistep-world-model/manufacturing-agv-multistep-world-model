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
$DeviceArgs = @("--device", "auto")
if ($RequireGpu) { $DeviceArgs += "--require-cuda" }

$Studies = @(
  @{
    Name = "steady_stress_four_hour"
    Output = "experiment_results\v149s_steady_stress_4h_development_v1"
    Log = "v149s_steady_stress_4h_development_console.log"
  },
  @{
    Name = "rush_baseline_four_hour"
    Output = "experiment_results\v149d_rush_baseline_4h_development_v1"
    Log = "v149d_rush_baseline_4h_development_console.log"
  }
)

foreach ($Study in $Studies) {
  $OutputDir = Join-Path $ProjectRoot $Study.Output
  $Log = Join-Path $ProjectRoot $Study.Log
  $PreviousErrorActionPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  & $Python $Runner --study $Study.Name --output-dir $OutputDir --max-steps 8000 @DeviceArgs 2>&1 | Tee-Object -FilePath $Log
  $PythonExitCode = $LASTEXITCODE
  $ErrorActionPreference = $PreviousErrorActionPreference
  if ($PythonExitCode -ne 0) {
    Write-Error "V14.9 boundary study $($Study.Name) exited with code ${PythonExitCode}. Full log: ${Log}"
    exit $PythonExitCode
  }
}

Write-Host "V14.9 operating-boundary development finished for both intermediate regimes."
