$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = "python"

& $Python (Join-Path $ProjectRoot "run_anylogic_python_reference.py") `
  --seeds "41001,41002,41003,41004,41005,41006,41007,41008,41009,41010" `
  --horizons "1,4,8" `
  --scenarios "steady,rush" `
  --output "paper_outputs/anylogic_validation/python_reference_runs.csv"

if ($LASTEXITCODE -ne 0) {
  throw "Matched Python reference experiment failed with exit code $LASTEXITCODE."
}

Write-Host "Matched Python reference experiment completed."
