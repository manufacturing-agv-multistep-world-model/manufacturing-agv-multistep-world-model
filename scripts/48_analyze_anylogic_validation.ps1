$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = "python"

& $Python (Join-Path $ProjectRoot "analyze_anylogic_validation.py") `
  --anylogic-csv "AGV_DT_AnyLogic_Validation/Manufacturing_AGV_DT_Validation/anylogic_validation_results.csv" `
  --python-reference "paper_outputs/anylogic_validation/python_reference_runs.csv" `
  --required-seeds "1,2,3" `
  --horizons "1,4,8" `
  --scenarios "steady,rush" `
  --output-dir "paper_outputs/anylogic_validation/final"

if ($LASTEXITCODE -ne 0) {
  throw "AnyLogic validation analysis failed with exit code $LASTEXITCODE."
}

Write-Host "AnyLogic validation audit and figures completed."
