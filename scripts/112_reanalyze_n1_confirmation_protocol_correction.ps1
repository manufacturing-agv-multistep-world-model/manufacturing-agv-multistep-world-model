param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Development = Join-Path $ProjectRoot "experiment_results\n1_arrival_v4_bounded_gate_development_v2"
$Output = Join-Path $ProjectRoot "experiment_results\n1_arrival_v4_frozen_confirmation_v1"
$Original = Join-Path $Output "confirmation_audit.md"
$Archived = Join-Path $Output "confirmation_audit_original_v1.md"

if (-not (Test-Path (Join-Path $Output "summary.csv"))) {
    throw "The completed frozen confirmation data are missing."
}
if (-not (Test-Path (Join-Path $Development "run_manifest.json"))) {
    throw "The bounded-gate development manifest is missing."
}
if ((Test-Path $Original) -and -not (Test-Path $Archived)) {
    Copy-Item -LiteralPath $Original -Destination $Archived
}

& $Python (Join-Path $ProjectRoot "analyze_n1_confirmation.py") `
    --result-dir $Output `
    --development-dir $Development `
    --bootstrap-replicates 10000 `
    --bootstrap-seed 38117 `
    --report-name confirmation_audit_corrected_v2.md
if ($LASTEXITCODE -ne 0) { throw "Corrected N1 confirmation analysis failed." }

Write-Host "Corrected analysis finished without rerunning simulation: $Output"
