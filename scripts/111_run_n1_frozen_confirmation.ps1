param(
    [string]$Python = "python",
    [switch]$RequireGpu
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$CudaAvailable = & $Python -c "import torch; print('1' if torch.cuda.is_available() else '0')"
if ($LASTEXITCODE -ne 0) { throw "Unable to probe PyTorch CUDA support." }
$Device = if ($CudaAvailable.Trim() -eq "1") { "cuda" } else { "cpu" }
if ($RequireGpu -and $Device -ne "cuda") { throw "GPU required but CUDA is unavailable." }

$Development = Join-Path $ProjectRoot "experiment_results\n1_arrival_v4_bounded_gate_development_v2"
$Output = Join-Path $ProjectRoot "experiment_results\n1_arrival_v4_frozen_confirmation_v1"
if (-not (Test-Path (Join-Path $Development "run_manifest.json"))) {
    throw "The passed bounded-gate development manifest is missing."
}
if (Test-Path (Join-Path $Output "summary.csv")) {
    throw "A completed frozen confirmation result already exists; refusing to overwrite it."
}

& $Python (Join-Path $ProjectRoot "run_multistep_decision_attribution.py") `
    --phase confirmation `
    --hours 1 `
    --env-seed-start 35001 `
    --env-seed-count 15 `
    --model-seeds 42,43,44 `
    --control-mode ensemble `
    --minimum-ensemble-agreement 2 `
    --planning-horizon 3 `
    --beam-width 8 `
    --risk-gate 0.75 `
    --override-mode evidence_gated `
    --capacity-mode baseline `
    --device $Device `
    --max-steps 2000 `
    --output-dir $Output
if ($LASTEXITCODE -ne 0) { throw "N1 frozen confirmation run failed." }

& $Python (Join-Path $ProjectRoot "analyze_n1_confirmation.py") `
    --result-dir $Output `
    --development-dir $Development `
    --bootstrap-replicates 10000 `
    --bootstrap-seed 38117
if ($LASTEXITCODE -ne 0) { throw "N1 frozen confirmation audit failed." }

Write-Host "N1 frozen confirmation finished: $Output"
