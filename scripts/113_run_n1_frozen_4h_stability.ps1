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
$Confirmation = Join-Path $ProjectRoot "experiment_results\n1_arrival_v4_frozen_confirmation_v1"
$Output = Join-Path $ProjectRoot "experiment_results\n1_arrival_v4_long_horizon_4h_v1"
if (-not (Test-Path (Join-Path $Confirmation "run_manifest.json"))) { throw "Independent confirmation result is missing." }
if (Test-Path (Join-Path $Output "summary.csv")) { throw "A completed 4 h result already exists; refusing to overwrite it." }

& $Python (Join-Path $ProjectRoot "run_multistep_decision_attribution.py") `
    --phase confirmation `
    --hours 4 `
    --env-seed-start 36001 `
    --env-seed-count 10 `
    --model-seeds 42,43,44 `
    --control-mode ensemble `
    --minimum-ensemble-agreement 2 `
    --planning-horizon 3 `
    --beam-width 8 `
    --risk-gate 0.75 `
    --override-mode evidence_gated `
    --capacity-mode baseline `
    --device $Device `
    --max-steps 10000 `
    --output-dir $Output
if ($LASTEXITCODE -ne 0) { throw "N1 frozen 4 h run failed." }

& $Python (Join-Path $ProjectRoot "analyze_n1_long_horizon.py") `
    --result-dir $Output `
    --confirmation-dir $Confirmation `
    --hours 4 `
    --bootstrap-replicates 10000 `
    --bootstrap-seed 39117
if ($LASTEXITCODE -ne 0) { throw "N1 frozen 4 h audit failed." }
Write-Host "N1 frozen 4 h stability run finished: $Output"
