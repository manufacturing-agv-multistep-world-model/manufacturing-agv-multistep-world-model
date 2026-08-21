param(
    [string]$Python = "python",
    [switch]$RequireGpu
)

$ErrorActionPreference = "Stop"
throw "This development entry point is retired with the pre-v4 models. Use scripts/109_run_n1_arrival_v4_development.ps1."
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$CudaAvailable = & $Python -c "import torch; print('1' if torch.cuda.is_available() else '0')"
if ($LASTEXITCODE -ne 0) { throw "Unable to probe PyTorch CUDA support." }
$Device = if ($CudaAvailable.Trim() -eq "1") { "cuda" } else { "cpu" }
if ($RequireGpu -and $Device -ne "cuda") { throw "GPU required but CUDA is unavailable." }
$Output = Join-Path $ProjectRoot "experiment_results\n1_decision_attribution_development_v3_ensemble"

& $Python (Join-Path $ProjectRoot "run_multistep_decision_attribution.py") `
    --phase development `
    --hours 1 `
    --env-seed-start 31001 `
    --env-seed-count 5 `
    --model-seeds 42,43,44 `
    --control-mode ensemble `
    --minimum-ensemble-agreement 2 `
    --planning-horizon 3 `
    --beam-width 8 `
    --risk-gate 0.75 `
    --override-mode safe_argmax `
    --capacity-mode stress `
    --device $Device `
    --max-steps 2000 `
    --output-dir $Output
if ($LASTEXITCODE -ne 0) { throw "N1 attribution development run failed." }

& $Python (Join-Path $ProjectRoot "analyze_multistep_decision_attribution.py") --result-dir $Output
if ($LASTEXITCODE -ne 0) { throw "N1 attribution development audit failed." }

Write-Host "N1 development run finished: $Output"
