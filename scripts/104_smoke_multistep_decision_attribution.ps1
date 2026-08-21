param(
    [string]$Python = "python",
    [switch]$RequireGpu
)

$ErrorActionPreference = "Stop"
throw "This smoke entry point is retired with the pre-v4 models. Use scripts/108_smoke_n1_arrival_v4_baseline.ps1."
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$CudaAvailable = & $Python -c "import torch; print('1' if torch.cuda.is_available() else '0')"
if ($LASTEXITCODE -ne 0) { throw "Unable to probe PyTorch CUDA support." }
$Device = if ($CudaAvailable.Trim() -eq "1") { "cuda" } else { "cpu" }
if ($RequireGpu -and $Device -ne "cuda") { throw "GPU required but CUDA is unavailable." }
$Output = Join-Path $ProjectRoot "experiment_results\n1_decision_attribution_smoke_v3_ensemble"

& $Python (Join-Path $ProjectRoot "run_multistep_decision_attribution.py") `
    --phase smoke `
    --hours 0.1 `
    --env-seed-start 30001 `
    --env-seed-count 1 `
    --model-seeds 42,43,44 `
    --control-mode ensemble `
    --minimum-ensemble-agreement 2 `
    --planning-horizon 3 `
    --beam-width 8 `
    --risk-gate 0.75 `
    --override-mode safe_argmax `
    --capacity-mode stress `
    --device $Device `
    --max-steps 400 `
    --output-dir $Output
if ($LASTEXITCODE -ne 0) { throw "N1 attribution smoke run failed." }

& $Python (Join-Path $ProjectRoot "analyze_multistep_decision_attribution.py") --result-dir $Output
if ($LASTEXITCODE -ne 0) { throw "N1 attribution smoke audit failed." }

Write-Host "N1 short attribution smoke run finished: $Output"
