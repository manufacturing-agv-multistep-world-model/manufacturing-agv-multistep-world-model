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
$Output = Join-Path $ProjectRoot "experiment_results\n1_arrival_v4_development_baseline_v1"

& $Python (Join-Path $ProjectRoot "run_multistep_decision_attribution.py") `
    --phase development `
    --hours 1 `
    --env-seed-start 33001 `
    --env-seed-count 5 `
    --model-seeds 42,43,44 `
    --control-mode ensemble `
    --minimum-ensemble-agreement 2 `
    --planning-horizon 3 `
    --beam-width 8 `
    --risk-gate 0.75 `
    --override-mode safe_argmax `
    --capacity-mode baseline `
    --device $Device `
    --max-steps 2000 `
    --output-dir $Output
if ($LASTEXITCODE -ne 0) { throw "N1 arrival-v4 development run failed." }

& $Python (Join-Path $ProjectRoot "analyze_multistep_decision_attribution.py") --result-dir $Output
if ($LASTEXITCODE -ne 0) { throw "N1 arrival-v4 development audit failed." }

Write-Host "N1 arrival-v4 baseline development run finished: $Output"
