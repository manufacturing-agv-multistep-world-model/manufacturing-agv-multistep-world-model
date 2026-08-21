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
$OneHour = Join-Path $ProjectRoot "experiment_results\n1_arrival_v4_energy_neutral_development_1h_v3"
$Output = Join-Path $ProjectRoot "experiment_results\n1_arrival_v4_energy_neutral_development_4h_v3"
if (-not (Test-Path (Join-Path $OneHour "run_manifest.json"))) { throw "The passed one-hour development manifest is missing." }
if (Test-Path (Join-Path $Output "summary.csv")) { throw "A completed energy-neutral 4 h development result already exists." }

& $Python (Join-Path $ProjectRoot "run_multistep_decision_attribution.py") `
    --phase development `
    --hours 4 `
    --env-seed-start 37001 `
    --env-seed-count 5 `
    --model-seeds 42,43,44 `
    --control-mode ensemble `
    --minimum-ensemble-agreement 2 `
    --planning-horizon 3 `
    --beam-width 8 `
    --risk-gate 0.75 `
    --override-mode energy_neutral_gated `
    --capacity-mode baseline `
    --device $Device `
    --max-steps 10000 `
    --output-dir $Output
if ($LASTEXITCODE -ne 0) { throw "N1 energy-neutral 4 h development run failed." }

& $Python (Join-Path $ProjectRoot "analyze_n1_energy_neutral_4h_development.py") `
    --result-dir $Output `
    --one-hour-dir $OneHour `
    --bootstrap-replicates 5000 `
    --bootstrap-seed 40117
if ($LASTEXITCODE -ne 0) { throw "N1 energy-neutral 4 h development audit failed." }
Write-Host "N1 energy-neutral 4 h development finished: $Output"
