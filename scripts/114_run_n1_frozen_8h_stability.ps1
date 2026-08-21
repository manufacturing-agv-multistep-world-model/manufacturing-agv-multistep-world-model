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
$FourHour = Join-Path $ProjectRoot "experiment_results\n1_arrival_v4_long_horizon_4h_v1"
$Output = Join-Path $ProjectRoot "experiment_results\n1_arrival_v4_long_horizon_8h_v1"
$FourHourStatus = Join-Path $FourHour "long_horizon_status.json"
if (-not (Test-Path $FourHourStatus)) { throw "The audited 4 h stability result is missing." }
if (-not (Get-Content -LiteralPath $FourHourStatus -Raw | ConvertFrom-Json).passed) { throw "The 4 h stability gate failed; 8 h execution is not permitted." }
if (Test-Path (Join-Path $Output "summary.csv")) { throw "A completed 8 h result already exists; refusing to overwrite it." }

& $Python (Join-Path $ProjectRoot "run_multistep_decision_attribution.py") `
    --phase confirmation `
    --hours 8 `
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
    --max-steps 20000 `
    --output-dir $Output
if ($LASTEXITCODE -ne 0) { throw "N1 frozen 8 h run failed." }

& $Python (Join-Path $ProjectRoot "analyze_n1_long_horizon.py") `
    --result-dir $Output `
    --confirmation-dir $Confirmation `
    --hours 8 `
    --bootstrap-replicates 10000 `
    --bootstrap-seed 39118
if ($LASTEXITCODE -ne 0) { throw "N1 frozen 8 h audit failed." }
Write-Host "N1 frozen 8 h stability run finished: $Output"
