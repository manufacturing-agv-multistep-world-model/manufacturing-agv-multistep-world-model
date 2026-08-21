param(
  [string]$Python = "python",
  [switch]$RequireGpu,
  [ValidateRange(1, 64)]
  [int]$CpuThreads = 4
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:OMP_NUM_THREADS = "$CpuThreads"
$env:MKL_NUM_THREADS = "$CpuThreads"
$env:OPENBLAS_NUM_THREADS = "$CpuThreads"
$env:NUMEXPR_NUM_THREADS = "$CpuThreads"

$CudaAvailable = & $Python -c "import torch; print('1' if torch.cuda.is_available() else '0')"
if ($LASTEXITCODE -ne 0) {
  throw "Unable to probe PyTorch CUDA support."
}
$Device = if ($CudaAvailable.Trim() -eq "1") { "cuda" } else { "cpu" }
if ($RequireGpu -and $Device -ne "cuda") {
  throw "GPU diagnostics were required, but CUDA is unavailable."
}
$TestCache = Join-Path $ProjectRoot "experiment_results\v14_dual_timescale_fresh_seed28313_transitions.pkl.gz"

foreach ($Seed in @(42, 43, 44)) {
  $ModelPath = Join-Path $ProjectRoot "world_model_runs\pi_gwm_multistep_v14_dual_timescale_v1_seed${Seed}\physics_graph_world_model_multistep.pt"
  $OutputDir = Join-Path $ProjectRoot "experiment_results\world_model_multistep_v14_dual_timescale_v1_seed${Seed}"
  if (-not (Test-Path -LiteralPath $ModelPath)) {
    throw "Missing V14 checkpoint for seed ${Seed}: ${ModelPath}"
  }
  & $Python (Join-Path $ProjectRoot "diagnose_world_model_multistep.py") `
    --model-path $ModelPath `
    --output-dir $OutputDir `
    --horizons 1,5,10 `
    --episodes 8 `
    --max-steps 4000 `
    --batch-size 256 `
    --sequence-stride 10 `
    --exploration-rate 0.35 `
    --agv-count 3 `
    --scenario rush `
    --capacity-mode stress `
    --seed 28313 `
    --transition-cache $TestCache `
    --device $Device
  if ($LASTEXITCODE -ne 0) {
    throw "V14 diagnostics failed for seed ${Seed}."
  }
}

$Report = Join-Path $ProjectRoot "experiment_results\v14_dual_timescale_open_loop_audit.md"
& $Python (Join-Path $ProjectRoot "analyze_v14_dual_timescale_diagnostics.py") `
  --results-root (Join-Path $ProjectRoot "experiment_results") `
  --report $Report `
  --bootstrap-samples 2000 `
  --bootstrap-seed 314159
if ($LASTEXITCODE -ne 0) {
  throw "V14 diagnostic audit failed."
}

Write-Host "V14 dual-timescale independent open-loop diagnostics finished."
