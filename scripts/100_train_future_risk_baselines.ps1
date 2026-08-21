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
$CudaAvailable = & $Python -c "import torch; print('1' if torch.cuda.is_available() else '0')"
if ($LASTEXITCODE -ne 0) { throw "Unable to probe PyTorch CUDA support." }
$Device = if ($CudaAvailable.Trim() -eq "1") { "cuda" } else { "cpu" }
if ($RequireGpu -and $Device -ne "cuda") {
  throw "GPU baseline training was required, but CUDA is unavailable."
}

$Cache = Join-Path $ProjectRoot "world_model_runs\pi_gwm_multistep_v12_charge_shared\transitions_congestion_v3_seed4200.pkl.gz"
$OutputRoot = Join-Path $ProjectRoot "world_model_runs\future_risk_baselines_v1"
& $Python (Join-Path $ProjectRoot "future_risk_baselines.py") `
  --transition-cache $Cache `
  --output-root $OutputRoot `
  --architectures mlp,gru,gnn `
  --seeds 42,43,44 `
  --epochs 40 `
  --batch-size 256 `
  --learning-rate 0.0003 `
  --weight-decay 0.0001 `
  --hidden-dim 96 `
  --future-risk-horizon 80 `
  --sequence-stride 2 `
  --split-seed 4200 `
  --device $Device
if ($LASTEXITCODE -ne 0) { throw "Future-risk baseline training failed." }
Write-Host "Future-risk baselines trained for three architectures and three seeds."
