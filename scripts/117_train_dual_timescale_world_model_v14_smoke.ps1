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

$CudaProbe = & $Python -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU'); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device')"
if ($LASTEXITCODE -ne 0) {
  throw "Unable to import PyTorch. Activate the intended Python environment first."
}
$Device = $CudaProbe[0].Trim().ToLowerInvariant()
Write-Host "PyTorch probe: $($CudaProbe -join ' | ')"
$DeviceArgs = @("--device", $Device)
if ($Device -eq "cuda") {
  $DeviceArgs += @("--amp", "--require-cuda")
} elseif ($RequireGpu) {
  throw "GPU smoke training was required, but CUDA is unavailable."
}

$Trainer = Join-Path $ProjectRoot "train_world_model_multistep.py"
$Cache = Join-Path $ProjectRoot "world_model_runs\pi_gwm_multistep_v14_dual_timescale_shared\transitions_schema_v4_long_seed4200.pkl.gz"
$InitCheckpoint = Join-Path $ProjectRoot "world_model_runs\pi_gwm_multistep_v13_multiscale_v2_seed42\physics_graph_world_model_multistep.pt"
$OutputDir = Join-Path $ProjectRoot "world_model_runs\pi_gwm_multistep_v14_dual_timescale_smoke_seed42"
foreach ($RequiredPath in @($InitCheckpoint)) {
  if (-not (Test-Path -LiteralPath $RequiredPath)) {
    throw "Missing V14 prerequisite: ${RequiredPath}"
  }
}

& $Python $Trainer `
  --model-variant v14 `
  --future-risk-horizon 80 `
  --future-terminal-horizon 80 `
  --init-checkpoint $InitCheckpoint `
  --freeze-v14-backbone `
  --episodes 12 `
  --max-steps 4000 `
  --epochs 1 `
  --batch-size 512 `
  --learning-rate 0.0003 `
  --weight-decay 0.0001 `
  --physics-weight 0.50 `
  --rollout-discount 0.95 `
  --training-horizon 3 `
  --sequence-stride 10 `
  --teacher-forcing-start 0.00 `
  --teacher-forcing-end 0.00 `
  --exploration-rate 0.35 `
  --hidden-dim 96 `
  --planning-horizon 5 `
  --beam-width 8 `
  --planning-discount 0.95 `
  --agv-count 3 `
  --env-variant full `
  --reward-mode hybrid `
  --scenario rush `
  --dispatch-rule dt_aware `
  --capacity-mode stress `
  --data-seed 4200 `
  --split-seed 4200 `
  --transition-cache $Cache `
  --cpu-threads $CpuThreads `
  --low-priority `
  --seed 42 `
  --output-dir $OutputDir `
  @DeviceArgs
if ($LASTEXITCODE -ne 0) {
  throw "V14 dual-timescale smoke training failed."
}

Write-Host "V14 smoke training passed: ${OutputDir}"
