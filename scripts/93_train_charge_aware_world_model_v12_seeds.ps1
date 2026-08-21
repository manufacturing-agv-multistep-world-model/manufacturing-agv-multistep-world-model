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
  Write-Host "CUDA detected. Charge-aware V12 training will use GPU mixed precision."
} else {
  if ($RequireGpu) {
    throw "GPU training was required, but this Python environment uses CPU-only PyTorch."
  }
  Write-Warning "CUDA is unavailable. V12 training will run on CPU and may be slow."
}

$Trainer = Join-Path $ProjectRoot "train_world_model_multistep.py"
$Cache = Join-Path $ProjectRoot "world_model_runs\pi_gwm_multistep_v12_charge_shared\transitions_congestion_v3_seed4200.pkl.gz"

foreach ($Seed in @(42, 43, 44)) {
  $OutputDir = Join-Path $ProjectRoot "world_model_runs\pi_gwm_multistep_v12_charge_seed${Seed}"
  & $Python $Trainer `
    --model-variant v12 `
    --episodes 12 `
    --max-steps 4000 `
    --epochs 80 `
    --batch-size 256 `
    --learning-rate 0.0003 `
    --weight-decay 0.0001 `
    --physics-weight 0.50 `
    --rollout-discount 0.95 `
    --training-horizon 10 `
    --sequence-stride 2 `
    --teacher-forcing-start 0.90 `
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
    --seed $Seed `
    --output-dir $OutputDir `
    @DeviceArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Charge-aware V12 training failed for seed ${Seed}."
  }
}

Write-Host "Charge-aware V12 training finished for seeds 42, 43, and 44."
