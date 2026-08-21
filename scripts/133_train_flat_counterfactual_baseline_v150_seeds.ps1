param(
  [string]$Python = "python",
  [switch]$RequireGpu,
  [ValidateRange(1, 64)]
  [int]$CpuThreads = 4
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Trainer = Join-Path $ProjectRoot "train_flat_counterfactual_baseline_v150.py"
$Cache = Join-Path $ProjectRoot "world_model_runs\pi_gwm_counterfactual_v141_shared\paired_material_samples_v2_seed14100.pkl.gz"
if (-not (Test-Path -LiteralPath $Cache)) {
  throw "Missing frozen V14.1 paired training cache: ${Cache}"
}
$DeviceArgs = @("--device", "auto")
if ($RequireGpu) {
  $DeviceArgs += @("--require-cuda", "--amp")
}

foreach ($Seed in @(42, 43, 44)) {
  $OutputDir = Join-Path $ProjectRoot "world_model_runs\flat_mlp_counterfactual_v150_seed${Seed}"
  & $Python $Trainer `
    --output-dir $OutputDir `
    --counterfactual-cache $Cache `
    --episodes 12 `
    --behavior-steps 4000 `
    --warmup-steps 1200 `
    --sample-stride 60 `
    --candidates-per-state 3 `
    --max-rollout-steps 500 `
    --epochs 60 `
    --patience 10 `
    --batch-size 256 `
    --learning-rate 0.0003 `
    --weight-decay 0.0001 `
    --data-seed 14100 `
    --split-seed 14100 `
    --seed $Seed `
    --cpu-threads $CpuThreads `
    @DeviceArgs
  if ($LASTEXITCODE -ne 0) {
    throw "V15.0 flat-MLP training failed for seed ${Seed}."
  }
}

Write-Host "V15.0 parameter-matched flat baselines finished for seeds 42, 43, and 44."

