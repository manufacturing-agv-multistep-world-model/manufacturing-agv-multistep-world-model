param(
  [string]$Python = "python",
  [switch]$RequireGpu,
  [ValidateRange(1, 64)]
  [int]$CpuThreads = 4
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Trainer = Join-Path $ProjectRoot "train_absolute_outcome_baseline_v151.py"
$Cache = Join-Path $ProjectRoot "world_model_runs\pi_gwm_counterfactual_v141_shared\paired_material_samples_v2_seed14100.pkl.gz"
if (-not (Test-Path -LiteralPath $Cache)) {
  throw "Missing frozen paired training cache: ${Cache}"
}
$DeviceArgs = @("--device", "auto")
if ($RequireGpu) {
  $DeviceArgs += @("--require-cuda", "--amp")
}

foreach ($Seed in @(42, 43, 44)) {
  $InitCheckpoint = Join-Path $ProjectRoot "world_model_runs\pi_gwm_multistep_v13_multiscale_v2_seed${Seed}\physics_graph_world_model_multistep.pt"
  if (-not (Test-Path -LiteralPath $InitCheckpoint)) {
    throw "Missing V13 initialization checkpoint for seed ${Seed}: ${InitCheckpoint}"
  }
  $OutputDir = Join-Path $ProjectRoot "world_model_runs\pi_gwm_absolute_v151_seed${Seed}"
  & $Python $Trainer `
    --init-checkpoint $InitCheckpoint `
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
    throw "V15.1 absolute-outcome training failed for seed ${Seed}."
  }
}

Write-Host "V15.1 absolute-outcome baselines finished for seeds 42, 43, and 44."
