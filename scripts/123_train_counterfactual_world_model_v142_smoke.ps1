param(
  [string]$Python = "python",
  [switch]$RequireGpu,
  [ValidateRange(1, 64)]
  [int]$CpuThreads = 4
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Trainer = Join-Path $ProjectRoot "train_counterfactual_world_model_v142.py"
$InitCheckpoint = Join-Path $ProjectRoot "world_model_runs\pi_gwm_counterfactual_v141_seed42\physics_graph_world_model_counterfactual.pt"
$Cache = Join-Path $ProjectRoot "world_model_runs\pi_gwm_counterfactual_v141_shared\paired_material_samples_v2_seed14100.pkl.gz"
$OutputDir = Join-Path $ProjectRoot "world_model_runs\pi_gwm_counterfactual_v142_smoke_seed42"
$DeviceArgs = @("--device", "auto")
if ($RequireGpu) {
  $DeviceArgs += @("--require-cuda", "--amp")
}

& $Python $Trainer `
  --init-checkpoint $InitCheckpoint `
  --counterfactual-cache $Cache `
  --output-dir $OutputDir `
  --episodes 12 `
  --behavior-steps 4000 `
  --warmup-steps 1200 `
  --sample-stride 60 `
  --candidates-per-state 3 `
  --max-rollout-steps 500 `
  --data-seed 14100 `
  --split-seed 14100 `
  --epochs 20 `
  --patience 6 `
  --batch-size 192 `
  --learning-rate 0.0001 `
  --weight-decay 0.0001 `
  --seed 42 `
  --cpu-threads $CpuThreads `
  @DeviceArgs
if ($LASTEXITCODE -ne 0) {
  throw "V14.2 smoke training failed."
}

Write-Host "V14.2 smoke training finished: ${OutputDir}"
