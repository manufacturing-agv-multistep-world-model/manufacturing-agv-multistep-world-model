param(
  [string]$Python = "python",
  [switch]$RequireGpu,
  [ValidateRange(1, 64)]
  [int]$CpuThreads = 4
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Trainer = Join-Path $ProjectRoot "train_counterfactual_world_model_v141.py"
$InitCheckpoint = Join-Path $ProjectRoot "world_model_runs\pi_gwm_multistep_v13_multiscale_v2_seed42\physics_graph_world_model_multistep.pt"
$OutputDir = Join-Path $ProjectRoot "world_model_runs\pi_gwm_counterfactual_v141_smoke_seed42"
$Cache = Join-Path $ProjectRoot "world_model_runs\pi_gwm_counterfactual_v141_smoke_shared\paired_material_samples_v2.pkl.gz"
$DeviceArgs = @("--device", "auto")
if ($RequireGpu) {
  $DeviceArgs += "--require-cuda"
}

& $Python $Trainer `
  --init-checkpoint $InitCheckpoint `
  --output-dir $OutputDir `
  --counterfactual-cache $Cache `
  --episodes 4 `
  --behavior-steps 2200 `
  --warmup-steps 800 `
  --sample-stride 100 `
  --candidates-per-state 2 `
  --max-rollout-steps 500 `
  --epochs 20 `
  --patience 6 `
  --batch-size 128 `
  --learning-rate 0.0003 `
  --weight-decay 0.0001 `
  --data-seed 14100 `
  --split-seed 14100 `
  --seed 42 `
  --cpu-threads $CpuThreads `
  @DeviceArgs
if ($LASTEXITCODE -ne 0) {
  throw "V14.1 paired-counterfactual smoke training failed."
}

Write-Host "V14.1 paired-counterfactual smoke training finished: ${OutputDir}"
