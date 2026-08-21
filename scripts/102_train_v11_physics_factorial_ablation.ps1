param(
  [ValidateSet("full_retrained", "no_physics_loss", "no_physical_features", "data_only", "all")]
  [string]$Condition = "all",
  [switch]$RequireGpu,
  [ValidateRange(1, 64)]
  [int]$CpuThreads = 4
)

$ErrorActionPreference = "Stop"
throw "This training entry point is retired because it predates independent per-template arrival streams. Use scripts/106_train_v11_independent_arrival_factorial.ps1."
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$env:OMP_NUM_THREADS = "$CpuThreads"
$env:MKL_NUM_THREADS = "$CpuThreads"
$env:OPENBLAS_NUM_THREADS = "$CpuThreads"
$env:NUMEXPR_NUM_THREADS = "$CpuThreads"

$CudaProbe = python -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU'); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device')"
if ($LASTEXITCODE -ne 0) {
  throw "Unable to import PyTorch. Activate the intended Python environment first."
}
$Device = $CudaProbe[0].Trim().ToLowerInvariant()
Write-Host "PyTorch probe: $($CudaProbe -join ' | ')"

$DeviceArgs = @("--device", $Device)
if ($Device -eq "cuda") {
  $DeviceArgs += @("--amp", "--require-cuda")
  Write-Host "CUDA detected. V11 factorial ablations will use GPU mixed precision."
} elseif ($RequireGpu) {
  throw "GPU training was required, but this Python environment uses CPU-only PyTorch."
} else {
  Write-Warning "CUDA is unavailable. Training will run on CPU."
}

$Conditions = @{
  full_retrained = @{
    PhysicalFeatureMode = "full"
    PhysicsWeight = "0.50"
    OutputStem = "pi_gwm_multistep_v11_factorial_full"
  }
  no_physics_loss = @{
    PhysicalFeatureMode = "full"
    PhysicsWeight = "0.00"
    OutputStem = "pi_gwm_multistep_v11_ablation_no_physics_loss"
  }
  no_physical_features = @{
    PhysicalFeatureMode = "zero"
    PhysicsWeight = "0.50"
    OutputStem = "pi_gwm_multistep_v11_ablation_no_physical_features"
  }
  data_only = @{
    PhysicalFeatureMode = "zero"
    PhysicsWeight = "0.00"
    OutputStem = "pi_gwm_multistep_v11_ablation_data_only"
  }
}

$SelectedConditions = if ($Condition -eq "all") {
  # The retrained full model runs first and creates the new schema-matched cache.
  @("full_retrained", "no_physics_loss", "no_physical_features", "data_only")
} else {
  @($Condition)
}

$FactorialCache = "world_model_runs/pi_gwm_multistep_v11_factorial_shared/transitions_schema_v3_seed4200.pkl.gz"

foreach ($ConditionName in $SelectedConditions) {
  $Config = $Conditions[$ConditionName]
  Write-Host "Starting preregistered condition: $ConditionName"
  foreach ($Seed in @(42, 43, 44)) {
    $OutputDir = "world_model_runs/$($Config.OutputStem)_seed${Seed}"
    python train_world_model_multistep.py `
      --model-variant v11 `
      --physical-feature-mode $Config.PhysicalFeatureMode `
      --episodes 60 `
      --max-steps 400 `
      --epochs 80 `
      --batch-size 256 `
      --learning-rate 0.0003 `
      --weight-decay 0.0001 `
      --physics-weight $Config.PhysicsWeight `
      --rollout-discount 0.90 `
      --training-horizon 5 `
      --sequence-stride 1 `
      --teacher-forcing-start 0.90 `
      --teacher-forcing-end 0.00 `
      --exploration-rate 0.25 `
      --hidden-dim 96 `
      --planning-horizon 3 `
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
      --transition-cache $FactorialCache `
      --cpu-threads $CpuThreads `
      --low-priority `
      --seed $Seed `
      --output-dir $OutputDir `
      @DeviceArgs
    if ($LASTEXITCODE -ne 0) {
      throw "V11 factorial ablation $ConditionName failed for seed ${Seed}."
    }
  }
}

Write-Host "Requested V11 physics-factorial ablation training finished."
