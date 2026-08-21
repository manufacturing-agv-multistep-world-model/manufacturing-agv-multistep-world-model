param(
  [string]$Python = "python",
  [switch]$RequireGpu
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Runner = Join-Path $ProjectRoot "analyze_v151_utility_sensitivity.py"
$ConfirmationDir = Join-Path $ProjectRoot "experiment_results\v151_paired_vs_absolute_confirmation_seed18400"
$Cache = Join-Path $ConfirmationDir "fresh_paired_samples_seed18400.pkl.gz"
$OutputDir = Join-Path $ProjectRoot "experiment_results\v151_utility_sensitivity_v1"
$DeviceArgs = @("--device", "auto")
if ($RequireGpu) { $DeviceArgs += "--require-cuda" }
$Checkpoints = @()
foreach ($Seed in @(42, 43, 44)) {
  $Paired = Join-Path $ProjectRoot "world_model_runs\pi_gwm_counterfactual_v141_seed${Seed}\physics_graph_world_model_counterfactual.pt"
  $Absolute = Join-Path $ProjectRoot "world_model_runs\pi_gwm_absolute_v151_seed${Seed}\absolute_outcome_graph_baseline.pt"
  if (-not (Test-Path -LiteralPath $Paired)) { throw "Missing paired checkpoint: ${Paired}" }
  if (-not (Test-Path -LiteralPath $Absolute)) { throw "Missing absolute checkpoint: ${Absolute}" }
  $Checkpoints += @("--paired-checkpoint", $Paired, "--absolute-checkpoint", $Absolute)
}

& $Python $Runner @Checkpoints `
  --confirmation-cache $Cache `
  --output-dir $OutputDir `
  --bootstrap-replicates 5000 `
  --bootstrap-seed 18599 `
  --batch-size 256 `
  @DeviceArgs
if ($LASTEXITCODE -ne 0) { throw "V15.1 utility sensitivity failed." }
Write-Host "V15.1 utility sensitivity finished: ${OutputDir}"
