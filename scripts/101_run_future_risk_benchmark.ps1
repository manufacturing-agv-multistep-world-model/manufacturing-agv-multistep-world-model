param(
  [string]$Python = "python",
  [switch]$RequireGpu
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$CudaAvailable = & $Python -c "import torch; print('1' if torch.cuda.is_available() else '0')"
if ($LASTEXITCODE -ne 0) { throw "Unable to probe PyTorch CUDA support." }
$Device = if ($CudaAvailable.Trim() -eq "1") { "cuda" } else { "cpu" }
if ($RequireGpu -and $Device -ne "cuda") {
  throw "GPU benchmark was required, but CUDA is unavailable."
}

$OutputDir = Join-Path $ProjectRoot "experiment_results\future_risk_architecture_benchmark_v1"
& $Python (Join-Path $ProjectRoot "evaluate_future_risk_benchmark.py") `
  --output-dir $OutputDir `
  --episodes 10 `
  --max-steps 4000 `
  --sequence-stride 10 `
  --batch-size 128 `
  --bootstrap-replicates 2000 `
  --seed 25313 `
  --device $Device
if ($LASTEXITCODE -ne 0) { throw "Future-risk architecture benchmark failed." }

$Report = Join-Path $ProjectRoot "experiment_results\future_risk_architecture_benchmark_v1_audit.md"
& $Python (Join-Path $ProjectRoot "analyze_future_risk_benchmark.py") `
  --benchmark-dir $OutputDir `
  --report $Report
if ($LASTEXITCODE -ne 0) { throw "Future-risk benchmark audit failed." }
Write-Host "Future-risk architecture benchmark finished."
