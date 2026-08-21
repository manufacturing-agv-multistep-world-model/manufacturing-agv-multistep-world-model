param(
  [switch]$RequireGpu,
  [ValidateRange(1, 64)]
  [int]$CpuThreads = 4
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$env:OMP_NUM_THREADS = "$CpuThreads"
$env:MKL_NUM_THREADS = "$CpuThreads"
$env:OPENBLAS_NUM_THREADS = "$CpuThreads"
$env:NUMEXPR_NUM_THREADS = "$CpuThreads"

$CudaAvailable = python -c "import torch; print('1' if torch.cuda.is_available() else '0')"
if ($LASTEXITCODE -ne 0) { throw "Unable to probe PyTorch CUDA support." }
$Device = if ($CudaAvailable.Trim() -eq "1") { "cuda" } else { "cpu" }
if ($RequireGpu -and $Device -ne "cuda") {
  throw "GPU evaluation was required, but CUDA is unavailable."
}

$OutputDir = Join-Path $ProjectRoot "experiment_results\v11_physics_factorial_arrival_v4_independent_v2"
python (Join-Path $ProjectRoot "evaluate_v11_physics_factorial.py") `
  --output-dir $OutputDir `
  --episodes 20 `
  --max-steps 400 `
  --sequence-stride 5 `
  --batch-size 128 `
  --seed 27413 `
  --device $Device
if ($LASTEXITCODE -ne 0) { throw "V11 arrival-v4 physics-factorial evaluation failed." }

$Report = Join-Path $ProjectRoot "experiment_results\v11_physics_factorial_arrival_v4_independent_v2_audit.md"
python (Join-Path $ProjectRoot "analyze_v11_physics_factorial.py") `
  --evaluation-dir $OutputDir `
  --report $Report `
  --bootstrap-replicates 5000 `
  --bootstrap-seed 27414
if ($LASTEXITCODE -ne 0) { throw "V11 arrival-v4 physics-factorial audit failed." }

Write-Host "V11 arrival-v4 physics-factorial independent evaluation finished."
