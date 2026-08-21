$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

python export_parameter_registry.py --output-dir "docs/generated_parameter_registry"

Write-Host "Parameter registry exported to docs/generated_parameter_registry"
