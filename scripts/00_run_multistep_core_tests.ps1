$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$env:PYTHONPATH = $ProjectRoot + [System.IO.Path]::PathSeparator + $env:PYTHONPATH

python -m unittest discover -s tests -p "test_*.py" -v
if ($LASTEXITCODE -ne 0) {
    throw "Multistep core tests failed."
}

Write-Host "Multistep core tests passed."
