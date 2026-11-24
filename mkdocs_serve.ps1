$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot ".venv-docs\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Documentatieomgeving ontbreekt. Volg de installatie in docs/guides/contribute.md."
}

Push-Location $PSScriptRoot
try {
    & $python -m mkdocs serve @args
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

exit $exitCode
