[CmdletBinding()]
param([Alias('refresh-services')][switch]$RefreshServices, [switch]$Lightweight, [switch]$Repair, [Alias('setup-only')][switch]$SetupOnly)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
try {
    & (Join-Path $PSScriptRoot 'scripts\bootstrap.ps1') -Repair:$Repair
    $appPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if ($SetupOnly) { exit 0 }
    if ($RefreshServices) { & $appPython scripts/bootstrap_services.py; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }
    $runArgs = @('scripts/run.py')
    if ($Lightweight) { $runArgs += '--lightweight' }
    & $appPython @runArgs
    exit $LASTEXITCODE
} catch { Write-Host $_.Exception.Message -ForegroundColor Red; exit 1 }

