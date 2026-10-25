param([switch]$Repair, [switch]$Build, [switch]$FreshRuntime)
$ErrorActionPreference = 'Stop'
$rootPath = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $rootPath
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$runtimePath = Join-Path $rootPath 'data\runtimes'
$null = New-Item -ItemType Directory -Force -Path $runtimePath
$lockPath = Join-Path $rootPath 'data\bootstrap.lock'
$bootstrapLock = $null
for ($attempt = 0; $attempt -lt 120; $attempt++) {
    try { $bootstrapLock = [IO.File]::Open($lockPath, 'OpenOrCreate', 'ReadWrite', 'None'); break }
    catch [IO.IOException] { if ($attempt -eq 0) { Write-Host 'Waiting for another environment setup...' }; Start-Sleep -Milliseconds 500 }
}
if (-not $bootstrapLock) { throw 'Another environment setup is still running. Please try again later.' }

function Get-VerifiedArchive([string]$url, [string]$target, [string]$sha) {
    if ((Test-Path -LiteralPath $target) -and (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -eq $sha) { return }
    $partialPath = $target + '.part'
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $request = [Net.HttpWebRequest]::Create($url)
    $request.Timeout = 60000
    $request.ReadWriteTimeout = 60000
    $response = $request.GetResponse()
    $inputStream = $response.GetResponseStream()
    $outputStream = [IO.File]::Open($partialPath, 'Create', 'Write', 'None')
    try {
        $buffer = New-Object byte[] (1024 * 1024)
        $received = [long]0
        while (($count = $inputStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
            $outputStream.Write($buffer, 0, $count)
            $received += $count
            if ($response.ContentLength -gt 0) {
                Write-Progress -Activity 'Downloading environment manager' -Status "$received / $($response.ContentLength) bytes" -PercentComplete ([Math]::Min(100, 100 * $received / $response.ContentLength))
            }
        }
    } finally { $outputStream.Dispose(); $inputStream.Dispose(); $response.Dispose(); Write-Progress -Activity 'Downloading environment manager' -Completed }
    if ((Get-FileHash -LiteralPath $partialPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sha) { throw 'Environment manager checksum mismatch. Retry to download again.' }
    Move-Item -LiteralPath $partialPath -Destination $target -Force
}

try {
    if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { throw 'Automatic first-run setup currently requires Windows x64.' }
    $systemPath = Join-Path $env:SystemRoot 'System32'
    if (-not (Test-Path -LiteralPath (Join-Path $systemPath 'msvcp140.dll')) -or -not (Test-Path -LiteralPath (Join-Path $systemPath 'vcruntime140_1.dll'))) {
        Write-Host 'Preparing Microsoft Visual C++ runtime. Windows may request administrator authorization.'
        $redistPath = Join-Path $runtimePath 'vc_redist.x64.exe'
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -UseBasicParsing -Uri 'https://aka.ms/vc14/vc_redist.x64.exe' -OutFile $redistPath
        $signature = Get-AuthenticodeSignature -LiteralPath $redistPath
        if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation') { throw 'Microsoft runtime signature verification failed; the installer was not executed.' }
        $redistProcess = Start-Process -FilePath $redistPath -ArgumentList '/install', '/quiet', '/norestart' -Verb RunAs -WindowStyle Hidden -Wait -PassThru
        if ($redistProcess.ExitCode -notin @(0, 3010, 1638)) { throw "Microsoft runtime installation failed (code $($redistProcess.ExitCode)). Administrator authorization may be required." }
        if ($redistProcess.ExitCode -eq 3010) { Write-Host 'Microsoft runtime installation requests a restart. Restart Windows if subsequent native module validation fails.' }
        if (-not (Test-Path -LiteralPath (Join-Path $systemPath 'msvcp140.dll'))) { throw 'Microsoft runtime is not usable yet. Restart Windows and run start.cmd again.' }
    }
    $appPython = Join-Path $rootPath '.venv\Scripts\python.exe'
    $pythonReady = $false
    if ((Test-Path -LiteralPath $appPython) -and -not $FreshRuntime) {
        & $appPython -c 'import sys, ssl, sqlite3, venv; assert sys.version_info[:2] == (3, 12)' 2>$null
        $pythonReady = $LASTEXITCODE -eq 0
    }
    if (-not $pythonReady) {
        Write-Host '[1/4] Preparing project-owned Python 3.12 (no system PATH changes)...'
        $uvPath = Join-Path $runtimePath 'uv-0.13.5'
        $uvExe = Join-Path $uvPath 'uv.exe'
        if (-not (Test-Path -LiteralPath $uvExe)) {
            $archivePath = Join-Path $runtimePath 'uv-0.13.5.zip'
            Get-VerifiedArchive 'https://releases.astral.sh/github/uv/releases/download/0.13.5/uv-x86_64-pc-windows-msvc.zip' $archivePath 'a38b8d1902b7cfbc76b423ba1bb456f623f42064cdc72628b6075b89afd1019a'
            Expand-Archive -LiteralPath $archivePath -DestinationPath $uvPath -Force
        }
        $env:UV_PYTHON_INSTALL_DIR = Join-Path $runtimePath 'python'
        $env:UV_CACHE_DIR = Join-Path $rootPath 'data\cache\uv'
        $env:UV_PYTHON_NO_REGISTRY = 'true'
        $env:UV_PYTHON_INSTALL_REGISTRY = 'false'
        $env:UV_PYTHON_INSTALL_BIN = 'false'
        $env:UV_PYTHON_PREFERENCE = 'only-managed'
        $venvPath = Join-Path $rootPath '.venv'
        if (Test-Path -LiteralPath $venvPath) {
            # Preserve a broken/old environment rather than deleting user files.
            $resolvedVenv = [IO.Path]::GetFullPath($venvPath)
            if ($resolvedVenv -ne [IO.Path]::GetFullPath((Join-Path $rootPath '.venv'))) { throw 'Unexpected environment path' }
            Move-Item -LiteralPath $resolvedVenv -Destination ($resolvedVenv + '.backup-' + [DateTime]::UtcNow.ToString('yyyyMMddHHmmss'))
        }
        & $uvExe venv --seed --python 3.12 $venvPath
        if ($LASTEXITCODE -ne 0) { throw 'Python installation failed. Check the network and restart start.cmd to retry.' }
    } else { Write-Host '[1/4] Python environment is usable.' }
    $bootstrapArgs = @((Join-Path $PSScriptRoot 'bootstrap_app.py'))
    if ($Repair) { $bootstrapArgs += '--repair' }
    if ($Build) { $bootstrapArgs += '--build' }
    & $appPython @bootstrapArgs
    if ($LASTEXITCODE -ne 0) { throw 'Application environment setup failed. See data\logs\bootstrap.log and retry start.cmd.' }
} finally { $bootstrapLock.Dispose() }
