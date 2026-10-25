@echo off
setlocal
cd /d "%~dp0"
set "BOOT_ARGS="
:parse_args
if "%~1"=="" goto launch
if /I "%~1"=="--lightweight" (
  set "BOOT_ARGS=%BOOT_ARGS% -Lightweight"
) else if /I "%~1"=="--refresh-services" (
  set "BOOT_ARGS=%BOOT_ARGS% -RefreshServices"
) else if /I "%~1"=="--repair" (
  set "BOOT_ARGS=%BOOT_ARGS% -Repair"
) else if /I "%~1"=="--setup-only" (
  set "BOOT_ARGS=%BOOT_ARGS% -SetupOnly"
) else (
  echo Unsupported startup option.
  exit /b 2
)
shift
goto parse_args
:launch
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %BOOT_ARGS%
if errorlevel 1 (
  echo Setup or launch failed. See data\logs\bootstrap.log and retry start.cmd.
  pause
  exit /b 1
)
exit /b %errorlevel%
