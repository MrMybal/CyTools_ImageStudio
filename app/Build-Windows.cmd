@echo off
setlocal
set "vcvars=%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCToolsInstallDir (
    if not exist "%vcvars%" (
        echo Open an x64 Visual Studio developer prompt and run this script.
        exit /b 1
    )
    call "%vcvars%" || exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_launcher.ps1"
exit /b %errorlevel%
