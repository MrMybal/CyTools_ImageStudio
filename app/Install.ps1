param([string]$Python = "python", [string]$CoreWheel = "")
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not $CoreWheel) {
    $wheel = Get-ChildItem -LiteralPath "$PSScriptRoot\packages" -Filter 'cytools_core-0.12.0-*-win_amd64.whl' -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $wheel) { throw 'Place the Windows CyToolsCore 0.12.0 wheel in app/packages or pass -CoreWheel.' }
    $CoreWheel = $wheel.FullName
}
$privatePython = Join-Path $root 'runtime\python\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $privatePython)) {
    & $Python -m venv (Join-Path $root 'runtime\python')
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $privatePython -m pip --disable-pip-version-check install --keyring-provider disabled $CoreWheel -r "$PSScriptRoot\requirements.txt"
exit $LASTEXITCODE
