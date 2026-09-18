[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$InstallRoot = 'C:\Program Files (x86)\Yamatana AI IME'
$RuntimeTarget = Join-Path $InstallRoot 'ai_runtime\YamatanaAIIME.exe'
$ServerTarget = Join-Path $InstallRoot 'mozc_server.exe'
$RuntimeSource = Join-Path $Root 'hotfix\YamatanaAIIME.exe'
$ServerSource = Join-Path $Root 'hotfix\mozc_server.exe'
if (-not (Test-Path -LiteralPath $RuntimeSource)) {
    $RuntimeSource = Join-Path $Root 'dist\YamatanaAIIME\YamatanaAIIME.exe'
}
if (-not (Test-Path -LiteralPath $ServerSource)) {
    $ServerSource = Get-ChildItem (Join-Path $Root 'build\mozc-src\src\bazel-out') -Recurse -Filter 'mozc_server.exe.exe' -File |
        Where-Object { $_.FullName -match 'x64_windows-opt-.*\\bin\\server\\mozc_server\.exe\.exe$' } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1 -ExpandProperty FullName
}

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]$identity
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    $scriptPath = (Resolve-Path $PSCommandPath).Path
    $child = Start-Process powershell.exe -Verb RunAs -Wait -PassThru -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $scriptPath
    )
    exit $child.ExitCode
}

foreach ($source in @($RuntimeSource, $ServerSource)) {
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Hotfix file not found: $source"
    }
}

Get-Process -Name YamatanaAIIME,mozc_server,mozc_renderer -ErrorAction SilentlyContinue |
    Stop-Process -Force
Start-Sleep -Seconds 2

Copy-Item -LiteralPath $RuntimeSource -Destination $RuntimeTarget -Force
Copy-Item -LiteralPath $ServerSource -Destination $ServerTarget -Force

$check = Start-Process -FilePath (Join-Path $InstallRoot 'ai_runtime\YamatanaAIIME.exe') -ArgumentList '--check' -Wait -PassThru
if ($check.ExitCode -ne 0) {
    throw "Installed runtime self-test failed with exit code $($check.ExitCode)"
}

Start-Process -FilePath (Join-Path $InstallRoot 'ai_runtime\YamatanaAIIME.exe')
Write-Host 'Yamatana AI IME hotfix applied successfully.' -ForegroundColor Green
