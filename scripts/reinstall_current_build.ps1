[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Msi = Join-Path $Root 'release\Yamatana-AI-IME-MOZC-Ver-2.1.0-beta-x64.msi'
$RuntimeSource = Join-Path $Root 'dist\YamatanaAIIME'
$InstallRoot = 'C:\Program Files (x86)\Yamatana AI IME'
$RuntimeTarget = Join-Path $InstallRoot 'ai_runtime'
$ServerTarget = Join-Path $InstallRoot 'mozc_server.exe'
$StatusPath = Join-Path $Root 'release\reinstall-current-build.json'

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]$identity
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    $scriptPath = (Resolve-Path $PSCommandPath).Path
    $child = Start-Process powershell.exe -Verb RunAs -Wait -PassThru -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $scriptPath
    )
    exit $child.ExitCode
}

if (-not (Test-Path -LiteralPath $Msi)) { throw "MSI not found: $Msi" }
if (-not (Test-Path -LiteralPath (Join-Path $RuntimeSource 'YamatanaAIIME.exe'))) {
    throw "PyInstaller runtime not found: $RuntimeSource"
}

$server = Get-ChildItem (Join-Path $Root 'build\mozc-src\src\bazel-out') -Recurse -Filter 'mozc_server.exe.exe' -File |
    Where-Object { $_.FullName -match 'x64_windows-opt-.*\\bin\\server\\mozc_server\.exe\.exe$' } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $server -or $server.Length -lt 1000000) {
    throw 'Release mozc_server.exe build was not found.'
}

Write-Host 'Stopping Yamatana/Mozc processes...'
Get-Process -Name YamatanaAIIME,mozc_server,mozc_renderer -ErrorAction SilentlyContinue |
    Stop-Process -Force
Start-Sleep -Seconds 2

$products = Get-ItemProperty 'HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*' -ErrorAction SilentlyContinue |
    Where-Object { $_.DisplayName -like '*Fumiori*' -or $_.DisplayName -like '*Yamatana*' } |
    Select-Object -First 1
if ($products -and $products.PSChildName) {
    Write-Host "Uninstalling $($products.DisplayName) ($($products.PSChildName))..."
    $uninstall = Start-Process msiexec.exe -Wait -PassThru -ArgumentList @(
        '/x', $products.PSChildName, '/qn', '/norestart', '/L*v', (Join-Path $Root 'release\uninstall-current-build.log')
    )
    if ($uninstall.ExitCode -notin @(0, 1605, 3010)) {
        throw "Uninstall failed with exit code $($uninstall.ExitCode)"
    }
}

Write-Host "Installing $Msi..."
$install = Start-Process msiexec.exe -Wait -PassThru -ArgumentList @(
    '/i', $Msi, '/qn', '/norestart', '/L*v', (Join-Path $Root 'release\install-current-build.log')
)
if ($install.ExitCode -notin @(0, 3010)) {
    throw "Install failed with exit code $($install.ExitCode)"
}

Write-Host 'Applying current Python runtime...'
Get-Process -Name YamatanaAIIME,mozc_server,mozc_renderer -ErrorAction SilentlyContinue |
    Stop-Process -Force
Start-Sleep -Seconds 2
robocopy $RuntimeSource $RuntimeTarget /E /R:2 /W:1 /NJH /NJS /NDL /NC /NS | Out-Null
if ($LASTEXITCODE -gt 7) { throw "Runtime copy failed with robocopy exit code $LASTEXITCODE" }

Write-Host "Applying current Mozc server: $($server.FullName)"
Copy-Item -LiteralPath $server.FullName -Destination $ServerTarget -Force

$runtimeExe = Join-Path $RuntimeTarget 'YamatanaAIIME.exe'
$check = Start-Process -FilePath $runtimeExe -ArgumentList '--check' -Wait -PassThru
$result = [ordered]@{
    Success = ($check.ExitCode -eq 0)
    InstallExitCode = $install.ExitCode
    CheckExitCode = $check.ExitCode
    Runtime = $runtimeExe
    Server = $ServerTarget
}
$result | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding utf8
if ($check.ExitCode -ne 0) { throw "Installed runtime self-test failed with exit code $($check.ExitCode)" }

Start-Process -FilePath $runtimeExe
Write-Host 'Current build installed successfully.' -ForegroundColor Green
