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

$processNames = @(
    'YamatanaAIIME'
    'mozc_broker'
    'mozc_server'
    'mozc_renderer'
)

# The broker can respawn mozc_server while the files are being replaced. Stop
# the tray/broker first, then stop any remaining Mozc processes and wait until
# Windows reports that they are actually gone.
foreach ($name in @('YamatanaAIIME', 'mozc_broker')) {
    Get-Process -Name $name -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
    & "$env:SystemRoot\System32\taskkill.exe" /F /T /IM "$name.exe" 2>$null | Out-Null
}
Get-Process -Name $processNames -ErrorAction SilentlyContinue |
    Stop-Process -Force -ErrorAction SilentlyContinue

$processDeadline = (Get-Date).AddSeconds(20)
do {
    $running = @(Get-Process -Name $processNames -ErrorAction SilentlyContinue)
    if ($running.Count -eq 0) {
        break
    }
    Start-Sleep -Milliseconds 250
} while ((Get-Date) -lt $processDeadline)

if ($running.Count -ne 0) {
    $names = ($running | Select-Object -ExpandProperty ProcessName -Unique) -join ', '
    throw "Could not stop IME processes before replacing hotfix files: $names"
}

function Copy-HotfixFile {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    $attempts = 40
    for ($attempt = 1; $attempt -le $attempts; $attempt++) {
        try {
            if (Test-Path -LiteralPath $Destination) {
                $stream = [System.IO.File]::Open(
                    $Destination,
                    [System.IO.FileMode]::Open,
                    [System.IO.FileAccess]::ReadWrite,
                    [System.IO.FileShare]::None
                )
                $stream.Dispose()
            }
            Copy-Item -LiteralPath $Source -Destination $Destination -Force -ErrorAction Stop
            return
        } catch {
            if ($attempt -eq $attempts) {
                throw "Could not replace '$Destination' after $attempts attempts. The file is still locked or inaccessible. Last error: $($_.Exception.Message)"
            }
            Start-Sleep -Milliseconds 250
        }
    }
}

Copy-HotfixFile -Source $RuntimeSource -Destination $RuntimeTarget
Copy-HotfixFile -Source $ServerSource -Destination $ServerTarget

$check = Start-Process -FilePath (Join-Path $InstallRoot 'ai_runtime\YamatanaAIIME.exe') -ArgumentList '--check' -Wait -PassThru
if ($check.ExitCode -ne 0) {
    throw "Installed runtime self-test failed with exit code $($check.ExitCode)"
}

Start-Process -FilePath (Join-Path $InstallRoot 'ai_runtime\YamatanaAIIME.exe')
Write-Host 'Yamatana AI IME hotfix applied successfully.' -ForegroundColor Green
