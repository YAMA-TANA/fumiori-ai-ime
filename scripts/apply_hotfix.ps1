[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$InstallRoot = 'C:\Program Files (x86)\Yamatana AI IME'
$RuntimeTarget = Join-Path $InstallRoot 'ai_runtime\YamatanaAIIME.exe'
$ServerTarget = Join-Path $InstallRoot 'mozc_server.exe'
$CacheServiceName = 'MozcCacheService'
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
    'mozc_cache_service'
    'mozc_server'
    'mozc_renderer'
)

function Stop-LockingProcesses {
    # MozcCacheService keeps mozc_server.exe open even after the converter
    # process itself is terminated. Stop the service before killing the
    # remaining processes so the binary can actually be replaced.
    $service = Get-Service -Name $CacheServiceName -ErrorAction SilentlyContinue
    if ($service -and $service.Status -ne 'Stopped') {
        Stop-Service -Name $CacheServiceName -Force -ErrorAction SilentlyContinue
        try {
            $service.WaitForStatus('Stopped', '00:00:05')
        } catch {
            # The process-kill pass below is still useful if SCM is slow.
        }
    }

    foreach ($name in $processNames) {
        # Start-Process keeps taskkill's "process not found" stderr out of
        # PowerShell's native-command error pipeline. Exit code 128 is normal.
        Start-Process -FilePath "$env:SystemRoot\System32\taskkill.exe" `
            -ArgumentList @('/F', '/T', '/IM', "$name.exe") `
            -WindowStyle Hidden -Wait -PassThru | Out-Null
    }
}

$cacheServiceWasRunning = $false
$initialCacheService = Get-Service -Name $CacheServiceName -ErrorAction SilentlyContinue
if ($initialCacheService -and $initialCacheService.Status -eq 'Running') {
    $cacheServiceWasRunning = $true
}

function Copy-HotfixFile {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    if (Test-Path -LiteralPath $Destination) {
        $sourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Source).Hash
        $destinationHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Destination).Hash
        if ($sourceHash -eq $destinationHash) {
            Write-Host "Already current: $Destination"
            return
        }

        # The MSI marks Mozc binaries read-only. Clear that attribute before
        # probing/replacing the destination; otherwise even an elevated copy
        # fails with AccessDenied when no process is holding the file.
        $destinationItem = Get-Item -LiteralPath $Destination -Force
        if ($destinationItem.IsReadOnly) {
            $destinationItem.IsReadOnly = $false
        }
    }

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
            Stop-LockingProcesses
            Start-Sleep -Milliseconds 250
        }
    }
}

Stop-LockingProcesses
Start-Sleep -Milliseconds 250
try {
    Copy-HotfixFile -Source $RuntimeSource -Destination $RuntimeTarget
    Copy-HotfixFile -Source $ServerSource -Destination $ServerTarget

    $check = Start-Process -FilePath (Join-Path $InstallRoot 'ai_runtime\YamatanaAIIME.exe') -ArgumentList '--check' -Wait -PassThru
    if ($check.ExitCode -ne 0) {
        throw "Installed runtime self-test failed with exit code $($check.ExitCode)"
    }

    if (@(Get-Process -Name YamatanaAIIME -ErrorAction SilentlyContinue).Count -eq 0) {
        Start-Process -FilePath (Join-Path $InstallRoot 'ai_runtime\YamatanaAIIME.exe')
    }
} finally {
    if ($cacheServiceWasRunning) {
        Start-Service -Name $CacheServiceName -ErrorAction SilentlyContinue
    }
}
Write-Host 'Yamatana AI IME hotfix applied successfully.' -ForegroundColor Green
