# Fumiori AI IME clean-PC installer. The two official release archives are pinned
# by SHA-256: v2.1.0 is the full MSI with models, v2.1.1 is a hotfix only.
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$ReleaseVersion = 'v2.1.1-beta'
$MsiName = 'Yamatana-AI-IME-MOZC-Ver-2.1.0-beta-x64.msi'
$MsiUrl = 'https://github.com/YAMA-TANA/fumiori-ai-ime/releases/download/v2.1.0-beta/' + $MsiName
$MsiHash = 'DEFE3A958C3C72751851CA67560488E7A279E3BF69DFEBAA51B00396E2D4B455'
$HotfixName = 'Yamatana-AI-IME-v2.1.1-beta-candidate10.zip'
$HotfixUrl = 'https://github.com/YAMA-TANA/fumiori-ai-ime/releases/download/v2.1.1-beta/' + $HotfixName
$HotfixHash = '0176933270A75B04CAD182D73543EBC4EC16272A5C798CD609D7021383235C51'
$InstallRoot = 'C:\Program Files (x86)\Yamatana AI IME'
$WorkDir = Join-Path $env:LOCALAPPDATA ('FumioriAIIME\Setup\' + $ReleaseVersion)
$LogPath = Join-Path $WorkDir 'install-msi.log'

function Assert-FileHash {
    param([string]$Path, [string]$Expected)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    if ($actual -ne $Expected) {
        Write-Warning ('SHA-256 mismatch; removing unexpected download: ' + $Path)
        Remove-Item -LiteralPath $Path -Force
        return $false
    }
    return $true
}

function Get-VerifiedReleaseFile {
    param([string]$Url, [string]$Name, [string]$Expected)
    $path = Join-Path $WorkDir $Name
    if (Assert-FileHash -Path $path -Expected $Expected) {
        Write-Host ('Already verified: ' + $Name)
        return $path
    }
    $part = $path + '.download'
    if (Test-Path -LiteralPath $part) { Remove-Item -LiteralPath $part -Force }
    Write-Host ('Downloading official release: ' + $Name)
    try {
        Invoke-WebRequest -Uri $Url -UseBasicParsing -OutFile $part -ErrorAction Stop
        if (-not (Assert-FileHash -Path $part -Expected $Expected)) {
            throw ('Download hash verification failed: ' + $Name)
        }
        Move-Item -LiteralPath $part -Destination $path -Force
    } finally {
        if (Test-Path -LiteralPath $part) { Remove-Item -LiteralPath $part -Force }
    }
    Write-Host ('SHA-256 verified: ' + $Name)
    return $path
}

try {
    if ($env:OS -ne 'Windows_NT' -or -not [Environment]::Is64BitOperatingSystem) {
        throw 'This installer requires 64-bit Windows 10/11.'
    }
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]$identity
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Host 'Administrator permission is needed. Please approve the Windows UAC prompt.'
        $argument = '-NoProfile -ExecutionPolicy Bypass -File "' + $PSCommandPath + '"'
        $child = Start-Process -FilePath (Join-Path $PSHOME 'powershell.exe') -Verb RunAs -ArgumentList $argument -Wait -PassThru
        exit $child.ExitCode
    }

    New-Item -ItemType Directory -Path $WorkDir -Force | Out-Null
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Write-Host ('Fumiori AI IME ' + $ReleaseVersion + ' clean-PC installation')
    Write-Host ('Installer log: ' + $LogPath)

    $msiPath = Get-VerifiedReleaseFile -Url $MsiUrl -Name $MsiName -Expected $MsiHash
    $zipPath = Get-VerifiedReleaseFile -Url $HotfixUrl -Name $HotfixName -Expected $HotfixHash

    $hotfixDir = Join-Path $WorkDir 'hotfix-extracted'
    if (Test-Path -LiteralPath $hotfixDir) { Remove-Item -LiteralPath $hotfixDir -Recurse -Force }
    Expand-Archive -LiteralPath $zipPath -DestinationPath $hotfixDir -Force
    $hotfixScript = Join-Path $hotfixDir 'scripts\apply_hotfix.ps1'
    $runtimeFile = Join-Path $hotfixDir 'hotfix\YamatanaAIIME.exe'
    $serverFile = Join-Path $hotfixDir 'hotfix\mozc_server.exe'
    foreach ($required in @($hotfixScript, $runtimeFile, $serverFile)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw ('Published hotfix is missing: ' + $required)
        }
    }

    Write-Host 'Installing full MSI (including the offline AI models)...'
    $process = Start-Process -FilePath (Join-Path $env:WINDIR 'System32\msiexec.exe') -Wait -PassThru -ArgumentList @(
        '/i', ('"' + $msiPath + '"'), '/passive', '/norestart',
        '/L*v', ('"' + $LogPath + '"')
    )
    $msiCode = [int]$process.ExitCode
    if ($msiCode -notin @(0, 3010, 1641)) {
        throw ('MSI failed with exit code ' + $msiCode + '. See ' + $LogPath)
    }
    if (-not (Test-Path -LiteralPath (Join-Path $InstallRoot 'mozc_server.exe'))) {
        throw ('MSI returned success but the IME was not found in ' + $InstallRoot)
    }

    Write-Host 'Applying the v2.1.1-beta Dual-Encoder update...'
    & $hotfixScript
    if (-not $?) { throw 'The Dual-Encoder update script did not complete.' }
    foreach ($pair in @(
        @($runtimeFile, (Join-Path $InstallRoot 'ai_runtime\YamatanaAIIME.exe')),
        @($serverFile, (Join-Path $InstallRoot 'mozc_server.exe'))
    )) {
        if (-not (Test-Path -LiteralPath $pair[1] -PathType Leaf) -or
            (Get-FileHash -LiteralPath $pair[0] -Algorithm SHA256).Hash -ne
            (Get-FileHash -LiteralPath $pair[1] -Algorithm SHA256).Hash) {
            throw ('Installed hotfix binary verification failed: ' + $pair[1])
        }
    }
    Write-Host ('Installation verified. MSI log: ' + $LogPath) -ForegroundColor Green
    Write-Host 'Sign out or restart Windows, then select Fumiori AI IME with Win + Space.' -ForegroundColor Yellow
    if ($msiCode -in @(3010, 1641)) {
        Write-Host 'RESTART REQUIRED: Windows Installer requested a PC restart.' -ForegroundColor Yellow
        exit 3010
    }
    exit 0
} catch {
    Write-Host ('Installation failed: ' + $_.Exception.Message) -ForegroundColor Red
    Write-Host ('For MSI diagnostics, see: ' + $LogPath) -ForegroundColor Yellow
    exit 1
}
