[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Msi = Join-Path $RepoRoot "release\Yamatana-AI-IME-MOZC-Ver-2.1.0-beta-x64.msi"
$UninstallLog = Join-Path $RepoRoot "release\uninstall-clean.log"
$InstallLog = Join-Path $RepoRoot "release\install-clean.log"
$StatusResult = Join-Path $RepoRoot "release\install-result.json"

# Check administrator privileges
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "Elevating with administrator privileges..." -ForegroundColor Yellow
    $scriptPath = $PSCommandPath
    $proc = Start-Process powershell.exe -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$scriptPath`""
    ) -Verb RunAs -PassThru -Wait
    exit $proc.ExitCode
}

Write-Host "=== Yamatana / Fumiori AI IME Clean Reinstall ===" -ForegroundColor Cyan

# 1. Stop processes
Write-Host "1. Stopping running processes..." -ForegroundColor Cyan
Get-Process -Name YamatanaAIIME,mozc_server,mozc_renderer -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

# 2. Uninstall old product if installed
$oldProduct = Get-ItemProperty "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*" -ErrorAction SilentlyContinue |
    Where-Object { $_.DisplayName -like "*Fumiori*" -or $_.DisplayName -like "*Yamatana*" } |
    Select-Object -First 1

if ($oldProduct -and $oldProduct.PSChildName) {
    $guid = $oldProduct.PSChildName
    Write-Host "2. Uninstalling previous version: $($oldProduct.DisplayName) ($guid)..." -ForegroundColor Cyan
    $p = Start-Process msiexec.exe -ArgumentList @('/x', "`"$guid`"", '/qn', '/norestart', '/L*v', "`"$UninstallLog`"") -Wait -PassThru
    Write-Host "   Uninstall exit code: $($p.ExitCode)"
    Start-Sleep -Seconds 2
} else {
    Write-Host "2. No existing installation found (skipping uninstall)" -ForegroundColor Gray
}

# 3. Install new MSI
Write-Host "3. Installing new Dual-Encoder 70M MSI..." -ForegroundColor Cyan
Write-Host "   Target MSI: $Msi"
$p = Start-Process msiexec.exe -ArgumentList @('/i', "`"$Msi`"", '/qn', '/norestart', '/L*v', "`"$InstallLog`"") -Wait -PassThru
$installExitCode = $p.ExitCode
Write-Host "   Install exit code: $installExitCode"

if ($installExitCode -ne 0 -and $installExitCode -ne 3010) {
    $errObj = @{
        Success = $false
        ExitCode = $installExitCode
        Message = "MSI installation failed with exit code $installExitCode"
    }
    $errObj | ConvertTo-Json | Set-Content -Path $StatusResult -Encoding utf8
    throw "MSI installation failed with exit code $installExitCode"
}

# 4. Verify installation
$runtimeExe = "C:\Program Files (x86)\Yamatana AI IME\ai_runtime\YamatanaAIIME.exe"
$checkCode = -1
if (Test-Path $runtimeExe) {
    Write-Host "4. Running startup self-test on installed binary..." -ForegroundColor Cyan
    $proc = Start-Process -FilePath $runtimeExe -ArgumentList "--check" -Wait -PassThru
    $checkCode = $proc.ExitCode
    Write-Host "   Self-test exit code: $checkCode"
    
    Write-Host "5. Launching background tray..." -ForegroundColor Cyan
    Start-Process -FilePath $runtimeExe
} else {
    Write-Host "   Error: $runtimeExe was not found after installation" -ForegroundColor Red
}

$resObj = @{
    Success = ($installExitCode -in 0, 3010) -and ($checkCode -eq 0)
    InstallExitCode = $installExitCode
    CheckExitCode = $checkCode
    MsiPath = $Msi
}
$resObj | ConvertTo-Json | Set-Content -Path $StatusResult -Encoding utf8
Write-Host "=== Reinstall completed successfully! ===" -ForegroundColor Green
