@echo off
setlocal
echo ===================================================
echo   Yamatana / Fumiori AI IME Clean Reinstall
echo ===================================================
echo.

:: Check for Administrator privileges
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Elevating privileges...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process cmd -ArgumentList '/c \"\"%~f0\"\"' -Verb RunAs"
    exit /b
)

echo 1. Stopping running processes...
taskkill /F /IM YamatanaAIIME.exe 2>nul
taskkill /F /IM mozc_server.exe 2>nul
taskkill /F /IM mozc_renderer.exe 2>nul
timeout /t 2 /nobreak > nul

echo 2. Uninstalling previous version...
powershell -NoProfile -Command "Get-ItemProperty 'HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*' | Where-Object { $_.DisplayName -like '*Fumiori*' -or $_.DisplayName -like '*Yamatana*' } | ForEach-Object { $id = $_.PSChildName; Write-Host 'Uninstalling: ' $id; Start-Process msiexec.exe -ArgumentList @('/x', $id, '/qn', '/norestart') -Wait }"

echo 3. Installing new Dual-Encoder 70M MSI...
set "MSI=c:\Users\lotof\Videos\research\live2d\yamatana-ai-ime\release\Yamatana-AI-IME-MOZC-Ver-2.1.0-beta-x64.msi"
msiexec /i "%MSI%" /qn /norestart /L*v "c:\Users\lotof\Videos\research\live2d\yamatana-ai-ime\release\install-clean.log"
echo Install finished with code: %errorlevel%

echo 4. Performing self-test and starting tray...
set "EXE=C:\Program Files (x86)\Yamatana AI IME\ai_runtime\YamatanaAIIME.exe"
"%EXE%" --check
if %errorlevel% equ 0 (
    echo [SUCCESS] Startup self-test passed!
    start "" "%EXE%"
) else (
    echo [ERROR] Startup check failed
)

echo.
echo ===================================================
echo   Clean Reinstall Finished!
echo ===================================================
timeout /t 3
