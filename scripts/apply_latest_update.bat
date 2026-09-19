@echo off
setlocal
echo ========================================================
echo   Applying Latest Dual-Encoder Runtime Hotfix
echo ========================================================
echo.

:: Check Admin
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Elevating to Administrator...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process cmd -ArgumentList '/c \"\"%~f0\"\"' -Verb RunAs"
    exit /b
)

echo 1. Stopping running processes...
taskkill /F /IM YamatanaAIIME.exe 2>nul
taskkill /F /IM mozc_server.exe 2>nul
timeout /t 2 /nobreak > nul

set "SRC_DIR=c:\Users\lotof\Videos\research\live2d\yamatana-ai-ime\dist\YamatanaAIIME"
set "DST_DIR=C:\Program Files (x86)\Yamatana AI IME\ai_runtime"

echo 2. Copying YamatanaAIIME.exe...
copy /Y "%SRC_DIR%\YamatanaAIIME.exe" "%DST_DIR%\YamatanaAIIME.exe"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to copy YamatanaAIIME.exe! File might be locked.
    pause
    exit /b 1
)

echo 3. Syncing _internal directory...
robocopy "%SRC_DIR%\_internal" "%DST_DIR%\_internal" /E /R:2 /W:1 /NJH /NJS /NDL /NC /NS >nul 2>&1

echo 4. Verifying copied binary...
dir "%DST_DIR%\YamatanaAIIME.exe" | findstr YamatanaAIIME.exe

echo 5. Performing startup self-test...
"%DST_DIR%\YamatanaAIIME.exe" --check
if %errorlevel% neq 0 (
    echo [ERROR] Binary startup check failed!
    pause
    exit /b 1
)

echo 6. Starting YamatanaAIIME...
start "" "%DST_DIR%\YamatanaAIIME.exe"

echo.
echo ========================================================
echo   SUCCESS! Dual-Encoder Batch Runtime Applied!
echo ========================================================
timeout /t 4
