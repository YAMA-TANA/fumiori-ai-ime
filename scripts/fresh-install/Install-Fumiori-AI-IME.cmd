@echo off
setlocal
set "INSTALLER=%~dp0install-fresh.ps1"
if not exist "%INSTALLER%" (
  echo [ERROR] install-fresh.ps1 was not found. Extract the entire setup ZIP before running this file.
  pause
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%INSTALLER%"
set "CODE=%ERRORLEVEL%"
echo.
if "%CODE%"=="0" echo Fumiori AI IME installation completed. Sign out or restart Windows before selecting the IME.
if "%CODE%"=="3010" echo Fumiori AI IME installation completed. RESTART YOUR PC to finish setup.
if not "%CODE%"=="0" if not "%CODE%"=="3010" echo Installation could not be completed. Please check the installer log shown above. Exit code: %CODE%
echo.
pause
exit /b %CODE%
