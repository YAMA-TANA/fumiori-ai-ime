@echo off
setlocal

set "SCRIPT=%~dp0install-msi.ps1"
if not exist "%SCRIPT%" (
  echo install-msi.ps1 が見つかりません。
  pause
  exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" %*
set "CODE=%ERRORLEVEL%"

echo.
if "%CODE%"=="0" echo インストールが完了しました。
if "%CODE%"=="3010" echo インストールは成功しました。PCの再起動が必要です。
if "%CODE%"=="1641" echo インストールは成功しました。PCの再起動が必要です。
if not "%CODE%"=="0" if not "%CODE%"=="3010" if not "%CODE%"=="1641" echo MSI終了コード %CODE% で終了しました。ログを確認してください。
echo.
pause
exit /b %CODE%
