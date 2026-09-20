@echo off
chcp 65001 >nul
setlocal
set "INSTALLER=%~dp0install-fresh.ps1"
if not exist "%INSTALLER%" (
  echo [エラー] install-fresh.ps1 がありません。ZIP全体を「すべて展開」してから実行してください。
  pause
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%INSTALLER%"
set "CODE=%ERRORLEVEL%"
echo.
if "%CODE%"=="0" echo インストールが完了しました。サインアウトまたはPCの再起動後、Win + SpaceでFumiori AI IMEを選択してください。
if "%CODE%"=="3010" echo インストールは成功しました。PCの再起動が必要です。再起動後、Win + SpaceでFumiori AI IMEを選択してください。
if not "%CODE%"=="0" if not "%CODE%"=="3010" echo インストールを完了できませんでした。画面に表示されたログの場所をご確認ください。終了コード: %CODE%
echo.
pause
exit /b %CODE%
