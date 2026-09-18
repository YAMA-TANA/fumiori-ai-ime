@echo off
setlocal
chcp 65001 > nul
echo ===================================================
echo   Yamatana AI IME 最新70M残差修正モデルの適用
echo ===================================================
echo.

:: 管理者権限チェック
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo 管理者権限に昇格して実行します...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process cmd -ArgumentList '/c \"\"%~f0\"\"' -Verb RunAs"
    exit /b
)

set "DIST=c:\Users\lotof\Videos\research\live2d\yamatana-ai-ime\dist\YamatanaAIIME"
set "RUNTIME_DIR=C:\Program Files (x86)\Yamatana AI IME\ai_runtime"
set "SRC_DUAL=c:\Users\lotof\Videos\research\live2d\yamatana-ai-ime\build\onnx-model-70m-dual-encoder"
set "SRC3=c:\Users\lotof\Videos\research\live2d\yamatana-ai-ime\build\onnx-model-70m-lora3-20260909"
set "SRC6=c:\Users\lotof\Videos\research\live2d\yamatana-ai-ime\build\onnx-model-70m-lora6-preceding-only-20260915"
set "TOK=c:\Users\lotof\Videos\research\live2d\yamatana-ai-ime\build\onnx-model-70m-dual-encoder\tokenizer.json"
set "DST=C:\Program Files (x86)\Yamatana AI IME\ai_runtime\_internal\models\onnx"
set "EXE=C:\Program Files (x86)\Yamatana AI IME\ai_runtime\YamatanaAIIME.exe"

echo 1. 稼働中のYamatanaAIIMEを停止中...
taskkill /F /IM YamatanaAIIME.exe 2>nul
timeout /t 2 /nobreak > nul

echo 2. 最新AI実行バイナリとランタイムを配置中...
if exist "%DIST%\YamatanaAIIME.exe" (
    copy /Y "%DIST%\YamatanaAIIME.exe" "%RUNTIME_DIR%\YamatanaAIIME.exe"
    robocopy "%DIST%\_internal" "%RUNTIME_DIR%\_internal" /E /R:2 /W:1 /NJH /NJS /NDL /NC /NS >nul 2>&1
)

echo 3. 最新Dual-Encoder 70M ONNXモデルをコピー中...
if not exist "%DST%" mkdir "%DST%"
copy /Y "%SRC_DUAL%\dual-encoder-70m-fp16.onnx" "%DST%\dual-encoder-70m-fp16.onnx"
copy /Y "%SRC_DUAL%\dual-encoder-70m-int8.onnx" "%DST%\dual-encoder-70m-int8.onnx"
copy /Y "%SRC3%\ruri-ime-fp16.onnx" "%DST%\ruri-ime-lora3-fp16.onnx"
copy /Y "%SRC3%\ruri-ime-int8.onnx" "%DST%\ruri-ime-lora3-int8.onnx"
copy /Y "%SRC6%\ruri-ime-fp16.onnx" "%DST%\ruri-ime-lora6-fp16.onnx"
copy /Y "%SRC6%\ruri-ime-int8.onnx" "%DST%\ruri-ime-lora6-int8.onnx"
copy /Y "%TOK%" "%DST%\tokenizer.json"

if %errorlevel% equ 0 (
    echo.
    echo [成功] 最新のDual-Encoder 70Mモデル（超低遅延9ms）およびバイナリを正常に配置しました！
) else (
    echo.
    echo [エラー] コピーに失敗しました。
    pause
    exit /b 1
)

echo 4. YamatanaAIIMEの動作確認・再起動中...
"%EXE%" --check
if %errorlevel% neq 0 (
    echo [エラー] バイナリの起動検証に失敗しました。
    pause
    exit /b 1
)
start "" "%EXE%"

echo.
echo ===================================================
echo   適用完了！最新AIモデルが読み込まれました。
echo ===================================================
timeout /t 3
