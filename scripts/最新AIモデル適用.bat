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

set "SRC3=c:\Users\lotof\Videos\research\live2d\yamatana-ai-ime\build\onnx-model-70m-lora3-20260909"
set "SRC6=c:\Users\lotof\Videos\research\live2d\yamatana-ai-ime\build\onnx-model-70m-lora6-preceding-only-20260915"
set "TOK=c:\Users\lotof\Videos\research\live2d\yamatana-ai-ime\build\onnx-model-70m\tokenizer.json"
set "DST=C:\Program Files (x86)\Yamatana AI IME\ai_runtime\_internal\models\onnx"
set "EXE=C:\Program Files (x86)\Yamatana AI IME\ai_runtime\YamatanaAIIME.exe"

echo 1. 稼働中のYamatanaAIIMEを停止中...
taskkill /F /IM YamatanaAIIME.exe 2>nul
timeout /t 2 /nobreak > nul

echo 2. 最新ONNXモデルをコピー中...
copy /Y "%SRC3%\ruri-ime-fp16.onnx" "%DST%\ruri-ime-lora3-fp16.onnx"
copy /Y "%SRC3%\ruri-ime-int8.onnx" "%DST%\ruri-ime-lora3-int8.onnx"
copy /Y "%SRC6%\ruri-ime-fp16.onnx" "%DST%\ruri-ime-lora6-fp16.onnx"
copy /Y "%SRC6%\ruri-ime-int8.onnx" "%DST%\ruri-ime-lora6-int8.onnx"
copy /Y "%TOK%" "%DST%\tokenizer.json"

if %errorlevel% equ 0 (
    echo.
    echo [成功] 最新の70Mモデル（正答率96.7%%）を正常に配置しました！
) else (
    echo.
    echo [エラー] コピーに失敗しました。
    pause
    exit /b 1
)

echo 3. YamatanaAIIMEを再起動中...
start "" "%EXE%"

echo.
echo ===================================================
echo   適用完了！最新AIモデルが読み込まれました。
echo ===================================================
timeout /t 3
