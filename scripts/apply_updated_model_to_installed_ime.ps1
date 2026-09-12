# PowerShell script to update installed Yamatana AI IME models and restart the AI ranker

$sourceLora3 = "c:\Users\lotof\Videos\research\live2d\yamatana-ai-ime\build\onnx-model-70m-lora3-20260909"
$sourceLora6 = "c:\Users\lotof\Videos\research\live2d\yamatana-ai-ime\build\onnx-model-70m-lora6-preceding-only-20260915"
$tokenizerSource = "c:\Users\lotof\Videos\research\live2d\yamatana-ai-ime\build\onnx-model-70m\tokenizer.json"
$targetDir = "C:\Program Files (x86)\Yamatana AI IME\ai_runtime\_internal\models\onnx"
$exePath = "C:\Program Files (x86)\Yamatana AI IME\ai_runtime\YamatanaAIIME.exe"

$requiredSources = @(
    "$sourceLora3\ruri-ime-fp16.onnx",
    "$sourceLora3\ruri-ime-int8.onnx",
    "$sourceLora6\ruri-ime-fp16.onnx",
    "$sourceLora6\ruri-ime-int8.onnx",
    $tokenizerSource
)
foreach ($source in $requiredSources) {
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Required model source is missing: $source"
    }
}
if (-not (Test-Path -LiteralPath $targetDir)) {
    throw "Installed model directory is missing: $targetDir"
}

Write-Host "Stopping running YamatanaAIIME processes..." -ForegroundColor Cyan
Get-Process -Name "YamatanaAIIME" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

Write-Host "Copying updated ONNX models to $targetDir..." -ForegroundColor Cyan
Copy-Item "$sourceLora3\ruri-ime-fp16.onnx" -Destination "$targetDir\ruri-ime-lora3-fp16.onnx" -Force
Copy-Item "$sourceLora3\ruri-ime-int8.onnx" -Destination "$targetDir\ruri-ime-lora3-int8.onnx" -Force
Copy-Item "$sourceLora6\ruri-ime-fp16.onnx" -Destination "$targetDir\ruri-ime-lora6-fp16.onnx" -Force
Copy-Item "$sourceLora6\ruri-ime-int8.onnx" -Destination "$targetDir\ruri-ime-lora6-int8.onnx" -Force
Copy-Item "$tokenizerSource" -Destination "$targetDir\tokenizer.json" -Force

Write-Host "Files updated successfully:" -ForegroundColor Green
Get-ChildItem -Path $targetDir | Select-Object Name, Length, LastWriteTime | Format-Table

Write-Host "Restarting YamatanaAIIME..." -ForegroundColor Cyan
Start-Process -FilePath $exePath

Write-Host "Done!" -ForegroundColor Green
