# PowerShell script to update installed Yamatana AI IME models and restart the AI ranker

$sourceDual = "c:\Users\lotof\Videos\research\live2d\yamatana-ai-ime\build\onnx-model-70m-dual-encoder"
$sourceLora3 = "c:\Users\lotof\Videos\research\live2d\yamatana-ai-ime\build\onnx-model-70m-lora3-20260909"
$sourceLora6 = "c:\Users\lotof\Videos\research\live2d\yamatana-ai-ime\build\onnx-model-70m-lora6-preceding-only-20260915"
$tokenizerSource = "c:\Users\lotof\Videos\research\live2d\yamatana-ai-ime\build\onnx-model-70m-dual-encoder\tokenizer.json"
$targetDir = "C:\Program Files (x86)\Yamatana AI IME\ai_runtime\_internal\models\onnx"
$exePath = "C:\Program Files (x86)\Yamatana AI IME\ai_runtime\YamatanaAIIME.exe"

$requiredSources = @(
    "$sourceDual\dual-encoder-70m-fp16.onnx",
    "$sourceDual\dual-encoder-70m-int8.onnx",
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

Write-Host "Copying updated executable and runtime..." -ForegroundColor Cyan
$distDir = "c:\Users\lotof\Videos\research\live2d\yamatana-ai-ime\dist\YamatanaAIIME"
$runtimeDir = "C:\Program Files (x86)\Yamatana AI IME\ai_runtime"
if (Test-Path -LiteralPath "$distDir\YamatanaAIIME.exe") {
    Copy-Item "$distDir\YamatanaAIIME.exe" -Destination "$runtimeDir\YamatanaAIIME.exe" -Force
    robocopy "$distDir\_internal" "$runtimeDir\_internal" /E /R:2 /W:1 /NJH /NJS /NDL /NC /NS | Out-Null
}

Write-Host "Copying updated Dual-Encoder and ONNX models to $targetDir..." -ForegroundColor Cyan
Copy-Item "$sourceDual\dual-encoder-70m-fp16.onnx" -Destination "$targetDir\dual-encoder-70m-fp16.onnx" -Force
Copy-Item "$sourceDual\dual-encoder-70m-int8.onnx" -Destination "$targetDir\dual-encoder-70m-int8.onnx" -Force
Copy-Item "$sourceLora3\ruri-ime-fp16.onnx" -Destination "$targetDir\ruri-ime-lora3-fp16.onnx" -Force
Copy-Item "$sourceLora3\ruri-ime-int8.onnx" -Destination "$targetDir\ruri-ime-lora3-int8.onnx" -Force
Copy-Item "$sourceLora6\ruri-ime-fp16.onnx" -Destination "$targetDir\ruri-ime-lora6-fp16.onnx" -Force
Copy-Item "$sourceLora6\ruri-ime-int8.onnx" -Destination "$targetDir\ruri-ime-lora6-int8.onnx" -Force
Copy-Item "$tokenizerSource" -Destination "$targetDir\tokenizer.json" -Force

Write-Host "Verifying executable startup..." -ForegroundColor Cyan
$proc = Start-Process -FilePath $exePath -ArgumentList "--check" -Wait -PassThru
if ($proc.ExitCode -ne 0) {
    throw "YamatanaAIIME.exe startup self-test failed with code $($proc.ExitCode)"
}
Write-Host "Files updated successfully:" -ForegroundColor Green
Get-ChildItem -Path $targetDir | Select-Object Name, Length, LastWriteTime | Format-Table

Write-Host "Restarting YamatanaAIIME..." -ForegroundColor Cyan
Start-Process -FilePath $exePath

Write-Host "Done! AI IME is running." -ForegroundColor Green
