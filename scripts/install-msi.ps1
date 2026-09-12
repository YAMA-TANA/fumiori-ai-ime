[CmdletBinding()]
param(
  [Parameter(Position = 0)]
  [string]$MsiPath
)

$ErrorActionPreference = 'Stop'

function Resolve-MsiPath {
  param([string]$RequestedPath)

  if ($RequestedPath) {
    $candidate = Resolve-Path -LiteralPath $RequestedPath -ErrorAction Stop
    if ($candidate.Path -notlike '*.msi') {
      throw "MSIファイルを指定してください: $($candidate.Path)"
    }
    return $candidate.Path
  }

  $releaseDir = Join-Path $PSScriptRoot '..\release'
  $latest = Get-ChildItem -LiteralPath $releaseDir -Filter 'Yamatana-AI-IME-MOZC-Ver-*.msi' -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
  if (-not $latest) {
    throw "インストールするMSIが見つかりません: $releaseDir"
  }
  return $latest.FullName
}

function Get-InstallMessage {
  param([int]$ExitCode)

  switch ($ExitCode) {
    0    { return 'インストールが完了しました。' }
    3010 { return 'インストールは成功しました。PCの再起動が必要です。' }
    1641 { return 'インストールは成功しました。PCの再起動が必要です。' }
    1602 { return 'インストールはキャンセルされました。' }
    1603 { return 'インストールに失敗しました（1603）。ログを確認してください。' }
    default { return "インストールに失敗しました（MSI終了コード: $ExitCode）。ログを確認してください。" }
  }
}

$resolvedMsi = Resolve-MsiPath $MsiPath
$logPath = Join-Path (Split-Path -Parent $resolvedMsi) 'Yamatana-AI-IME-install.log'
$msiexec = Join-Path $env:WINDIR 'System32\msiexec.exe'

Write-Host "Yamatana AI IME をインストールします: $resolvedMsi"
Write-Host "インストールログ: $logPath"

$process = Start-Process -FilePath $msiexec -Wait -PassThru -ArgumentList @(
  '/i', $resolvedMsi,
  '/passive',
  '/norestart',
  '/L*v', $logPath
)
$exitCode = [int]$process.ExitCode

if ($exitCode -in @(0, 3010, 1641)) {
  if ($exitCode -eq 0) {
    Write-Host (Get-InstallMessage $exitCode) -ForegroundColor Green
  } else {
    Write-Host (Get-InstallMessage $exitCode) -ForegroundColor Yellow
    Write-Host '再起動するまで、IMEの登録変更が一部反映されない場合があります。' -ForegroundColor Yellow
  }
} else {
  Write-Host (Get-InstallMessage $exitCode) -ForegroundColor Red
}
Write-Host "MSI終了コード: $exitCode"

exit $exitCode
