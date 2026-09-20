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

  # The script is copied into the same folder as the MSI for distribution;
  # the source-tree build also keeps its MSI under ../release.
  $locations = @($PSScriptRoot, (Join-Path $PSScriptRoot '..\release'))
  foreach ($directory in $locations) {
    if (-not (Test-Path -LiteralPath $directory -PathType Container)) { continue }
    $latest = Get-ChildItem -LiteralPath $directory -Filter 'Yamatana-AI-IME-MOZC-Ver-*.msi' -File |
      Sort-Object LastWriteTime -Descending |
      Select-Object -First 1
    if ($latest) { return $latest.FullName }
  }
  throw "インストールするMSIが見つかりません。次のフォルダーを確認してください: $($locations -join ', ')"
}

function Get-InstallMessage {
  param([int]$ExitCode)

  switch ($ExitCode) {
    0    { return 'インストールが完了しました。IMEが表示されない場合はサインアウトまたは再起動してください。' }
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

Write-Host "Fumiori AI IME をインストールします: $resolvedMsi"
Write-Host "インストールログ: $logPath"

$process = Start-Process -FilePath $msiexec -Wait -PassThru -ArgumentList @(
  '/i', ('"' + $resolvedMsi + '"'),
  '/passive',
  '/norestart',
  '/L*v', ('"' + $logPath + '"')
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
