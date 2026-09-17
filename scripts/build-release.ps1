[CmdletBinding()]
param(
  [string]$ProductVersion = '2.0.10.0',
  [string]$ReleaseLabel = '2.0.10-beta',
  [switch]$SkipMozcDependencies
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location $RepoRoot
try {
  # Set these before PyInstaller so the EXE file version advances together
  # with the MSI ProductVersion. Windows Installer otherwise retains an older
  # same-version runtime during a major upgrade.
  $env:YAMATANA_PRODUCT_VERSION = $ProductVersion
  $env:YAMATANA_RELEASE_LABEL = $ReleaseLabel
  & (Join-Path $PSScriptRoot 'fetch-model.ps1')
  if (-not $?) { throw 'Model fetch failed' }

  python -m pip install --disable-pip-version-check -r requirements-build.txt
  if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed' }
  $env:YAMATANA_HEADLESS_TESTS = '1'
  python -m pytest -q
  if ($LASTEXITCODE -ne 0) { throw 'Python tests failed' }
  python -m PyInstaller --noconfirm --clean ai_ime_tray.spec
  if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed' }

  & (Join-Path $PSScriptRoot 'prepare_mozc_source.ps1') -SkipDependencyDownload:$SkipMozcDependencies
  if (-not $?) { throw 'Mozc source preparation failed' }
  $MozcSrc = Join-Path $RepoRoot 'build\mozc-src\src'
  Push-Location $MozcSrc
  try {
    python build_tools/build_qt.py --release --confirm_license --target_arch=x64
    if ($LASTEXITCODE -ne 0) { throw 'Qt build failed' }
    $Bazel = Get-Command bazelisk -ErrorAction SilentlyContinue
    if (-not $Bazel) { $Bazel = Get-Command bazel -ErrorAction Stop }
    & $Bazel.Source test '//rewriter:ai_rewriter_test' '--config=release_build' '--platforms=//:windows-x86_64'
    if ($LASTEXITCODE -ne 0) { throw 'Mozc AI rewriter test failed' }
    $Targets = @(
      '//win32/tip:mozc_tip32',
      '//win32/tip:mozc_tip64',
      '//win32/broker:mozc_broker_main',
      '//server:mozc_server_win',
      '//win32/cache_service:mozc_cache_service',
      '//renderer/win32:win32_renderer_main',
      '//gui/tool:mozc_tool_win',
      '//win32/custom_action:custom_action'
    )
    & $Bazel.Source build @Targets '--config=release_build' '--platforms=//:windows-x86_64'
    if ($LASTEXITCODE -ne 0) { throw 'Mozc component build failed' }
  } finally { Pop-Location }

  if (-not (Get-Command wix -ErrorAction SilentlyContinue)) {
    dotnet tool install --global wix --version 4.0.5
    if ($LASTEXITCODE -ne 0) { throw 'WiX installation failed' }
  }
  wix extension add WixToolset.UI.wixext/4.0.5 --global
  if ($LASTEXITCODE -ne 0) { throw 'WiX UI extension installation failed' }

  python scripts/build_ai_msi.py
  if ($LASTEXITCODE -ne 0) { throw 'MSI build failed' }

  $Msi = Join-Path $RepoRoot "release\Yamatana-AI-IME-MOZC-Ver-$ReleaseLabel-x64.msi"
  $Admin = Join-Path $RepoRoot 'build\admin-image'
  New-Item -ItemType Directory -Force -Path $Admin | Out-Null
  $Process = Start-Process msiexec.exe -Wait -PassThru -ArgumentList @('/a', $Msi, "TARGETDIR=$Admin", '/qn')
  if ($Process.ExitCode -ne 0) { throw "MSI administrative extraction failed: $($Process.ExitCode)" }
  python scripts/validate_distribution.py $Msi $Admin
  if ($LASTEXITCODE -ne 0) { throw 'MSI validation failed' }

  $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Msi).Hash
  $Line = "$Hash  $([IO.Path]::GetFileName($Msi))`n"
  [IO.File]::WriteAllText((Join-Path $RepoRoot 'release\SHA256SUMS.txt'), $Line, [Text.UTF8Encoding]::new($false))
  Copy-Item (Join-Path $PSScriptRoot 'install-msi.ps1') (Join-Path $RepoRoot 'release\install-msi.ps1') -Force
  Copy-Item (Join-Path $PSScriptRoot 'Install-Yamatana-AI-IME.cmd') (Join-Path $RepoRoot 'release\Install-Yamatana-AI-IME.cmd') -Force
  Write-Host "RELEASE_BUILD_PASS: $Msi $Hash"
} finally {
  Pop-Location
}
