[CmdletBinding()]
param([switch]$SkipDependencyDownload)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Config = Get-Content -Raw -LiteralPath (Join-Path $RepoRoot 'build-config.json') | ConvertFrom-Json
$Checkout = Join-Path $RepoRoot 'build\mozc-src'

if (-not (Test-Path -LiteralPath (Join-Path $Checkout '.git'))) {
  New-Item -ItemType Directory -Force -Path (Split-Path $Checkout -Parent) | Out-Null
  & git clone --filter=blob:none $Config.mozc.repository $Checkout
  if ($LASTEXITCODE -ne 0) { throw 'Mozc clone failed' }
}

& git -C $Checkout fetch origin $Config.mozc.commit --depth=1
if ($LASTEXITCODE -ne 0) { throw 'Mozc fetch failed' }
& git -C $Checkout checkout --detach $Config.mozc.commit
if ($LASTEXITCODE -ne 0) { throw 'Mozc checkout failed' }

# The checkout is a disposable build workspace. Reset tracked files before
# copying the current overlay; otherwise repeated local builds can accumulate
# duplicate C++ helpers and make patch matching ambiguous. Keep downloaded
# dependencies and Bazel outputs so subsequent local release builds stay fast.
& git -C $Checkout reset --hard $Config.mozc.commit
if ($LASTEXITCODE -ne 0) { throw 'Mozc checkout reset failed' }

$ActualCommit = (& git -C $Checkout rev-parse HEAD).Trim()
if ($ActualCommit -ne $Config.mozc.commit) { throw "Mozc commit mismatch: $ActualCommit" }

$Overlay = Join-Path $RepoRoot 'mozc\overlay'
Get-ChildItem -LiteralPath $Overlay -File -Recurse | ForEach-Object {
  $relative = $_.FullName.Substring($Overlay.Length + 1)
  $destination = Join-Path $Checkout $relative
  New-Item -ItemType Directory -Force -Path (Split-Path $destination -Parent) | Out-Null
  Copy-Item -LiteralPath $_.FullName -Destination $destination -Force
}

python (Join-Path $RepoRoot 'scripts\patch_mozc_surrounding_context.py') --checkout $Checkout
if ($LASTEXITCODE -ne 0) { throw 'Mozc surrounding context patch failed' }

python (Join-Path $RepoRoot 'scripts\patch_mozc_ai_prefetch.py') --checkout $Checkout
if ($LASTEXITCODE -ne 0) { throw 'Mozc AI prefetch patch failed' }

python (Join-Path $RepoRoot 'scripts\patch_mozc_installer_tray.py') --checkout $Checkout
if ($LASTEXITCODE -ne 0) { throw 'Mozc installer tray patch failed' }

python (Join-Path $RepoRoot 'scripts\modernize_mozc_ui.py') --checkout $Checkout
if ($LASTEXITCODE -ne 0) { throw 'Mozc modern UI patch failed' }

$Vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $Vswhere)) { throw 'vswhere.exe not found' }
$VsRoot = (& $Vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.ATLMFC -property installationPath).Trim()
if (-not $VsRoot) { throw 'Visual Studio ATL/MFC component not found' }
$Msvc = Get-ChildItem -LiteralPath (Join-Path $VsRoot 'VC\Tools\MSVC') -Directory | Sort-Object Name -Descending | Select-Object -First 1
if (-not $Msvc) { throw 'MSVC tools directory not found' }
$Atl = Join-Path $Msvc.FullName 'atlmfc'
$AtlHeaders = Join-Path $Checkout 'src\base\win32\atl_include'
$AtlLib = Join-Path $Checkout 'src\base\win32\atl_lib'
New-Item -ItemType Directory -Force -Path $AtlHeaders,(Join-Path $AtlLib 'x86') | Out-Null
Copy-Item -Path (Join-Path $Atl 'include\*') -Destination $AtlHeaders -Recurse -Force
Copy-Item -LiteralPath (Join-Path $Atl 'lib\x64\atls.lib') -Destination (Join-Path $AtlLib 'atls.lib') -Force
Copy-Item -LiteralPath (Join-Path $Atl 'lib\x86\atls.lib') -Destination (Join-Path $AtlLib 'x86\atls.lib') -Force

if (-not $SkipDependencyDownload) {
  Push-Location (Join-Path $Checkout 'src')
  try {
    python build_tools/update_deps.py
    if ($LASTEXITCODE -ne 0) { throw 'Mozc dependency download failed' }
  } finally { Pop-Location }
}

Write-Host "MOZC_SOURCE_READY: $ActualCommit"
