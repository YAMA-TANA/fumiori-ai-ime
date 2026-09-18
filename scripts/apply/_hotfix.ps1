[CmdletBinding()]
param()

$canonical = (Resolve-Path (Join-Path $PSScriptRoot '..\apply_hotfix.ps1')).Path
& $canonical
exit $LASTEXITCODE
