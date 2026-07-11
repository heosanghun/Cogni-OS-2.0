[CmdletBinding()]
param(
    [string]$OutputPath
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$source = Join-Path $root 'launcher\CogniBoardLauncher.cs'
if (-not $OutputPath) {
    $OutputPath = Join-Path $root 'CogniBoard.exe'
}
$parent = Split-Path -Parent $OutputPath
if ($parent -and -not (Test-Path -LiteralPath $parent)) {
    New-Item -ItemType Directory -Path $parent | Out-Null
}
if (Test-Path -LiteralPath $OutputPath) {
    Remove-Item -LiteralPath $OutputPath -Force
}

Add-Type `
    -Path $source `
    -OutputAssembly $OutputPath `
    -OutputType WindowsApplication `
    -ReferencedAssemblies 'System.dll','System.Windows.Forms.dll'

$artifact = Get-Item -LiteralPath $OutputPath
if ($artifact.Length -le 0) {
    throw 'Launcher compilation produced an empty artifact.'
}
Write-Output $artifact.FullName
