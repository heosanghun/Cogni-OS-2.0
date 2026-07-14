[CmdletBinding()]
param(
    [string]$Treeish = 'HEAD',
    [string]$OutputDirectory,
    [switch]$RunModelSmoke,
    [string]$ModelPath = 'C:\Project\cognios\gemma4-e4b-it',
    [string]$ManualPdfPath
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = (Get-Command python -ErrorAction Stop).Source

if (
    $Treeish -notmatch '^[A-Za-z0-9._/-]{1,128}$' -or
    $Treeish.StartsWith('-') -or
    $Treeish.Contains('..')
) {
    throw 'Treeish contains unsupported characters.'
}

$commitOid = (& git -C $root rev-parse --verify --end-of-options "$Treeish^{commit}").Trim()
if ($LASTEXITCODE -ne 0 -or $commitOid -notmatch '^[0-9a-f]{40,64}$') {
    throw 'Treeish did not resolve to one immutable Git commit.'
}
$commitEpoch = (& git -C $root show -s --format=%ct $commitOid).Trim()
if ($LASTEXITCODE -ne 0 -or $commitEpoch -notmatch '^\d{9,12}$') {
    throw 'Could not determine the commit timestamp.'
}

function Get-LowerSha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-NoReparseAncestors([string]$Path) {
    $current = Get-Item -LiteralPath $Path -Force
    while ($null -ne $current) {
        if (($current.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Release path crosses a reparse point: $($current.FullName)"
        }
        $current = $current.Parent
    }
}

function Get-ArchiveEntrySha256([string]$ArchivePath, [string]$EntryName, [string]$Prefix) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($ArchivePath)
    try {
        $seen = New-Object 'Collections.Generic.HashSet[string]' ([StringComparer]::Ordinal)
        $matches = @()
        foreach ($entry in $zip.Entries) {
            $name = $entry.FullName
            if (
                -not $name.StartsWith($Prefix, [StringComparison]::Ordinal) -or
                $name.Contains('../') -or
                $name.Contains('..\') -or
                $name.StartsWith('/') -or
                $name.Contains(':')
            ) {
                throw "Unsafe source archive entry: $name"
            }
            if (-not $seen.Add($name)) {
                throw "Duplicate source archive entry: $name"
            }
            if ($name -ceq $EntryName) {
                $matches += $entry
            }
        }
        if ($matches.Count -ne 1) {
            throw "Archive must contain exactly one $EntryName entry."
        }
        $stream = $matches[0].Open()
        try {
            $sha = [Security.Cryptography.SHA256]::Create()
            try {
                $bytes = $sha.ComputeHash($stream)
            }
            finally {
                $sha.Dispose()
            }
        }
        finally {
            $stream.Dispose()
        }
        return ([BitConverter]::ToString($bytes)).Replace('-', '').ToLowerInvariant()
    }
    finally {
        $zip.Dispose()
    }
}

$workRoot = Join-Path $root 'work'
if (-not (Test-Path -LiteralPath $workRoot)) {
    New-Item -ItemType Directory -Path $workRoot | Out-Null
}
$workRoot = (Resolve-Path -LiteralPath $workRoot).Path
Assert-NoReparseAncestors $workRoot
$scratch = Join-Path $workRoot ('release-build-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $scratch | Out-Null

$publishStage = $null
$publishedOutput = $null
$releaseVersion = $null
$archiveCheckpoint = $null
$sourceArchiveName = $null
$buildSucceeded = $false
$oldSourceDateEpoch = $env:SOURCE_DATE_EPOCH
$oldPythonHashSeed = $env:PYTHONHASHSEED

try {
    $env:SOURCE_DATE_EPOCH = $commitEpoch
    $env:PYTHONHASHSEED = '0'
    $prefix = 'Cogni-OS-2-Genesis-source/'
    $rawArchive = Join-Path $scratch 'source.zip'
    & git -C $root -c core.autocrlf=false archive `
        --format=zip `
        --prefix=$prefix `
        --output=$rawArchive `
        $commitOid
    if ($LASTEXITCODE -ne 0) {
        throw 'git archive failed.'
    }

    $extractRoot = Join-Path $scratch 'extract'
    Expand-Archive -LiteralPath $rawArchive -DestinationPath $extractRoot
    $source = (Resolve-Path (
        Join-Path $extractRoot 'Cogni-OS-2-Genesis-source'
    )).Path
    Assert-NoReparseAncestors $source

    Push-Location $source
    try {
        $releaseVersion = (& $python -c (
            'from cogni_os.version import __version__;print(__version__)'
        )).Trim()
        if ($LASTEXITCODE -ne 0 -or $releaseVersion -notmatch '^\d+\.\d+\.\d+$') {
            throw 'Could not determine the version from the archived commit.'
        }
        $expectedCheckpoint = (& $python -c (
            'from cogni_core.cts_policy import DEFAULT_CHECKPOINT_SHA256;' +
            'print(DEFAULT_CHECKPOINT_SHA256)'
        )).Trim()
        if ($LASTEXITCODE -ne 0 -or $expectedCheckpoint -notmatch '^[0-9a-f]{64}$') {
            throw 'Could not load the archived CTS checkpoint trust root.'
        }
    }
    finally {
        Pop-Location
    }

    $checkpointEntry = $prefix + 'cogni_core/cts_policy_checkpoint.json'
    $archiveCheckpoint = Get-ArchiveEntrySha256 `
        $rawArchive $checkpointEntry $prefix
    if ($archiveCheckpoint -ne $expectedCheckpoint) {
        throw (
            'Source archive changed the CTS policy checkpoint bytes: ' +
            "$archiveCheckpoint != $expectedCheckpoint"
        )
    }
    $extractedCheckpoint = Get-LowerSha256 (
        Join-Path $source 'cogni_core\cts_policy_checkpoint.json'
    )
    if ($extractedCheckpoint -ne $expectedCheckpoint) {
        throw 'Extracted source changed the CTS policy checkpoint bytes.'
    }

    if ([String]::IsNullOrWhiteSpace($OutputDirectory)) {
        $OutputDirectory = Join-Path $root (
            "release\Cogni-OS-2-Genesis-v$releaseVersion"
        )
    }
    $publishedOutput = [IO.Path]::GetFullPath($OutputDirectory)
    if (Test-Path -LiteralPath $publishedOutput) {
        throw "Release output already exists; refusing to merge: $publishedOutput"
    }
    $outputParent = Split-Path -Parent $publishedOutput
    if (-not (Test-Path -LiteralPath $outputParent)) {
        New-Item -ItemType Directory -Path $outputParent | Out-Null
    }
    $outputParent = (Resolve-Path -LiteralPath $outputParent).Path
    Assert-NoReparseAncestors $outputParent
    $publishStage = Join-Path $outputParent (
        '.cogni-release-staging-' + [Guid]::NewGuid().ToString('N')
    )
    New-Item -ItemType Directory -Path $publishStage | Out-Null

    $sourceArchiveName = "Cogni-OS-2-Genesis-v$releaseVersion-source.zip"
    Move-Item -LiteralPath $rawArchive -Destination (
        Join-Path $publishStage $sourceArchiveName
    )

    # Preserve one clean expanded source tree beside the native launcher so
    # double-click discovery never depends on an untrusted external checkout.
    Copy-Item -LiteralPath $source -Destination $publishStage -Recurse
    $bundledCheckpoint = Get-LowerSha256 (
        Join-Path $publishStage (
            'Cogni-OS-2-Genesis-source\cogni_core\cts_policy_checkpoint.json'
        )
    )
    if ($bundledCheckpoint -ne $expectedCheckpoint) {
        throw 'Bundled expanded source changed the CTS policy checkpoint bytes.'
    }

    Push-Location $source
    try {
        & $python -c (
            'from cogni_core.cts_policy import load_default_bounded_cts_controller;' +
            'load_default_bounded_cts_controller(device=''cpu'');' +
            'print(''checkpoint_preflight=PASS'')'
        )
        if ($LASTEXITCODE -ne 0) {
            throw 'Extracted CTS checkpoint preflight failed.'
        }

        $wheelDirectory = Join-Path $scratch 'wheel'
        New-Item -ItemType Directory -Path $wheelDirectory | Out-Null
        & $python -m pip wheel . `
            --no-deps --no-build-isolation --wheel-dir $wheelDirectory
        if ($LASTEXITCODE -ne 0) {
            throw 'Wheel build failed.'
        }
        $wheelName = "cogni_os-$releaseVersion-py3-none-any.whl"
        Copy-Item -LiteralPath (Join-Path $wheelDirectory $wheelName) `
            -Destination (Join-Path $publishStage $wheelName)

        $exeName = "CogniBoard-v$releaseVersion.exe"
        & powershell -NoProfile -ExecutionPolicy Bypass `
            -File (Join-Path $source 'scripts\build_windows_launcher.ps1') `
            -OutputPath (Join-Path $publishStage $exeName)
        if ($LASTEXITCODE -ne 0) {
            throw 'Windows launcher build failed.'
        }

        if ($RunModelSmoke) {
            if (-not (Test-Path -LiteralPath $ModelPath -PathType Container)) {
                throw "Model path does not exist: $ModelPath"
            }
            & $python -u scripts\validate_agent_runtime.py `
                --model $ModelPath `
                --manifest config\gemma4-e4b-it.manifest.toml `
                --max-new-tokens 96
            if ($LASTEXITCODE -ne 0) {
                throw 'Extracted bundle model smoke failed.'
            }
        }
    }
    finally {
        Pop-Location
    }

    foreach ($name in @(
        "COGNI_OS_$releaseVersion`_RELEASE_NOTES_KO.md",
        "COGNI_OS_$releaseVersion`_VALIDATION_ADDENDUM_KO.md"
    )) {
        $candidate = Join-Path $source (Join-Path 'release' $name)
        if (Test-Path -LiteralPath $candidate) {
            Copy-Item -LiteralPath $candidate -Destination (
                Join-Path $publishStage $name
            )
        }
    }
    $manual = Join-Path $source 'docs\COGNIBOARD_USER_MANUAL_PLAYBOOK_KO.md'
    if (Test-Path -LiteralPath $manual) {
        Copy-Item -LiteralPath $manual -Destination (
            Join-Path $publishStage 'COGNIBOARD_USER_MANUAL_PLAYBOOK_KO.md'
        )
    }
    if (-not [String]::IsNullOrWhiteSpace($ManualPdfPath)) {
        $manualPdf = [IO.Path]::GetFullPath($ManualPdfPath)
        if (-not (Test-Path -LiteralPath $manualPdf -PathType Leaf)) {
            throw "Manual PDF does not exist: $manualPdf"
        }
        Assert-NoReparseAncestors (Split-Path -Parent $manualPdf)
        Copy-Item -LiteralPath $manualPdf -Destination (
            Join-Path $publishStage (
                "COGNIBOARD_USER_MANUAL_PLAYBOOK_KO_v$releaseVersion.pdf"
            )
        )
    }

    $pythonVersion = (& $python --version 2>&1 | Out-String).Trim()
    $pipVersion = (& $python -m pip --version 2>&1 | Out-String).Trim()
    $manifestLines = @(
        "release_version=$releaseVersion",
        "commit_oid=$commitOid",
        "commit_epoch=$commitEpoch",
        "source_archive=$sourceArchiveName",
        "source_archive_sha256=$(Get-LowerSha256 (Join-Path $publishStage $sourceArchiveName))",
        "cts_checkpoint_sha256=$archiveCheckpoint",
        "python_executable=$python",
        "python_version=$pythonVersion",
        "pip_version=$pipVersion",
        "source_date_epoch=$commitEpoch"
    )
    $utf8 = New-Object Text.UTF8Encoding($false)
    [IO.File]::WriteAllLines(
        (Join-Path $publishStage 'BUILD_MANIFEST.txt'),
        $manifestLines,
        $utf8
    )

    $checksumTargets = Get-ChildItem -LiteralPath $publishStage -File |
        Where-Object { $_.Name -ne 'SHA256SUMS.txt' } |
        Sort-Object Name
    $checksumLines = foreach ($file in $checksumTargets) {
        '{0}  {1}' -f (Get-LowerSha256 $file.FullName), $file.Name
    }
    [IO.File]::WriteAllLines(
        (Join-Path $publishStage 'SHA256SUMS.txt'),
        $checksumLines,
        $utf8
    )

    Move-Item -LiteralPath $publishStage -Destination $publishedOutput
    $publishStage = $null
    $buildSucceeded = $true
}
finally {
    $env:SOURCE_DATE_EPOCH = $oldSourceDateEpoch
    $env:PYTHONHASHSEED = $oldPythonHashSeed
    try {
        if ($null -ne $publishStage -and (Test-Path -LiteralPath $publishStage)) {
            $resolvedStage = (Resolve-Path -LiteralPath $publishStage).Path
            Assert-NoReparseAncestors $resolvedStage
            Remove-Item -LiteralPath $resolvedStage -Recurse -Force
        }
        if (Test-Path -LiteralPath $scratch) {
            $resolvedScratch = (Resolve-Path -LiteralPath $scratch).Path
            if (-not $resolvedScratch.StartsWith(
                $workRoot + '\',
                [StringComparison]::OrdinalIgnoreCase
            )) {
                throw "Unsafe scratch cleanup target: $resolvedScratch"
            }
            Assert-NoReparseAncestors $resolvedScratch
            Remove-Item -LiteralPath $resolvedScratch -Recurse -Force
        }
    }
    catch {
        if ($buildSucceeded) {
            throw
        }
        Write-Warning ("Release cleanup also failed: " + $_.Exception.Message)
    }
}

Write-Output "release_output=$publishedOutput"
Write-Output "source_archive=$sourceArchiveName"
Write-Output "checkpoint_sha256=$archiveCheckpoint"
Write-Output "commit_oid=$commitOid"
Write-Output "release_version=$releaseVersion"
Write-Output 'release_bundle=PASS'
