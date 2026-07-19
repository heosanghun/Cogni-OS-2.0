[CmdletBinding()]
param(
    [string]$Treeish = 'HEAD',
    [string]$OutputDirectory,
    [switch]$PublishRelease,
    [string]$ReleaseEvidenceSummaryPath,
    [string]$ReleaseEvidenceSummarySha256,
    [string]$CpuGateEvidencePath,
    [string]$CpuGateEvidenceSha256,
    [string]$Gpu5GateEvidencePath,
    [string]$Gpu5GateEvidenceSha256,
    [string]$ReleaseAttestationPath,
    [string]$ReleaseAttestationSha256,
    [string]$ReleaseAttestationSignaturePath,
    [string]$ReleaseAttestationSignatureSha256,
    [string]$VerifierPublicKeyPath,
    [string]$RuntimeEvidencePath,
    [string]$RuntimeEvidenceSha256,
    [string]$CompletionEvidencePath,
    [string]$CompletionEvidenceSha256,
    [string]$IdentityPreEvidencePath,
    [string]$IdentityPreEvidenceSha256,
    [string]$IdentityPostEvidencePath,
    [string]$IdentityPostEvidenceSha256,
    [string]$ConfigEvidencePath,
    [string]$ConfigEvidenceSha256,
    [string]$DeviceEvidencePath,
    [string]$DeviceEvidenceSha256,
    [string]$ModelInventoryPath,
    [string]$ModelInventorySha256,
    [string]$AcceptanceBundleRoot,
    [string]$ManualPdfPath,
    [string]$ManualPdfSha256
)

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath([IO.Path]::Combine($PSScriptRoot, '..'))
$PinnedReleaseToolchainPolicySha256 = (
    '9d95a317c52c4f1e6a76444af3f03891f6514e4a3be554083b075b255a630438'
)
$toolchainPolicyHandle = $null
$pinnedPowerShellHandle = $null
$pinnedPythonHandle = $null
$pinnedGitHandle = $null
$powerShellExecutable = $null
$python = $null
$gitExecutable = $null
$pythonExecutableSha = $null
$gitExecutableSha = $null
$verifiedSnapshotHandles = [Collections.Generic.List[IO.FileStream]]::new()

function Test-EarlyCanonicalAbsolutePath([string]$Path) {
    if ([String]::IsNullOrWhiteSpace($Path) -or -not [IO.Path]::IsPathRooted($Path)) {
        return $false
    }
    try {
        $full = [IO.Path]::GetFullPath($Path)
        return $full.Equals($Path, [StringComparison]::OrdinalIgnoreCase)
    }
    catch {
        return $false
    }
}

function Assert-EarlyRegularNoReparseFile([string]$Path, [string]$Label) {
    if (
        -not (Test-EarlyCanonicalAbsolutePath $Path) -or
        -not [IO.File]::Exists($Path)
    ) {
        throw "$Label must be an existing canonical absolute regular file."
    }
    $file = [IO.FileInfo]::new($Path)
    if (
        -not $file.Exists -or
        -not $file.FullName.Equals($Path, [StringComparison]::OrdinalIgnoreCase) -or
        (($file.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)
    ) {
        throw "$Label must resolve to its exact canonical regular-file path."
    }
    $current = $file.Directory
    while ($null -ne $current) {
        if (($current.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "$Label path crosses a reparse point: $($current.FullName)"
        }
        $current = $current.Parent
    }
}

function Assert-EarlyDirectoryNoReparse([string]$Path, [string]$Label) {
    if (
        -not (Test-EarlyCanonicalAbsolutePath $Path) -or
        -not [IO.Directory]::Exists($Path)
    ) {
        throw "$Label must be an existing canonical absolute directory."
    }
    $current = [IO.DirectoryInfo]::new($Path)
    while ($null -ne $current) {
        if (($current.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "$Label path crosses a reparse point: $($current.FullName)"
        }
        $current = $current.Parent
    }
}

function Get-EarlyStreamSha256([IO.Stream]$Stream) {
    $Stream.Position = 0
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $digest = $sha.ComputeHash($Stream)
    }
    finally {
        $sha.Dispose()
        $Stream.Position = 0
    }
    return ([BitConverter]::ToString($digest)).Replace('-', '').ToLowerInvariant()
}

function Open-EarlyPinnedFile([string]$Path, [string]$ExpectedSha256, [string]$Label) {
    if ($ExpectedSha256 -cnotmatch '^[0-9a-f]{64}$') {
        throw "$Label policy digest must be an exact lowercase SHA-256 digest."
    }
    Assert-EarlyRegularNoReparseFile $Path $Label
    $handle = [IO.File]::Open(
        $Path,
        [IO.FileMode]::Open,
        [IO.FileAccess]::Read,
        [IO.FileShare]::Read
    )
    try {
        $actualSha256 = Get-EarlyStreamSha256 $handle
        if ($actualSha256 -cne $ExpectedSha256) {
            throw "$Label digest differs from the source-pinned policy."
        }
        return [PSCustomObject]@{
            Path = $Path
            Sha256 = $actualSha256
            Handle = $handle
        }
    }
    catch {
        $handle.Dispose()
        throw
    }
}

function Read-EarlyPinnedToolchainPolicy([string]$Path, [string]$ExpectedSha256) {
    Assert-EarlyRegularNoReparseFile $Path 'Release toolchain policy'
    $handle = [IO.File]::Open(
        $Path,
        [IO.FileMode]::Open,
        [IO.FileAccess]::Read,
        [IO.FileShare]::Read
    )
    try {
        if ($handle.Length -lt 2 -or $handle.Length -gt 65536) {
            throw 'Release toolchain policy has an unsupported byte length.'
        }
        $actualSha256 = Get-EarlyStreamSha256 $handle
        if ($actualSha256 -cne $ExpectedSha256) {
            throw 'Release toolchain policy bytes differ from the source-pinned digest.'
        }
        $bytes = [Array]::CreateInstance([byte], [int]$handle.Length)
        $offset = 0
        while ($offset -lt $bytes.Length) {
            $read = $handle.Read($bytes, $offset, $bytes.Length - $offset)
            if ($read -eq 0) {
                throw 'Release toolchain policy ended before its declared length.'
            }
            $offset += $read
        }
        $strictUtf8 = [Text.UTF8Encoding]::new($false, $true)
        try {
            $webExtensions = [Reflection.Assembly]::Load(
                'System.Web.Extensions, Version=4.0.0.0, Culture=neutral, ' +
                'PublicKeyToken=31bf3856ad364e35'
            )
            $serializerType = $webExtensions.GetType(
                'System.Web.Script.Serialization.JavaScriptSerializer',
                $true
            )
            $serializer = [Activator]::CreateInstance($serializerType)
            $deserializeObject = $serializerType.GetMethod(
                'DeserializeObject',
                [Type[]]@([string])
            )
            $policy = $deserializeObject.Invoke(
                $serializer,
                [object[]]@($strictUtf8.GetString($bytes))
            )
        }
        catch {
            throw 'Release toolchain policy is not strict UTF-8 JSON.'
        }
        $expectedKeys = @(
            'schema',
            'status',
            'runner_mode',
            'powershell_path',
            'powershell_sha256',
            'python_path',
            'python_sha256',
            'git_path',
            'git_sha256',
            'build_closure_manifest_path',
            'build_closure_manifest_sha256',
            'offline_wheelhouse_manifest_path',
            'offline_wheelhouse_manifest_sha256'
        )
        $actualKeys = @($policy.Keys)
        if ($actualKeys.Count -ne $expectedKeys.Count) {
            throw 'Release toolchain policy keys do not match the closed schema.'
        }
        foreach ($key in $expectedKeys) {
            if ($actualKeys -cnotcontains $key) {
                throw "Release toolchain policy is missing exact key: $key"
            }
        }
        if ($policy['schema'] -cne 'cogni.release.toolchain-policy.v2') {
            throw 'Release toolchain policy schema is unsupported.'
        }
        return [PSCustomObject]@{
            Policy = $policy
            Handle = $handle
        }
    }
    catch {
        $handle.Dispose()
        throw
    }
}

Assert-EarlyDirectoryNoReparse $root 'Repository root'

if ($PublishRelease) {
    try {
        # Publication must establish its executables without consulting PATH or
        # executing Git/Python.  The source-pinned policy is admitted using only
        # PowerShell/.NET primitives, then its file handle remains locked for the
        # complete build so the admitted bytes cannot be replaced underneath it.
        $toolchainPolicyPath = [IO.Path]::GetFullPath(
            [IO.Path]::Combine($root, 'config', 'release-toolchain-policy.json')
        )
        $toolchainAdmission = Read-EarlyPinnedToolchainPolicy `
            $toolchainPolicyPath $PinnedReleaseToolchainPolicySha256
        $toolchainPolicyHandle = $toolchainAdmission.Handle
        $toolchainPolicy = $toolchainAdmission.Policy
        $pinKeys = @(
            'runner_mode',
            'powershell_path',
            'powershell_sha256',
            'python_path',
            'python_sha256',
            'git_path',
            'git_sha256',
            'build_closure_manifest_path',
            'build_closure_manifest_sha256',
            'offline_wheelhouse_manifest_path',
            'offline_wheelhouse_manifest_sha256'
        )
        if ($toolchainPolicy['status'] -ceq 'unconfigured') {
            foreach ($key in $pinKeys) {
                if ($null -ne $toolchainPolicy[$key]) {
                    throw 'Unconfigured release toolchain policy must contain null tool pins.'
                }
            }
            throw (
                'Release toolchain policy is not approved; publication is blocked ' +
                'before tool discovery.'
            )
        }
        if ($toolchainPolicy['status'] -cne 'approved') {
            throw 'Release toolchain policy status must be approved or unconfigured.'
        }
        foreach ($key in $pinKeys) {
            if (-not ($toolchainPolicy[$key] -is [string])) {
                throw "Approved release toolchain policy key must be a string: $key"
            }
        }
        if ($toolchainPolicy['runner_mode'] -cne 'protected-no-profile-isolated-runner') {
            throw 'Approved publication requires protected-no-profile-isolated-runner mode.'
        }
        $processArguments = @([Environment]::GetCommandLineArgs())
        if (
            $processArguments -cnotcontains '-NoProfile' -or
            $processArguments -cnotcontains '-NonInteractive'
        ) {
            throw 'Approved publication requires a NoProfile non-interactive host.'
        }
        $runningPowerShell = [IO.Path]::GetFullPath(
            [Diagnostics.Process]::GetCurrentProcess().MainModule.FileName
        )
        if (-not $runningPowerShell.Equals(
            $toolchainPolicy['powershell_path'],
            [StringComparison]::OrdinalIgnoreCase
        )) {
            throw 'Running PowerShell host path differs from the source-pinned policy.'
        }
        $powerShellAdmission = Open-EarlyPinnedFile `
            $runningPowerShell $toolchainPolicy['powershell_sha256'] `
            'Policy-approved PowerShell executable'
        $pinnedPowerShellHandle = $powerShellAdmission.Handle
        $powerShellExecutable = $powerShellAdmission.Path

        # Merely naming closure manifests is not enough to prove that every
        # imported build module and wheel came from those exact bytes.  Until
        # the protected runner installs and executes exclusively from the
        # source-pinned offline wheelhouse, publication must remain closed.
        # This check deliberately precedes every Python/Git discovery or
        # invocation, including version probes.
        throw (
            'EXTERNAL_BLOCKER: protected whole-build closure and offline ' +
            'wheelhouse enforcement is not available in this local runner; ' +
            'publication is blocked before Python or Git execution.'
        )

        $pythonAdmission = Open-EarlyPinnedFile `
            $toolchainPolicy['python_path'] $toolchainPolicy['python_sha256'] `
            'Policy-approved Python executable'
        $pinnedPythonHandle = $pythonAdmission.Handle
        $python = $pythonAdmission.Path
        $pythonExecutableSha = $pythonAdmission.Sha256
        $gitAdmission = Open-EarlyPinnedFile `
            $toolchainPolicy['git_path'] $toolchainPolicy['git_sha256'] `
            'Policy-approved Git executable'
        $pinnedGitHandle = $gitAdmission.Handle
        $gitExecutable = $gitAdmission.Path
        $gitExecutableSha = $gitAdmission.Sha256
    }
    catch {
        if ($null -ne $pinnedGitHandle) { $pinnedGitHandle.Dispose() }
        if ($null -ne $pinnedPythonHandle) { $pinnedPythonHandle.Dispose() }
        if ($null -ne $pinnedPowerShellHandle) { $pinnedPowerShellHandle.Dispose() }
        if ($null -ne $toolchainPolicyHandle) { $toolchainPolicyHandle.Dispose() }
        throw
    }
}
else {
    # Artifact-only developer builds are intentionally convenient and remain
    # explicitly UNVERIFIED; they may discover local tools through PATH.
    $powerShellExecutable = [Diagnostics.Process]::GetCurrentProcess().MainModule.FileName
    $python = (Microsoft.PowerShell.Core\Get-Command python `
        -CommandType Application -ErrorAction Stop |
        Microsoft.PowerShell.Utility\Select-Object -First 1).Source
    $gitExecutable = (Microsoft.PowerShell.Core\Get-Command git `
        -CommandType Application -ErrorAction Stop |
        Microsoft.PowerShell.Utility\Select-Object -First 1).Source
}

# Caller/profile functions and aliases must not intercept release filesystem
# decisions.  These script-local wrappers bind every critical operation to its
# built-in module explicitly; the bootstrap above intentionally uses .NET only.
function Resolve-Path {
    param([Parameter(Mandatory)][string]$LiteralPath)
    Microsoft.PowerShell.Management\Resolve-Path -LiteralPath $LiteralPath
}
function Expand-Archive {
    param(
        [Parameter(Mandatory)][string]$LiteralPath,
        [Parameter(Mandatory)][string]$DestinationPath
    )
    Microsoft.PowerShell.Archive\Expand-Archive `
        -LiteralPath $LiteralPath -DestinationPath $DestinationPath
}
function Copy-Item {
    param(
        [Parameter(Mandatory)][string]$LiteralPath,
        [Parameter(Mandatory)][string]$Destination,
        [switch]$Recurse
    )
    Microsoft.PowerShell.Management\Copy-Item `
        -LiteralPath $LiteralPath -Destination $Destination -Recurse:$Recurse
}
function Move-Item {
    param(
        [Parameter(Mandatory)][string]$LiteralPath,
        [Parameter(Mandatory)][string]$Destination
    )
    Microsoft.PowerShell.Management\Move-Item `
        -LiteralPath $LiteralPath -Destination $Destination
}
function Get-ChildItem {
    param(
        [Parameter(Mandatory)][string]$LiteralPath,
        [switch]$Recurse,
        [switch]$File,
        [switch]$Force
    )
    Microsoft.PowerShell.Management\Get-ChildItem `
        -LiteralPath $LiteralPath -Recurse:$Recurse -File:$File -Force:$Force
}
function Test-Path {
    param(
        [Parameter(Mandatory)][string]$LiteralPath,
        [ValidateSet('Any', 'Container', 'Leaf')][string]$PathType = 'Any'
    )
    Microsoft.PowerShell.Management\Test-Path `
        -LiteralPath $LiteralPath -PathType $PathType
}
function Remove-Item {
    param(
        [Parameter(Mandatory)][string]$LiteralPath,
        [switch]$Recurse,
        [switch]$Force
    )
    Microsoft.PowerShell.Management\Remove-Item `
        -LiteralPath $LiteralPath -Recurse:$Recurse -Force:$Force
}
function New-Item {
    param(
        [Parameter(Mandatory)][string]$ItemType,
        [Parameter(Mandatory)][string]$Path,
        [switch]$Force
    )
    Microsoft.PowerShell.Management\New-Item `
        -ItemType $ItemType -Path $Path -Force:$Force
}
function Get-Item {
    param([Parameter(Mandatory)][string]$LiteralPath, [switch]$Force)
    Microsoft.PowerShell.Management\Get-Item `
        -LiteralPath $LiteralPath -Force:$Force
}
function Join-Path {
    param(
        [Parameter(Mandatory, Position = 0)][string]$Path,
        [Parameter(Mandatory, Position = 1)][string]$ChildPath
    )
    Microsoft.PowerShell.Management\Join-Path -Path $Path -ChildPath $ChildPath
}
function Split-Path {
    param(
        [Parameter(Mandatory, Position = 0)][string]$Path,
        [switch]$Parent,
        [switch]$Leaf,
        [switch]$Qualifier
    )
    if ($Parent) {
        return Microsoft.PowerShell.Management\Split-Path -Path $Path -Parent
    }
    if ($Leaf) {
        return Microsoft.PowerShell.Management\Split-Path -Path $Path -Leaf
    }
    if ($Qualifier) {
        return Microsoft.PowerShell.Management\Split-Path -Path $Path -Qualifier
    }
    Microsoft.PowerShell.Management\Split-Path -Path $Path
}
function Compare-Object {
    param(
        [Parameter(Mandatory)][object[]]$ReferenceObject,
        [Parameter(Mandatory)][object[]]$DifferenceObject,
        [switch]$CaseSensitive
    )
    Microsoft.PowerShell.Utility\Compare-Object `
        -ReferenceObject $ReferenceObject `
        -DifferenceObject $DifferenceObject `
        -CaseSensitive:$CaseSensitive
}
function Push-Location {
    param([Parameter(Mandatory, Position = 0)][string]$Path)
    Microsoft.PowerShell.Management\Push-Location -Path $Path
}
function Pop-Location { Microsoft.PowerShell.Management\Pop-Location }
function Add-Type {
    param([Parameter(Mandatory)][string]$AssemblyName)
    Microsoft.PowerShell.Utility\Add-Type -AssemblyName $AssemblyName
}

if (
    $Treeish -notmatch '^[A-Za-z0-9._/-]{1,128}$' -or
    $Treeish.StartsWith('-') -or
    $Treeish.Contains('..')
) {
    throw 'Treeish contains unsupported characters.'
}

function Get-LowerSha256([string]$Path) {
    $stream = [IO.File]::Open(
        $Path,
        [IO.FileMode]::Open,
        [IO.FileAccess]::Read,
        [IO.FileShare]::Read
    )
    try {
        return Get-EarlyStreamSha256 $stream
    }
    finally {
        $stream.Dispose()
    }
}

function Assert-ExactSha256([string]$Value, [string]$Label) {
    if ($Value -cnotmatch '^[0-9a-f]{64}$') {
        throw "$Label must be an exact lowercase SHA-256 digest."
    }
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

function Test-FullyQualifiedCanonicalPath([string]$Path) {
    if ([String]::IsNullOrWhiteSpace($Path) -or -not [IO.Path]::IsPathRooted($Path)) {
        return $false
    }
    try {
        $qualifier = Split-Path -Path $Path -Qualifier
        if ([String]::IsNullOrWhiteSpace($qualifier)) {
            return $false
        }
        $full = [IO.Path]::GetFullPath($Path)
        return $full.Equals($Path, [StringComparison]::OrdinalIgnoreCase)
    }
    catch {
        return $false
    }
}

function Get-SafeRelativePath([string]$TreeRoot, [string]$Path) {
    $resolvedRoot = (Resolve-Path -LiteralPath $TreeRoot).Path.TrimEnd('\', '/')
    $resolvedPath = (Resolve-Path -LiteralPath $Path).Path
    $rootPrefix = $resolvedRoot + [IO.Path]::DirectorySeparatorChar
    if (-not $resolvedPath.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Release tree member escapes its trusted root: $resolvedPath"
    }
    $relative = $resolvedPath.Substring($rootPrefix.Length).Replace('\', '/')
    if (
        [String]::IsNullOrWhiteSpace($relative) -or
        [IO.Path]::IsPathRooted($relative) -or
        $relative.StartsWith('/') -or
        $relative.Contains(':') -or
        $relative -match '(^|/)\.\.(/|$)' -or
        $relative -match '[\x00-\x1f\x7f]'
    ) {
        throw "Unsafe release relative path: $relative"
    }
    return $relative
}

function Resolve-TrustedEvidenceFile([string]$Path, [string]$Label) {
    if (
        [String]::IsNullOrWhiteSpace($Path) -or
        -not (Test-FullyQualifiedCanonicalPath $Path) -or
        -not (Test-Path -LiteralPath $Path -PathType Leaf)
    ) {
        throw "$Label must be an existing absolute regular file."
    }
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    Assert-NoReparseAncestors $resolved
    return $resolved
}

function New-VerifiedEvidenceSnapshot(
    [string]$Path,
    [string]$ExpectedSha256,
    [string]$Label,
    [string]$SnapshotDirectory,
    [string]$SnapshotName
) {
    if (-not [String]::IsNullOrWhiteSpace($ExpectedSha256)) {
        Assert-ExactSha256 $ExpectedSha256 "$Label digest"
    }
    $resolved = Resolve-TrustedEvidenceFile $Path $Label
    $snapshot = Join-Path $SnapshotDirectory $SnapshotName
    if (Test-Path -LiteralPath $snapshot) {
        throw "$Label snapshot already exists."
    }
    $input = $null
    $output = $null
    try {
        # The external bytes are admitted once while exclusively opened.  The
        # private output handle stays open with read-only sharing until the
        # whole publication finishes, so neither the validator nor any later
        # copy can be redirected by atomic path replacement.
        $input = [IO.File]::Open(
            $resolved,
            [IO.FileMode]::Open,
            [IO.FileAccess]::Read,
            [IO.FileShare]::None
        )
        $output = [IO.File]::Open(
            $snapshot,
            [IO.FileMode]::CreateNew,
            [IO.FileAccess]::ReadWrite,
            [IO.FileShare]::Read
        )
        $input.CopyTo($output)
        $output.Flush($true)
        $actualSha = Get-EarlyStreamSha256 $output
        if (
            -not [String]::IsNullOrWhiteSpace($ExpectedSha256) -and
            $actualSha -cne $ExpectedSha256
        ) {
            throw "$Label digest does not match the operator-pinned digest."
        }
        [void]$script:verifiedSnapshotHandles.Add($output)
        $output = $null
    }
    finally {
        if ($null -ne $output) { $output.Dispose() }
        if ($null -ne $input) { $input.Dispose() }
    }
    return [PSCustomObject]@{
        Path = $snapshot
        Sha256 = $actualSha
        Label = $Label
        Handle = $script:verifiedSnapshotHandles[
            $script:verifiedSnapshotHandles.Count - 1
        ]
    }
}

function New-LockedUtf8TextSnapshot(
    [string]$Path,
    [string]$Text,
    [string]$Label
) {
    if (Test-Path -LiteralPath $Path) {
        throw "$Label snapshot already exists."
    }
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes($Text)
    $output = $null
    try {
        $output = [IO.File]::Open(
            $Path,
            [IO.FileMode]::CreateNew,
            [IO.FileAccess]::ReadWrite,
            [IO.FileShare]::Read
        )
        $output.Write($bytes, 0, $bytes.Length)
        $output.Flush($true)
        $actualSha = Get-EarlyStreamSha256 $output
        [void]$script:verifiedSnapshotHandles.Add($output)
        $output = $null
    }
    finally {
        if ($null -ne $output) { $output.Dispose() }
    }
    return [PSCustomObject]@{
        Path = $Path
        Sha256 = $actualSha
        Label = $Label
        Handle = $script:verifiedSnapshotHandles[
            $script:verifiedSnapshotHandles.Count - 1
        ]
    }
}

function Assert-SnapshotDigest($Snapshot, [string]$ExpectedSha256) {
    Assert-ExactSha256 $ExpectedSha256 "$($Snapshot.Label) staged digest"
    if ((Get-EarlyStreamSha256 $Snapshot.Handle) -cne $ExpectedSha256) {
        throw "$($Snapshot.Label) private snapshot changed after admission."
    }
}

function Copy-VerifiedSnapshotToBundle(
    $Snapshot,
    [string]$ExpectedSha256,
    [string]$DestinationName,
    [string]$PublishStage
) {
    Assert-SnapshotDigest $Snapshot $ExpectedSha256
    $destination = Join-Path $PublishStage $DestinationName
    $destinationParent = Split-Path -Parent $destination
    if (-not (Test-Path -LiteralPath $destinationParent)) {
        New-Item -ItemType Directory -Path $destinationParent -Force |
            Microsoft.PowerShell.Core\Out-Null
    }
    $output = $null
    try {
        $output = [IO.File]::Open(
            $destination,
            [IO.FileMode]::CreateNew,
            [IO.FileAccess]::Write,
            [IO.FileShare]::None
        )
        $Snapshot.Handle.Position = 0
        $Snapshot.Handle.CopyTo($output)
        $output.Flush($true)
        $Snapshot.Handle.Position = 0
    }
    finally {
        if ($null -ne $output) { $output.Dispose() }
    }
    if ((Get-LowerSha256 $destination) -cne $ExpectedSha256) {
        throw "$($Snapshot.Label) bundle copy changed after validation."
    }
}

function Read-LockedUtf8SnapshotText($Snapshot) {
    $Snapshot.Handle.Position = 0
    $reader = [IO.StreamReader]::new(
        $Snapshot.Handle,
        [Text.UTF8Encoding]::new($false, $true),
        $true,
        4096,
        $true
    )
    try {
        return $reader.ReadToEnd()
    }
    finally {
        $reader.Dispose()
        $Snapshot.Handle.Position = 0
    }
}

function Get-ClosedTreeChecksumLines([string]$TreeRoot) {
    $resolvedRoot = (Resolve-Path -LiteralPath $TreeRoot).Path
    Assert-NoReparseAncestors $resolvedRoot
    $entries = foreach ($file in Get-ChildItem -LiteralPath $resolvedRoot -Recurse -File -Force) {
        Assert-NoReparseAncestors $file.FullName
        $relative = Get-SafeRelativePath $resolvedRoot $file.FullName
        if (
            [String]::IsNullOrWhiteSpace($relative) -or
            $relative.StartsWith('/') -or
            $relative.Contains('../') -or
            $relative -match '[\x00-\x1f\x7f]'
        ) {
            throw "Unsafe expanded-source relative path: $relative"
        }
        [PSCustomObject]@{
            Relative = $relative
            Line = ('{0}  {1}' -f (Get-LowerSha256 $file.FullName), $relative)
        }
    }
    $ordered = @(
        $entries |
            Microsoft.PowerShell.Utility\Sort-Object `
                -Property Relative -CaseSensitive
    )
    if ($ordered.Count -eq 0) {
        throw 'Expanded source tree must contain at least one regular file.'
    }
    return @(
        $ordered | Microsoft.PowerShell.Core\ForEach-Object { $_.Line }
    )
}

function Assert-NoForbiddenPackagedSource([string]$TreeRoot) {
    # The artifact-only bundle intentionally exports the complete committed
    # source tree.  Git ignore rules are not a release boundary: a force-added
    # model, cache, or credential file would otherwise be copied verbatim into
    # both the source archive and expanded launcher tree.
    $forbiddenDirectories = [Collections.Generic.HashSet[string]]::new(
        [StringComparer]::OrdinalIgnoreCase
    )
    foreach ($name in @(
        '.git', '.cache', '.mypy_cache', '.pytest_cache', '.ruff_cache',
        '__pycache__', 'build', 'dist', 'node_modules', 'output', 'outputs',
        'tmp', 'work'
    )) {
        [void]$forbiddenDirectories.Add($name)
    }
    $forbiddenNames = [Collections.Generic.HashSet[string]]::new(
        [StringComparer]::OrdinalIgnoreCase
    )
    foreach ($name in @(
        '.env', '.netrc', '.npmrc', '.pypirc',
        'credentials', 'credentials.json', 'id_dsa', 'id_ecdsa',
        'id_ed25519', 'id_rsa', 'secrets.json', 'service-account.json',
        'token.json'
    )) {
        [void]$forbiddenNames.Add($name)
    }
    $forbiddenExtensions = [Collections.Generic.HashSet[string]]::new(
        [StringComparer]::OrdinalIgnoreCase
    )
    foreach ($extension in @(
        '.bin', '.ckpt', '.gguf', '.h5', '.key', '.onnx', '.p12', '.pem',
        '.pfx', '.pickle', '.pkl', '.pt', '.pth', '.pyc', '.pyo',
        '.safetensors'
    )) {
        [void]$forbiddenExtensions.Add($extension)
    }

    foreach ($file in Get-ChildItem -LiteralPath $TreeRoot -Recurse -File -Force) {
        $relative = Get-SafeRelativePath $TreeRoot $file.FullName
        $segments = $relative.Split('/')
        for ($index = 0; $index -lt ($segments.Length - 1); $index += 1) {
            if ($forbiddenDirectories.Contains($segments[$index])) {
                throw "Forbidden cache/build directory in packaged source: $relative"
            }
        }
        $leaf = $segments[$segments.Length - 1]
        if (
            $forbiddenNames.Contains($leaf) -or
            $leaf.StartsWith('.env.', [StringComparison]::OrdinalIgnoreCase)
        ) {
            throw "Forbidden credential file in packaged source: $relative"
        }
        if ($forbiddenExtensions.Contains([IO.Path]::GetExtension($leaf))) {
            throw "Forbidden model/cache/credential artifact in packaged source: $relative"
        }

        # Private-key blocks remain forbidden even when hidden behind an
        # innocent filename.  Keys are small text artifacts; the bounded scan
        # avoids materialising an arbitrary large source member a second time.
        if ($file.Length -le 2097152) {
            $text = [Text.Encoding]::ASCII.GetString(
                [IO.File]::ReadAllBytes($file.FullName)
            )
            if ($text -match '-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----') {
                throw "Private key material is forbidden in packaged source: $relative"
            }
        }
    }
}

function Get-LockedTreeChecksumLines($Handles, $ExpectedLines) {
    $expected = @($ExpectedLines)
    $locked = @($Handles)
    if ($expected.Count -ne $locked.Count) {
        throw 'Locked release inventory handle count differs from its manifest.'
    }
    $lines = [Collections.Generic.List[string]]::new()
    for ($index = 0; $index -lt $expected.Count; $index += 1) {
        if (
            -not ($expected[$index] -is [string]) -or
            $expected[$index] -cnotmatch '^[0-9a-f]{64}  (.+)$'
        ) {
            throw 'Locked release inventory contains an invalid record.'
        }
        $relative = $Matches[1]
        $lockedSha256 = Get-EarlyStreamSha256 $locked[$index]
        $line = '{0}  {1}' -f $lockedSha256, $relative
        if ($line -cne $expected[$index]) {
            throw "Release file changed after its read lock: $relative"
        }
        [void]$lines.Add($line)
    }
    return @($lines.ToArray())
}

function New-VerifiedAcceptanceBundleSnapshot(
    [string]$BundleRoot,
    [string]$SnapshotRoot
) {
    if (
        [String]::IsNullOrWhiteSpace($BundleRoot) -or
        -not (Test-FullyQualifiedCanonicalPath $BundleRoot) -or
        -not (Test-Path -LiteralPath $BundleRoot -PathType Container)
    ) {
        throw 'Acceptance bundle root must be an existing canonical absolute directory.'
    }
    $resolvedRoot = (Resolve-Path -LiteralPath $BundleRoot).Path
    Assert-NoReparseAncestors $resolvedRoot
    $before = @(Get-ClosedTreeChecksumLines $resolvedRoot)
    $effectiveChecklist = 'docs/COGNIBOARD_EFFECTIVE_ACCEPTANCE_CHECKLIST_KO.md'
    $seenChecklist = $false
    $seenEvidence = $false
    $snapshots = [Collections.Generic.List[object]]::new()
    New-Item -ItemType Directory -Path $SnapshotRoot |
        Microsoft.PowerShell.Core\Out-Null
    foreach ($line in $before) {
        if ($line -cnotmatch '^([0-9a-f]{64})  (.+)$') {
            throw 'Acceptance bundle inventory contains an invalid record.'
        }
        $expectedSha256 = $Matches[1]
        $relative = $Matches[2]
        if ($relative -ceq $effectiveChecklist) {
            $seenChecklist = $true
        }
        elseif ($relative.StartsWith('release/evidence/', [StringComparison]::Ordinal)) {
            $seenEvidence = $true
        }
        else {
            throw "Acceptance bundle contains an out-of-contract path: $relative"
        }
        $sourcePath = Join-Path $resolvedRoot ($relative.Replace('/', '\'))
        $destinationPath = Join-Path $SnapshotRoot ($relative.Replace('/', '\'))
        $destinationParent = Split-Path -Parent $destinationPath
        if (-not (Test-Path -LiteralPath $destinationParent)) {
            New-Item -ItemType Directory -Path $destinationParent -Force |
                Microsoft.PowerShell.Core\Out-Null
        }
        $fileSnapshot = New-VerifiedEvidenceSnapshot `
            $sourcePath $expectedSha256 "Acceptance bundle file $relative" `
            $destinationParent (Split-Path -Leaf $destinationPath)
        [void]$snapshots.Add([PSCustomObject]@{
            Relative = $relative
            Snapshot = $fileSnapshot
        })
    }
    if (-not $seenChecklist -or -not $seenEvidence) {
        throw 'Acceptance bundle requires one effective checklist and detached evidence.'
    }
    $after = @(Get-ClosedTreeChecksumLines $resolvedRoot)
    if (
        $before.Count -ne $after.Count -or
        @(Compare-Object `
            -ReferenceObject $before `
            -DifferenceObject $after `
            -CaseSensitive).Count -ne 0
    ) {
        throw 'Acceptance bundle changed while its private snapshot was admitted.'
    }
    return [PSCustomObject]@{
        Root = $SnapshotRoot
        Checklist = Join-Path $SnapshotRoot (
            $effectiveChecklist.Replace('/', '\')
        )
        Inventory = $before
        Files = @($snapshots.ToArray())
    }
}

function Open-VerifiedSourceReadLocks([string]$TreeRoot, $ExpectedLines) {
    $resolvedRoot = (Resolve-Path -LiteralPath $TreeRoot).Path
    Assert-NoReparseAncestors $resolvedRoot
    $locks = [Collections.Generic.List[IO.FileStream]]::new()
    try {
        foreach ($line in @($ExpectedLines)) {
            if (
                -not ($line -is [string]) -or
                $line -cnotmatch '^([0-9a-f]{64})  (.+)$'
            ) {
                throw 'Expanded-source checksum inventory contains an invalid record.'
            }
            $expectedSha256 = $Matches[1]
            $relative = $Matches[2]
            if (
                [String]::IsNullOrWhiteSpace($relative) -or
                $relative.StartsWith('/') -or
                $relative.Contains(':') -or
                $relative -match '(^|/)\.\.(/|$)' -or
                $relative -match '[\x00-\x1f\x7f]'
            ) {
                throw "Unsafe expanded-source lock path: $relative"
            }
            $path = Join-Path $resolvedRoot ($relative.Replace('/', '\'))
            Assert-NoReparseAncestors $path
            $handle = [IO.File]::Open(
                $path,
                [IO.FileMode]::Open,
                [IO.FileAccess]::Read,
                [IO.FileShare]::Read
            )
            try {
                if ((Get-EarlyStreamSha256 $handle) -cne $expectedSha256) {
                    throw "Expanded-source file changed before its read lock: $relative"
                }
                [void]$locks.Add($handle)
                $handle = $null
            }
            finally {
                if ($null -ne $handle) { $handle.Dispose() }
            }
        }

        # The handles above deny write/delete/rename replacement on every
        # admitted file.  A second closed inventory detects files inserted or
        # removed while the lock set was being acquired.
        $lockedLines = @(Get-ClosedTreeChecksumLines $resolvedRoot)
        if (
            @($ExpectedLines).Count -ne $lockedLines.Count -or
            @(Compare-Object `
                -ReferenceObject @($ExpectedLines) `
                -DifferenceObject $lockedLines `
                -CaseSensitive).Count -ne 0
        ) {
            throw 'Expanded-source inventory changed while acquiring read locks.'
        }
        return @($locks.ToArray())
    }
    catch {
        foreach ($lockedHandle in $locks) { $lockedHandle.Dispose() }
        throw
    }
}

function Assert-ClosedTreeInventory(
    [string]$TreeRoot,
    $ExpectedLines,
    [string]$Label
) {
    $actual = @(Get-ClosedTreeChecksumLines $TreeRoot)
    if (
        @($ExpectedLines).Count -ne $actual.Count -or
        @(Compare-Object `
            -ReferenceObject @($ExpectedLines) `
            -DifferenceObject $actual `
            -CaseSensitive).Count -ne 0
    ) {
        throw "$Label closed inventory changed after admission."
    }
}

function Get-CanonicalChecksumDigest($Lines) {
    $canonical = (($Lines -join "`n") + "`n")
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes($canonical)
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $digest = $sha.ComputeHash($bytes)
    }
    finally {
        $sha.Dispose()
    }
    return ([BitConverter]::ToString($digest)).Replace('-', '').ToLowerInvariant()
}

function Write-Utf8LfLines([string]$Path, $Lines) {
    $canonical = (($Lines -join "`n") + "`n")
    [IO.File]::WriteAllText($Path, $canonical, [Text.UTF8Encoding]::new($false))
}

function New-ReleasePublishStage([string]$PublishedOutput) {
    if (Test-Path -LiteralPath $PublishedOutput) {
        throw "Release output already exists; refusing to merge: $PublishedOutput"
    }
    $outputParent = Split-Path -Parent $PublishedOutput
    if (-not (Test-Path -LiteralPath $outputParent)) {
        New-Item -ItemType Directory -Path $outputParent |
            Microsoft.PowerShell.Core\Out-Null
    }
    $outputParent = (Resolve-Path -LiteralPath $outputParent).Path
    Assert-NoReparseAncestors $outputParent
    $stage = Join-Path $outputParent (
        '.cogni-release-staging-' + [Guid]::NewGuid().ToString('N')
    )
    New-Item -ItemType Directory -Path $stage |
        Microsoft.PowerShell.Core\Out-Null
    return $stage
}

function Get-ArchiveEntrySha256([string]$ArchivePath, [string]$EntryName, [string]$Prefix) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($ArchivePath)
    try {
        $seen = [Collections.Generic.HashSet[string]]::new(
            [StringComparer]::OrdinalIgnoreCase
        )
        $matches = @()
        $entryCount = 0
        [long]$expandedBytes = 0
        foreach ($entry in $zip.Entries) {
            $name = $entry.FullName
            $entryCount += 1
            $expandedBytes += $entry.Length
            if (
                -not $name.StartsWith($Prefix, [StringComparison]::Ordinal) -or
                $name.Contains('../') -or
                $name.Contains('\') -or
                $name.StartsWith('/') -or
                $name.Contains(':') -or
                $name.Contains([char]0) -or
                $entryCount -gt 20000 -or
                $entry.Length -gt 536870912 -or
                $expandedBytes -gt 2147483648
            ) {
                throw "Unsafe source archive entry: $name"
            }
            $relative = $name.Substring($Prefix.Length).TrimEnd('/')
            if ($relative.Length -gt 0) {
                foreach ($segment in $relative.Split('/')) {
                    if (
                        [String]::IsNullOrEmpty($segment) -or
                        $segment -eq '.' -or
                        $segment -eq '..' -or
                        $segment.EndsWith('.') -or
                        $segment.EndsWith(' ') -or
                        $segment -match '^(?i:con|prn|aux|nul|clock\$|com[1-9]|lpt[1-9])(?:\..*)?$'
                    ) {
                        throw "Unsafe source archive entry: $name"
                    }
                }
            }
            $normalizedName = $name.TrimEnd('/')
            if (-not $seen.Add($normalizedName)) {
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

$python = Resolve-TrustedEvidenceFile $python 'Python application'
$gitExecutable = Resolve-TrustedEvidenceFile $gitExecutable 'Git application'
$pythonExecutableName = [IO.Path]::GetFileName($python)
if ($pythonExecutableName -cnotmatch '^[A-Za-z0-9._+-]{1,128}$') {
    throw 'Python application basename is unsafe for release metadata.'
}
if ($PublishRelease) {
    if (
        $python -cne $toolchainPolicy.python_path -or
        $pythonExecutableSha -cne $toolchainPolicy.python_sha256 -or
        $gitExecutable -cne $toolchainPolicy.git_path -or
        $gitExecutableSha -cne $toolchainPolicy.git_sha256
    ) {
        throw 'Policy-approved release toolchain identity changed after admission.'
    }
}
else {
    $pythonExecutableSha = Get-LowerSha256 $python
    $gitExecutableSha = Get-LowerSha256 $gitExecutable
}
$savedGitEnvironment = @{}
foreach ($entry in [Environment]::GetEnvironmentVariables('Process').GetEnumerator()) {
    $name = [string]$entry.Key
    if ($name.StartsWith('GIT_', [StringComparison]::OrdinalIgnoreCase)) {
        $savedGitEnvironment[$name] = [string]$entry.Value
        [Environment]::SetEnvironmentVariable($name, $null, 'Process')
    }
}
[Environment]::SetEnvironmentVariable('GIT_CONFIG_NOSYSTEM', '1', 'Process')
[Environment]::SetEnvironmentVariable('GIT_CONFIG_SYSTEM', 'NUL', 'Process')
[Environment]::SetEnvironmentVariable('GIT_CONFIG_GLOBAL', 'NUL', 'Process')
[Environment]::SetEnvironmentVariable('GIT_TERMINAL_PROMPT', '0', 'Process')
[Environment]::SetEnvironmentVariable('GIT_OPTIONAL_LOCKS', '0', 'Process')
$gitSafetyArguments = @(
    '-c', 'core.fsmonitor=false',
    '-c', 'core.hooksPath=NUL',
    '-c', 'credential.helper=',
    '-c', 'filter.lfs.required=false',
    '-c', 'filter.lfs.smudge=',
    '-c', 'filter.lfs.clean='
)
$commitOid = (& $gitExecutable @gitSafetyArguments -C $root `
    rev-parse --verify --end-of-options "$Treeish^{commit}").Trim()
if ($LASTEXITCODE -ne 0 -or $commitOid -notmatch '^[0-9a-f]{40}$') {
    throw 'Treeish did not resolve to one immutable Git commit.'
}
if ($PublishRelease) {
    $headOid = (& $gitExecutable @gitSafetyArguments -C $root `
        rev-parse --verify HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $headOid -cne $commitOid) {
        throw 'Publication requires Treeish to resolve to the current exact HEAD commit.'
    }
    $dirtyStatus = @(& $gitExecutable @gitSafetyArguments -C $root `
        status --porcelain=v1 --untracked-files=all)
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not verify the publication worktree state.'
    }
    if ($dirtyStatus.Count -ne 0) {
        throw 'Publication requires a completely clean current HEAD worktree.'
    }
}
$commitEpoch = (& $gitExecutable @gitSafetyArguments -C $root `
    show -s --format=%ct $commitOid).Trim()
if ($LASTEXITCODE -ne 0 -or $commitEpoch -notmatch '^\d{9,12}$') {
    throw 'Could not determine the commit timestamp.'
}

$workRoot = Join-Path $root 'work'
if (-not (Test-Path -LiteralPath $workRoot)) {
    New-Item -ItemType Directory -Path $workRoot |
        Microsoft.PowerShell.Core\Out-Null
}
$workRoot = (Resolve-Path -LiteralPath $workRoot).Path
Assert-NoReparseAncestors $workRoot
$scratch = Join-Path $workRoot ('release-build-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $scratch |
    Microsoft.PowerShell.Core\Out-Null

$publishStage = $null
$publishedOutput = $null
$releaseVersion = $null
$archiveCheckpoint = $null
$sourceArchiveName = $null
$releaseEvidenceStatus = 'UNVERIFIED'
$releaseEvidenceSummary = $null
$releaseEvidenceSummarySha = $null
$releaseValidation = $null
$sourceTreeDigest = $null
$sourceReadLocks = @()
$payloadStageReadLocks = @()
$finalStageReadLocks = @()
$rawArchiveLock = $null
$rawArchiveSha = $null
$manualPdfSnapshot = $null
$buildSucceeded = $false
$savedPythonEnvironment = @{}
$oldSourceDateEpoch = $env:SOURCE_DATE_EPOCH
$pythonBuildArguments = if ($PublishRelease) {
    @('-I', '-S', '-B')
}
else {
    @()
}

try {
    foreach ($entry in [Environment]::GetEnvironmentVariables('Process').GetEnumerator()) {
        $name = [string]$entry.Key
        if ($name.StartsWith('PYTHON', [StringComparison]::OrdinalIgnoreCase)) {
            $savedPythonEnvironment[$name] = [string]$entry.Value
            [Environment]::SetEnvironmentVariable($name, $null, 'Process')
        }
    }
    $env:SOURCE_DATE_EPOCH = $commitEpoch
    $env:PYTHONHASHSEED = '0'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $prefix = 'Cogni-OS-2-Genesis-source/'
    $rawArchive = Join-Path $scratch 'source.zip'
    & $gitExecutable @gitSafetyArguments -C $root -c core.autocrlf=false archive `
        --format=zip `
        --prefix=$prefix `
        --output=$rawArchive `
        $commitOid
    if ($LASTEXITCODE -ne 0) {
        throw 'git archive failed.'
    }

    Assert-NoReparseAncestors $rawArchive
    $rawArchiveLock = [IO.File]::Open(
        $rawArchive,
        [IO.FileMode]::Open,
        [IO.FileAccess]::Read,
        [IO.FileShare]::Read
    )
    $rawArchiveSha = Get-EarlyStreamSha256 $rawArchiveLock

    # Validate every archive entry (prefix, traversal, absolute/drive paths,
    # duplicates) before Expand-Archive can materialize any untrusted name.
    # The same bounded pass captures the checkpoint digest for the later trust
    # root comparison.
    $checkpointEntry = $prefix + 'cogni_core/cts_policy_checkpoint.json'
    $archiveCheckpoint = Get-ArchiveEntrySha256 `
        $rawArchive $checkpointEntry $prefix

    $extractRoot = Join-Path $scratch 'extract'
    [void][IO.Directory]::CreateDirectory($extractRoot)
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::ExtractToDirectory($rawArchive, $extractRoot)
    $source = (Resolve-Path (
        Join-Path $extractRoot 'Cogni-OS-2-Genesis-source'
    )).Path
    Assert-NoReparseAncestors $source

    # Admit and (for publication) lock the complete SUBJECT before reading or
    # executing any archived project bytes.  The version and checkpoint trust
    # root are then parsed as inert static data, never imported as Python.
    $archivedTreeChecksums = @(Get-ClosedTreeChecksumLines $source)
    Assert-NoForbiddenPackagedSource $source
    $sourceChecksumManifestContentSha = Get-CanonicalChecksumDigest $archivedTreeChecksums
    if ($PublishRelease) {
        $sourceReadLocks = @(
            Open-VerifiedSourceReadLocks $source $archivedTreeChecksums
        )
    }
    $versionMatches = [regex]::Matches(
        [IO.File]::ReadAllText((Join-Path $source 'cogni_os\version.py')),
        '(?m)^__version__\s*=\s*"(\d+\.\d+\.\d+)"\s*$'
    )
    if ($versionMatches.Count -ne 1) {
        throw 'Could not determine one exact version from the archived commit.'
    }
    $releaseVersion = $versionMatches[0].Groups[1].Value
    $checkpointMatches = [regex]::Matches(
        [IO.File]::ReadAllText((Join-Path $source 'cogni_core\cts_policy.py')),
        '(?ms)^DEFAULT_CHECKPOINT_SHA256\s*=\s*\(\s*"([0-9a-f]{64})"\s*\)'
    )
    if ($checkpointMatches.Count -ne 1) {
        throw 'Could not load one archived CTS checkpoint trust root.'
    }
    $expectedCheckpoint = $checkpointMatches[0].Groups[1].Value

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

    if ($PublishRelease) {
        $trustedReleaseBuilder = Resolve-TrustedEvidenceFile (
            Join-Path $root 'scripts\build_release_bundle.ps1'
        ) 'Trusted current release builder'
        $archivedReleaseBuilder = Resolve-TrustedEvidenceFile (
            Join-Path $source 'scripts\build_release_bundle.ps1'
        ) 'Archived release builder'
        if (
            (Get-LowerSha256 $trustedReleaseBuilder) -cne
            (Get-LowerSha256 $archivedReleaseBuilder)
        ) {
            throw 'Executing release builder differs from the exact archived commit.'
        }
        $archivedToolchainPolicy = Resolve-TrustedEvidenceFile (
            Join-Path $source 'config\release-toolchain-policy.json'
        ) 'Archived release toolchain policy'
        if (
            (Get-LowerSha256 $archivedToolchainPolicy) -cne
            $PinnedReleaseToolchainPolicySha256
        ) {
            throw 'Archived toolchain policy differs from the source-pinned policy.'
        }
        $trustedVerifier = Resolve-TrustedEvidenceFile (
            Join-Path $root 'scripts\validate_release_evidence.py'
        ) 'Trusted current release-evidence validator'
        $archivedVerifier = Resolve-TrustedEvidenceFile (
            Join-Path $source 'scripts\validate_release_evidence.py'
        ) 'Archived release-evidence validator'
        $trustedVerifierSha = Get-LowerSha256 $trustedVerifier
        if ($trustedVerifierSha -cne (Get-LowerSha256 $archivedVerifier)) {
            throw 'Trusted current validator bytes differ from the exact archived commit.'
        }
        $trustedPolicy = Resolve-TrustedEvidenceFile (
            Join-Path $root 'config\release-verifier-policy.json'
        ) 'Trusted current release verifier policy'
        $archivedPolicy = Resolve-TrustedEvidenceFile (
            Join-Path $source 'config\release-verifier-policy.json'
        ) 'Archived release verifier policy'
        $verifierPolicySha = Get-LowerSha256 $archivedPolicy
        if ((Get-LowerSha256 $trustedPolicy) -cne $verifierPolicySha) {
            throw 'Trusted current verifier policy differs from the exact archived commit.'
        }

        $trustedAcceptanceValidator = Resolve-TrustedEvidenceFile (
            Join-Path $root 'scripts\validate_master_acceptance_checklist.py'
        ) 'Trusted current master-acceptance validator'
        $archivedAcceptanceValidator = Resolve-TrustedEvidenceFile (
            Join-Path $source 'scripts\validate_master_acceptance_checklist.py'
        ) 'Archived master-acceptance validator'
        $acceptanceValidatorSha = Get-LowerSha256 $archivedAcceptanceValidator
        if (
            (Get-LowerSha256 $trustedAcceptanceValidator) -cne
            $acceptanceValidatorSha
        ) {
            throw 'Trusted current acceptance validator differs from the exact archived commit.'
        }
        $trustedOutstandingRenderer = Resolve-TrustedEvidenceFile (
            Join-Path $root 'scripts\render_outstanding_checklist.py'
        ) 'Trusted current outstanding-checklist renderer'
        $archivedOutstandingRenderer = Resolve-TrustedEvidenceFile (
            Join-Path $source 'scripts\render_outstanding_checklist.py'
        ) 'Archived outstanding-checklist renderer'
        $outstandingRendererSha = Get-LowerSha256 $archivedOutstandingRenderer
        if (
            (Get-LowerSha256 $trustedOutstandingRenderer) -cne
            $outstandingRendererSha
        ) {
            throw 'Trusted current checklist renderer differs from the exact archived commit.'
        }

        $archivedMasterChecklist = Resolve-TrustedEvidenceFile (
            Join-Path $source 'docs\COGNIBOARD_MASTER_ACCEPTANCE_CHECKLIST_KO.md'
        ) 'Archived master acceptance checklist'
        $archivedOutstandingChecklist = Resolve-TrustedEvidenceFile (
            Join-Path $source 'docs\COGNIBOARD_OUTSTANDING_IMPLEMENTATION_CHECKLIST_KO.md'
        ) 'Archived outstanding implementation checklist'
        $archivedAcceptancePolicy = Resolve-TrustedEvidenceFile (
            Join-Path $source 'config\acceptance-evidence-policy.json'
        ) 'Archived acceptance evidence policy'
        $acceptancePolicySha = Get-LowerSha256 $archivedAcceptancePolicy
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

    if ($PublishRelease) {
        if ([String]::IsNullOrWhiteSpace($AcceptanceBundleRoot)) {
            throw (
                'Publication requires a detached effective acceptance bundle; ' +
                'the tracked master checklist is template-only.'
            )
        }
        if (-not [String]::IsNullOrWhiteSpace($ManualPdfPath)) {
            Assert-ExactSha256 $ManualPdfSha256 'Manual PDF digest'
        }
        elseif (-not [String]::IsNullOrWhiteSpace($ManualPdfSha256)) {
            throw 'ManualPdfSha256 requires ManualPdfPath.'
        }
        foreach ($digestContract in @(
            @($ReleaseEvidenceSummarySha256, 'Release evidence summary digest'),
            @($CpuGateEvidenceSha256, 'CPU gate evidence digest'),
            @($Gpu5GateEvidenceSha256, 'GPU5 gate evidence digest'),
            @($ReleaseAttestationSha256, 'Release attestation digest'),
            @($ReleaseAttestationSignatureSha256, 'Release attestation signature digest'),
            @($RuntimeEvidenceSha256, 'GPU5 runtime evidence digest'),
            @($CompletionEvidenceSha256, 'GPU5 completion evidence digest'),
            @($IdentityPreEvidenceSha256, 'GPU5 identity-pre evidence digest'),
            @($IdentityPostEvidenceSha256, 'GPU5 identity-post evidence digest'),
            @($ConfigEvidenceSha256, 'GPU5 config evidence digest'),
            @($DeviceEvidenceSha256, 'GPU5 device evidence digest'),
            @($ModelInventorySha256, 'GPU5 model inventory digest')
        )) {
            Assert-ExactSha256 $digestContract[0] $digestContract[1]
        }

        $evidenceSnapshotDirectory = Join-Path $scratch (
            'private-release-evidence-' + [Guid]::NewGuid().ToString('N')
        )
        New-Item -ItemType Directory -Path $evidenceSnapshotDirectory |
            Microsoft.PowerShell.Core\Out-Null
        Assert-NoReparseAncestors $evidenceSnapshotDirectory
        if (-not [String]::IsNullOrWhiteSpace($ManualPdfPath)) {
            $manualPdfSnapshot = New-VerifiedEvidenceSnapshot `
                $ManualPdfPath $ManualPdfSha256 'Manual PDF' `
                $evidenceSnapshotDirectory 'manual.pdf'
        }
        $trustedVerifierSnapshot = New-VerifiedEvidenceSnapshot `
            $trustedVerifier $trustedVerifierSha 'Trusted current release validator' `
            $evidenceSnapshotDirectory 'trusted-release-validator.py'
        $trustedVerifier = $trustedVerifierSnapshot.Path
        $summarySnapshot = New-VerifiedEvidenceSnapshot `
            $ReleaseEvidenceSummaryPath $ReleaseEvidenceSummarySha256 `
            'Release evidence summary' $evidenceSnapshotDirectory 'summary.json'
        $cpuSnapshot = New-VerifiedEvidenceSnapshot `
            $CpuGateEvidencePath $CpuGateEvidenceSha256 `
            'CPU gate evidence' $evidenceSnapshotDirectory 'cpu.json'
        $gpuSnapshot = New-VerifiedEvidenceSnapshot `
            $Gpu5GateEvidencePath $Gpu5GateEvidenceSha256 `
            'GPU5 gate evidence' $evidenceSnapshotDirectory 'gpu5.json'
        $attestationSnapshot = New-VerifiedEvidenceSnapshot `
            $ReleaseAttestationPath $ReleaseAttestationSha256 `
            'Release attestation' $evidenceSnapshotDirectory 'attestation.json'
        $signatureSnapshot = New-VerifiedEvidenceSnapshot `
            $ReleaseAttestationSignaturePath $ReleaseAttestationSignatureSha256 `
            'Release attestation signature' $evidenceSnapshotDirectory 'attestation.sig'
        # The key digest is not operator-selectable: the immutable policy inside
        # the exact source commit is the sole trust anchor.
        $publicKeySnapshot = New-VerifiedEvidenceSnapshot `
            $VerifierPublicKeyPath $null 'Verifier public key' `
            $evidenceSnapshotDirectory 'verifier-public-key.json'
        $runtimeSnapshot = New-VerifiedEvidenceSnapshot `
            $RuntimeEvidencePath $RuntimeEvidenceSha256 'GPU5 runtime evidence' `
            $evidenceSnapshotDirectory 'runtime.json'
        $completionSnapshot = New-VerifiedEvidenceSnapshot `
            $CompletionEvidencePath $CompletionEvidenceSha256 'GPU5 completion evidence' `
            $evidenceSnapshotDirectory 'completion.json'
        $identityPreSnapshot = New-VerifiedEvidenceSnapshot `
            $IdentityPreEvidencePath $IdentityPreEvidenceSha256 'GPU5 identity-pre evidence' `
            $evidenceSnapshotDirectory 'identity-pre.json'
        $identityPostSnapshot = New-VerifiedEvidenceSnapshot `
            $IdentityPostEvidencePath $IdentityPostEvidenceSha256 'GPU5 identity-post evidence' `
            $evidenceSnapshotDirectory 'identity-post.json'
        $configSnapshot = New-VerifiedEvidenceSnapshot `
            $ConfigEvidencePath $ConfigEvidenceSha256 'GPU5 config evidence' `
            $evidenceSnapshotDirectory 'config.json'
        $deviceSnapshot = New-VerifiedEvidenceSnapshot `
            $DeviceEvidencePath $DeviceEvidenceSha256 'GPU5 device evidence' `
            $evidenceSnapshotDirectory 'device.json'
        $modelInventorySnapshot = New-VerifiedEvidenceSnapshot `
            $ModelInventoryPath $ModelInventorySha256 'GPU5 model inventory' `
            $evidenceSnapshotDirectory 'model-inventory.json'
        $acceptanceBundleSnapshot = New-VerifiedAcceptanceBundleSnapshot `
            $AcceptanceBundleRoot (Join-Path $evidenceSnapshotDirectory 'acceptance')
        $acceptanceBundleDigest = Get-CanonicalChecksumDigest `
            $acceptanceBundleSnapshot.Inventory

        $releaseEvidenceSummarySha = $summarySnapshot.Sha256
        $modelManifest = Join-Path $source 'config\gemma4-e4b-it.manifest.toml'
        $expectedModelManifestSha = Get-LowerSha256 $modelManifest
        $runtimeConfig = Join-Path $source 'config\default.toml'
        $expectedConfigSha = Get-LowerSha256 $runtimeConfig
        $releaseValidationPath = Join-Path $scratch 'release-evidence-validation.json'
        $trustedVerifierText = Read-LockedUtf8SnapshotText $trustedVerifierSnapshot
        $releaseValidationLines = @($trustedVerifierText | & $python -I -S -B - `
            --summary $summarySnapshot.Path `
            --summary-sha256 $summarySnapshot.Sha256 `
            --cpu-evidence $cpuSnapshot.Path `
            --cpu-evidence-sha256 $cpuSnapshot.Sha256 `
            --gpu5-evidence $gpuSnapshot.Path `
            --gpu5-evidence-sha256 $gpuSnapshot.Sha256 `
            --attestation $attestationSnapshot.Path `
            --attestation-sha256 $attestationSnapshot.Sha256 `
            --attestation-signature $signatureSnapshot.Path `
            --attestation-signature-sha256 $signatureSnapshot.Sha256 `
            --verifier-public-key $publicKeySnapshot.Path `
            --verifier-public-key-sha256 $publicKeySnapshot.Sha256 `
            --verifier-policy $archivedPolicy `
            --verifier-policy-sha256 $verifierPolicySha `
            --runtime $runtimeSnapshot.Path `
            --runtime-sha256 $runtimeSnapshot.Sha256 `
            --completion $completionSnapshot.Path `
            --completion-sha256 $completionSnapshot.Sha256 `
            --identity-pre $identityPreSnapshot.Path `
            --identity-pre-sha256 $identityPreSnapshot.Sha256 `
            --identity-post $identityPostSnapshot.Path `
            --identity-post-sha256 $identityPostSnapshot.Sha256 `
            --config-evidence $configSnapshot.Path `
            --config-evidence-sha256 $configSnapshot.Sha256 `
            --device-evidence $deviceSnapshot.Path `
            --device-evidence-sha256 $deviceSnapshot.Sha256 `
            --model-inventory $modelInventorySnapshot.Path `
            --model-inventory-sha256 $modelInventorySnapshot.Sha256 `
            --model-manifest $modelManifest `
            --expected-source-commit $commitOid `
            --expected-model-manifest-sha256 $expectedModelManifestSha `
            --config $runtimeConfig `
            --expected-config-sha256 $expectedConfigSha `
            --source-repo $root `
            --expanded-source $source `
            --git-executable $gitExecutable `
            --git-executable-sha256 $gitExecutableSha `
            --stdout)
        $validatorExitCode = $LASTEXITCODE
        Assert-SnapshotDigest $trustedVerifierSnapshot $trustedVerifierSha
        if (
            $validatorExitCode -ne 0 -or
            $releaseValidationLines.Count -ne 1 -or
            [String]::IsNullOrWhiteSpace($releaseValidationLines[0])
        ) {
            throw 'Independent release-evidence validation failed.'
        }
        $releaseValidationText = $releaseValidationLines[0] + "`n"
        try {
            $releaseValidation = Microsoft.PowerShell.Utility\ConvertFrom-Json `
                -InputObject $releaseValidationText
        }
        catch {
            throw 'Release evidence validator returned malformed JSON.'
        }
        if (
            $releaseValidation.schema -cne 'cogni.release.validation.v2' -or
            $releaseValidation.status -cne 'passed' -or
            $releaseValidation.source_commit -cne $commitOid -or
            $releaseValidation.source_tree_digest -cnotmatch '^[0-9a-f]{64}$'
        ) {
            throw 'Release evidence validator returned an invalid result.'
        }
        $sourceTreeDigest = $releaseValidation.source_tree_digest

        foreach ($field in @(
            'model_tree_digest',
            'config_digest',
            'device_digest'
        )) {
            $value = $releaseValidation.$field
            if (-not ($value -is [string]) -or $value -cnotmatch '^[0-9a-f]{64}$') {
                throw "Release evidence validator returned invalid $field."
            }
        }
        $releaseValidationSnapshot = New-LockedUtf8TextSnapshot `
            $releaseValidationPath $releaseValidationText `
            'Release evidence validation result'
        $releaseValidationSha = $releaseValidationSnapshot.Sha256
        Assert-ClosedTreeInventory `
            $source $archivedTreeChecksums 'Pre-acceptance SUBJECT'
        & $python -I -S -B $archivedAcceptanceValidator `
            $archivedMasterChecklist
        $acceptanceExitCode = $LASTEXITCODE
        if ((Get-LowerSha256 $archivedAcceptanceValidator) -cne $acceptanceValidatorSha) {
            throw 'Archived acceptance validator changed during validation.'
        }
        if ($acceptanceExitCode -ne 0) {
            throw 'Archived master acceptance checklist validation failed.'
        }
        & $python -I -S -B $archivedOutstandingRenderer `
            --check `
            --source $archivedMasterChecklist `
            --output $archivedOutstandingChecklist
        $rendererExitCode = $LASTEXITCODE
        if ((Get-LowerSha256 $archivedOutstandingRenderer) -cne $outstandingRendererSha) {
            throw 'Archived checklist renderer changed during validation.'
        }
        if ($rendererExitCode -ne 0) {
            throw 'Archived outstanding checklist is not derived from the master ledger.'
        }
        & $python -I -S -B $archivedAcceptanceValidator `
            $acceptanceBundleSnapshot.Checklist `
            --release-attestation $attestationSnapshot.Path `
            --release-attestation-signature $signatureSnapshot.Path `
            --verifier-public-key $publicKeySnapshot.Path `
            --require-complete
        $effectiveAcceptanceExitCode = $LASTEXITCODE
        if ($effectiveAcceptanceExitCode -ne 0) {
            throw 'Detached effective acceptance checklist is not release-complete.'
        }
        Assert-ClosedTreeInventory `
            $source $archivedTreeChecksums 'Post-acceptance SUBJECT'

        # No externally visible release directory is created until release
        # evidence and both acceptance-ledger gates have passed exact scope.
        $publishStage = New-ReleasePublishStage $publishedOutput

        foreach ($snapshotContract in @(
            @($summarySnapshot, $summarySnapshot.Sha256, 'RELEASE_EVIDENCE_SUMMARY.json'),
            @($cpuSnapshot, $cpuSnapshot.Sha256, 'CPU_GATE_EVIDENCE.json'),
            @($gpuSnapshot, $gpuSnapshot.Sha256, 'GPU5_GATE_EVIDENCE.json'),
            @($attestationSnapshot, $attestationSnapshot.Sha256, 'RELEASE_ATTESTATION.json'),
            @($signatureSnapshot, $signatureSnapshot.Sha256, 'RELEASE_ATTESTATION.sig'),
            @($publicKeySnapshot, $publicKeySnapshot.Sha256, 'VERIFIER_PUBLIC_KEY.json'),
            @($runtimeSnapshot, $runtimeSnapshot.Sha256, 'GPU5_RUNTIME_EVIDENCE.json'),
            @($completionSnapshot, $completionSnapshot.Sha256, 'GPU5_COMPLETION_EVIDENCE.json'),
            @($identityPreSnapshot, $identityPreSnapshot.Sha256, 'GPU5_IDENTITY_PRE.json'),
            @($identityPostSnapshot, $identityPostSnapshot.Sha256, 'GPU5_IDENTITY_POST.json'),
            @($configSnapshot, $configSnapshot.Sha256, 'GPU5_CONFIG_EVIDENCE.json'),
            @($deviceSnapshot, $deviceSnapshot.Sha256, 'GPU5_DEVICE_EVIDENCE.json'),
            @($modelInventorySnapshot, $modelInventorySnapshot.Sha256, 'GPU5_MODEL_INVENTORY.json')
        )) {
            Copy-VerifiedSnapshotToBundle `
                $snapshotContract[0] $snapshotContract[1] `
                $snapshotContract[2] $publishStage
        }
        foreach ($acceptanceFile in $acceptanceBundleSnapshot.Files) {
            Copy-VerifiedSnapshotToBundle `
                $acceptanceFile.Snapshot $acceptanceFile.Snapshot.Sha256 `
                (Join-Path 'ACCEPTANCE_BUNDLE' $acceptanceFile.Relative) `
                $publishStage
        }
        Copy-Item -LiteralPath $archivedPolicy -Destination (
            Join-Path $publishStage 'RELEASE_VERIFIER_POLICY.json'
        )
        if ((Get-LowerSha256 (Join-Path $publishStage 'RELEASE_VERIFIER_POLICY.json')) -cne $verifierPolicySha) {
            throw 'Bundled release verifier policy changed after validation.'
        }
        Copy-VerifiedSnapshotToBundle `
            $releaseValidationSnapshot $releaseValidationSha `
            'RELEASE_EVIDENCE_VALIDATION.json' $publishStage
        $releaseEvidenceStatus = 'VERIFIED'
    }

    if (-not $PublishRelease) {
        $publishStage = New-ReleasePublishStage $publishedOutput
    }

    $sourceArchiveName = "Cogni-OS-2-Genesis-v$releaseVersion-source.zip"
    $bundledSourceArchive = Join-Path $publishStage $sourceArchiveName
    Copy-Item -LiteralPath $rawArchive -Destination $bundledSourceArchive
    if ((Get-LowerSha256 $bundledSourceArchive) -cne $rawArchiveSha) {
        throw 'Bundled source archive differs from the locked validated archive.'
    }

    # Preserve one clean expanded source tree beside the native launcher so
    # double-click discovery never depends on an untrusted external checkout.
    if ($PublishRelease) {
        Assert-ClosedTreeInventory `
            $source $archivedTreeChecksums 'Pre-package SUBJECT'
    }
    Copy-Item -LiteralPath $source -Destination $publishStage -Recurse
    $expandedSourceRoot = Join-Path $publishStage 'Cogni-OS-2-Genesis-source'
    $expandedTreeChecksums = @(Get-ClosedTreeChecksumLines $expandedSourceRoot)
    if (
        $archivedTreeChecksums.Count -ne $expandedTreeChecksums.Count -or
        @(Compare-Object `
            -ReferenceObject $archivedTreeChecksums `
            -DifferenceObject $expandedTreeChecksums `
            -CaseSensitive).Count -ne 0
    ) {
        throw 'Bundled expanded source inventory or bytes differ from the archived commit.'
    }
    $utf8 = [Text.UTF8Encoding]::new($false)
    Write-Utf8LfLines `
        (Join-Path $publishStage 'SOURCE_TREE_SHA256SUMS.txt') `
        $expandedTreeChecksums
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
        & $python @pythonBuildArguments -c (
            'from cogni_core.cts_policy import load_default_bounded_cts_controller;' +
            'load_default_bounded_cts_controller(device=''cpu'');' +
            'print(''checkpoint_preflight=PASS'')'
        )
        if ($LASTEXITCODE -ne 0) {
            throw 'Extracted CTS checkpoint preflight failed.'
        }

        $wheelDirectory = Join-Path $scratch 'wheel'
        New-Item -ItemType Directory -Path $wheelDirectory |
            Microsoft.PowerShell.Core\Out-Null
        & $python @pythonBuildArguments -m pip wheel . `
            --no-deps --no-build-isolation --wheel-dir $wheelDirectory
        if ($LASTEXITCODE -ne 0) {
            throw 'Wheel build failed.'
        }
        $wheelName = "cogni_os-$releaseVersion-py3-none-any.whl"
        Copy-Item -LiteralPath (Join-Path $wheelDirectory $wheelName) `
            -Destination (Join-Path $publishStage $wheelName)

        $exeName = "CogniBoard-v$releaseVersion.exe"
        & $powerShellExecutable -NoProfile -ExecutionPolicy Bypass `
            -File (Join-Path $source 'scripts\build_windows_launcher.ps1') `
            -OutputPath (Join-Path $publishStage $exeName)
        if ($LASTEXITCODE -ne 0) {
            throw 'Windows launcher build failed.'
        }

        & $python @pythonBuildArguments scripts\generate_release_sbom.py `
            --pyproject (Join-Path $source 'pyproject.toml') `
            --project-version $releaseVersion `
            --output (Join-Path $publishStage 'SBOM.cdx.json') `
            --notices (Join-Path $publishStage 'THIRD_PARTY_NOTICES.md') `
            --artifact (Join-Path $publishStage $sourceArchiveName) `
            --artifact (Join-Path $publishStage $wheelName) `
            --artifact (Join-Path $publishStage $exeName)
        if ($LASTEXITCODE -ne 0) {
            throw 'Release SBOM generation failed.'
        }

    }
    finally {
        Pop-Location
    }
    if ($PublishRelease) {
        Assert-ClosedTreeInventory `
            $source $archivedTreeChecksums 'Post-package SUBJECT'
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
        $manualPdfName = "COGNIBOARD_USER_MANUAL_PLAYBOOK_KO_v$releaseVersion.pdf"
        if ($PublishRelease) {
            Copy-VerifiedSnapshotToBundle `
                $manualPdfSnapshot $ManualPdfSha256 $manualPdfName $publishStage
        }
        else {
            $manualPdf = [IO.Path]::GetFullPath($ManualPdfPath)
            if (-not (Test-Path -LiteralPath $manualPdf -PathType Leaf)) {
                throw "Manual PDF does not exist: $manualPdf"
            }
            Assert-NoReparseAncestors (Split-Path -Parent $manualPdf)
            Copy-Item -LiteralPath $manualPdf -Destination (
                Join-Path $publishStage $manualPdfName
            )
        }
    }

    $pythonVersion = (
        & $python @pythonBuildArguments --version 2>&1 |
            Microsoft.PowerShell.Utility\Out-String
    ).Trim()
    if (
        $LASTEXITCODE -ne 0 -or
        $pythonVersion -cnotmatch '^Python [0-9]+\.[0-9]+\.[0-9]+(?:[A-Za-z0-9.+-]{0,32})?$'
    ) {
        throw 'Python application returned unsafe or malformed version metadata.'
    }
    $pipVersionOutput = (
        & $python @pythonBuildArguments -m pip --version 2>&1 |
            Microsoft.PowerShell.Utility\Out-String
    ).Trim()
    $pipVersionMatch = [regex]::Match(
        $pipVersionOutput,
        '^pip ([A-Za-z0-9][A-Za-z0-9._+-]{0,127}) from .+ \(python ([0-9]+\.[0-9]+)\)$'
    )
    if ($LASTEXITCODE -ne 0 -or -not $pipVersionMatch.Success) {
        throw 'Pip returned unsafe or malformed version metadata.'
    }
    $pipVersion = $pipVersionMatch.Groups[1].Value
    $pipPythonVersion = $pipVersionMatch.Groups[2].Value
    if ((Get-LowerSha256 $python) -cne $pythonExecutableSha) {
        throw 'Python application bytes changed during artifact assembly.'
    }
    $manifestLines = @(
        "release_version=$releaseVersion",
        "commit_oid=$commitOid",
        "commit_epoch=$commitEpoch",
        "source_archive=$sourceArchiveName",
        "source_archive_sha256=$(Get-LowerSha256 (Join-Path $publishStage $sourceArchiveName))",
        "cts_checkpoint_sha256=$archiveCheckpoint",
        "expanded_source_tree_manifest=SOURCE_TREE_SHA256SUMS.txt",
        "expanded_source_tree_manifest_sha256=$(Get-LowerSha256 (Join-Path $publishStage 'SOURCE_TREE_SHA256SUMS.txt'))",
        "expanded_source_tree_manifest_content_sha256=$sourceChecksumManifestContentSha",
        "expanded_source_file_count=$($expandedTreeChecksums.Count)",
        "artifact_build_status=PASS",
        "release_evidence_status=$releaseEvidenceStatus",
        "python_executable_name=$pythonExecutableName",
        "python_executable_sha256=$pythonExecutableSha",
        "python_version=$pythonVersion",
        "pip_version=$pipVersion",
        "pip_python_version=$pipPythonVersion",
        "source_date_epoch=$commitEpoch"
        "signature_status=unsigned-no-code-signing-certificate-provided"
        "sbom=SBOM.cdx.json"
        "third_party_notices=THIRD_PARTY_NOTICES.md"
    )
    if ($PublishRelease) {
        $manifestLines += @(
            "release_evidence_summary=RELEASE_EVIDENCE_SUMMARY.json",
            "release_evidence_summary_sha256=$releaseEvidenceSummarySha",
            "cpu_gate_raw_evidence=CPU_GATE_EVIDENCE.json",
            "cpu_gate_raw_evidence_sha256=$($releaseValidation.cpu_evidence_sha256)",
            "gpu5_gate_raw_evidence=GPU5_GATE_EVIDENCE.json",
            "gpu5_gate_raw_evidence_sha256=$($releaseValidation.gpu5_evidence_sha256)",
            "guard_source_tree_digest=$sourceTreeDigest",
            "gpu5_source_tree_digest=$($releaseValidation.source_tree_digest)",
            "gpu5_model_tree_digest=$($releaseValidation.model_tree_digest)",
            "gpu5_config_digest=$($releaseValidation.config_digest)",
            "gpu5_device_digest=$($releaseValidation.device_digest)",
            "gpu5_runtime_evidence=GPU5_RUNTIME_EVIDENCE.json",
            "gpu5_runtime_evidence_sha256=$($releaseValidation.runtime_evidence_sha256)",
            "gpu5_completion_evidence=GPU5_COMPLETION_EVIDENCE.json",
            "gpu5_completion_evidence_sha256=$($releaseValidation.completion_evidence_sha256)",
            "gpu5_identity_pre=GPU5_IDENTITY_PRE.json",
            "gpu5_identity_pre_sha256=$($releaseValidation.identity_pre_sha256)",
            "gpu5_identity_post=GPU5_IDENTITY_POST.json",
            "gpu5_identity_post_sha256=$($releaseValidation.identity_post_sha256)",
            "gpu5_config_evidence=GPU5_CONFIG_EVIDENCE.json",
            "gpu5_config_evidence_sha256=$($releaseValidation.config_evidence_sha256)",
            "gpu5_device_evidence=GPU5_DEVICE_EVIDENCE.json",
            "gpu5_device_evidence_sha256=$($releaseValidation.device_evidence_sha256)",
            "gpu5_model_inventory=GPU5_MODEL_INVENTORY.json",
            "gpu5_model_inventory_sha256=$($releaseValidation.model_inventory_sha256)",
            "release_attestation=RELEASE_ATTESTATION.json",
            "release_attestation_sha256=$($releaseValidation.attestation_sha256)",
            "release_attestation_signature=RELEASE_ATTESTATION.sig",
            "release_attestation_signature_sha256=$($releaseValidation.attestation_signature_sha256)",
            "verifier_public_key=VERIFIER_PUBLIC_KEY.json",
            "verifier_public_key_sha256=$($releaseValidation.verifier_public_key_sha256)",
            "release_verifier_policy=RELEASE_VERIFIER_POLICY.json",
            "release_verifier_policy_sha256=$($releaseValidation.verifier_policy_sha256)",
            "release_evidence_validation=RELEASE_EVIDENCE_VALIDATION.json",
            "release_evidence_validation_sha256=$releaseValidationSha",
            "acceptance_bundle=ACCEPTANCE_BUNDLE",
            "acceptance_bundle_inventory_sha256=$acceptanceBundleDigest",
            "independent_verifier_id=$($releaseValidation.verifier_id)"
        )
    }
    [IO.File]::WriteAllLines(
        (Join-Path $publishStage 'BUILD_MANIFEST.txt'),
        $manifestLines,
        $utf8
    )

    $checksumPath = Join-Path $publishStage 'SHA256SUMS.txt'
    if (Test-Path -LiteralPath $checksumPath) {
        throw 'Release stage unexpectedly already contains SHA256SUMS.txt.'
    }
    $payloadStageInventory = @(Get-ClosedTreeChecksumLines $publishStage)
    $payloadStageReadLocks = @(
        Open-VerifiedSourceReadLocks $publishStage $payloadStageInventory
    )
    $checksumLines = @(
        Get-LockedTreeChecksumLines $payloadStageReadLocks $payloadStageInventory
    )
    Write-Utf8LfLines $checksumPath $checksumLines

    # Admit and lock the exact final inventory, including SHA256SUMS.txt.  A
    # second inventory detects insertions while the locks are acquired.  The
    # protected isolated runner remains responsible for excluding a hostile
    # writer during the necessarily small unlock/rename interval.
    $finalStageInventory = @(Get-ClosedTreeChecksumLines $publishStage)
    $finalStageReadLocks = @(
        Open-VerifiedSourceReadLocks $publishStage $finalStageInventory
    )
    Assert-ClosedTreeInventory `
        $publishStage $finalStageInventory 'Final release stage'
    foreach ($handle in $finalStageReadLocks) { $handle.Dispose() }
    $finalStageReadLocks = @()
    foreach ($handle in $payloadStageReadLocks) { $handle.Dispose() }
    $payloadStageReadLocks = @()
    Move-Item -LiteralPath $publishStage -Destination $publishedOutput
    $publishStage = $null
    Assert-ClosedTreeInventory `
        $publishedOutput $finalStageInventory 'Published release output'
    $postMoveReadLocks = @(
        Open-VerifiedSourceReadLocks $publishedOutput $finalStageInventory
    )
    try {
        Assert-ClosedTreeInventory `
            $publishedOutput $finalStageInventory 'Locked published release output'
    }
    finally {
        foreach ($handle in $postMoveReadLocks) { $handle.Dispose() }
    }
    $buildSucceeded = $true
}
finally {
    $env:SOURCE_DATE_EPOCH = $oldSourceDateEpoch
    foreach ($entry in [Environment]::GetEnvironmentVariables('Process').GetEnumerator()) {
        $name = [string]$entry.Key
        if ($name.StartsWith('PYTHON', [StringComparison]::OrdinalIgnoreCase)) {
            [Environment]::SetEnvironmentVariable($name, $null, 'Process')
        }
    }
    foreach ($name in $savedPythonEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable(
            $name,
            $savedPythonEnvironment[$name],
            'Process'
        )
    }
    foreach ($entry in [Environment]::GetEnvironmentVariables('Process').GetEnumerator()) {
        $name = [string]$entry.Key
        if ($name.StartsWith('GIT_', [StringComparison]::OrdinalIgnoreCase)) {
            [Environment]::SetEnvironmentVariable($name, $null, 'Process')
        }
    }
    foreach ($name in $savedGitEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable(
            $name,
            $savedGitEnvironment[$name],
            'Process'
        )
    }
    foreach ($sourceReadLock in $sourceReadLocks) { $sourceReadLock.Dispose() }
    foreach ($stageReadLock in $finalStageReadLocks) { $stageReadLock.Dispose() }
    foreach ($stageReadLock in $payloadStageReadLocks) { $stageReadLock.Dispose() }
    foreach ($snapshotHandle in $verifiedSnapshotHandles) { $snapshotHandle.Dispose() }
    if ($null -ne $rawArchiveLock) { $rawArchiveLock.Dispose() }
    if ($null -ne $pinnedGitHandle) { $pinnedGitHandle.Dispose() }
    if ($null -ne $pinnedPythonHandle) { $pinnedPythonHandle.Dispose() }
    if ($null -ne $pinnedPowerShellHandle) { $pinnedPowerShellHandle.Dispose() }
    if ($null -ne $toolchainPolicyHandle) { $toolchainPolicyHandle.Dispose() }
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
        Microsoft.PowerShell.Utility\Write-Warning `
            ("Release cleanup also failed: " + $_.Exception.Message)
    }
}

Microsoft.PowerShell.Utility\Write-Output "release_output=$publishedOutput"
Microsoft.PowerShell.Utility\Write-Output "source_archive=$sourceArchiveName"
Microsoft.PowerShell.Utility\Write-Output "checkpoint_sha256=$archiveCheckpoint"
Microsoft.PowerShell.Utility\Write-Output "commit_oid=$commitOid"
Microsoft.PowerShell.Utility\Write-Output "release_version=$releaseVersion"
Microsoft.PowerShell.Utility\Write-Output 'artifact_build=PASS'
Microsoft.PowerShell.Utility\Write-Output `
    "release_evidence_status=$releaseEvidenceStatus"
