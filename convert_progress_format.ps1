# One-time migration; the trainer itself does not accept the old progress.bin format.
# Valid only for full-material games (two rooks, including dragons and held rooks).
param(
    [Parameter(Mandatory = $true)][string]$InputPath,
    [Parameter(Mandatory = $true)][string]$OutputPath
)
$ErrorActionPreference = 'Stop'
$source = (Resolve-Path -LiteralPath $InputPath).Path
$destination = [IO.Path]::GetFullPath($OutputPath)
$bytes = [IO.File]::ReadAllBytes($source)
$count = 81 * 1548
if ($bytes.Length -ne (8 + $count * 4) -or [BitConverter]::ToUInt32($bytes, 0) -ne 0x6f50524f) {
    throw 'Expected the old header + bias + q16 weights progress.bin (501,560 bytes).'
}
$bias = [BitConverter]::ToInt32($bytes, 4)
if (($bias % 4) -ne 0) { throw 'Exact q16 migration requires bias divisible by 4.' }
$delta = $bias / 4
$result = [byte[]]::new($count * 8)
$changed = 0
for ($i = 0; $i -lt $count; $i++) {
    $piece = $i % 1548
    [long]$weight = [BitConverter]::ToInt32($bytes, 8 + 4 * $i)
    if ($piece -ge 1224 -or ($piece -ge 85 -and $piece -lt 87) -or ($piece -ge 88 -and $piece -lt 90)) {
        $weight += $delta
        $changed++
    }
    if ($weight -lt [int]::MinValue -or $weight -gt [int]::MaxValue) { throw 'q16 weight overflow.' }
    $encoded = [BitConverter]::GetBytes([double]$weight / 65536.0)
    if (-not [BitConverter]::IsLittleEndian) { [Array]::Reverse($encoded) }
    [Array]::Copy($encoded, 0, $result, $i * 8, 8)
}
if ([IO.File]::Exists($destination)) {
    if (-not $source.Equals($destination, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Output already exists: $destination"
    }
    $backup = $destination + '.q16-with-bias.bak'
    if ([IO.File]::Exists($backup)) { throw "Backup already exists: $backup" }
    [IO.File]::Copy($source, $backup, $false)
    Write-Output "Backup: $backup"
}
$temporary = $destination + '.converting'
if ([IO.File]::Exists($temporary)) { throw "Temporary file already exists: $temporary" }
[IO.File]::WriteAllBytes($temporary, $result)
if ([IO.File]::Exists($destination)) {
    [IO.File]::Replace($temporary, $destination, [NullString]::Value)
} else {
    [IO.File]::Move($temporary, $destination)
}
Write-Output "Converted: $destination"
Write-Output "Size=$($result.Length); bias_q16=$bias; rook_term_delta=$delta; adjusted_weights=$changed"
Write-Output 'Integer sums and progress buckets are unchanged for two-rook positions. Rook-odds positions are NOT preserved.'
