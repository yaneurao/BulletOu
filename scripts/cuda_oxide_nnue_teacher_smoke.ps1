<#
.SYNOPSIS
Generate a HalfKP NNUE train fixture from one teacher file, then run the
cuda-oxide NNUE loss/Ranger step smoke against it in WSL2.

.DESCRIPTION
This script intentionally keeps the root BulletOu workspace and the nested
cuda-oxide workspace separate:

1. Windows/root workspace: run export_nnue_forward_fixture --train-fixture to
   materialise one real teacher batch, targets, entry weights, and weights into
   a local ignored fixture file under target/.
2. WSL2/cuda-oxide workspace: build/load the cuda-oxide kernels and run
   --nnue-loss-ranger-step-smoke against that fixture.
   Pass -RunFixtureTrain to also run the non-comparing --nnue-fixture-train
   loop against the same fixtures.
   Pass -TrainedForwardFixture with -RunFixtureTrain to write the final
   trained weights as a BOUNFWD1 forward fixture.
   Pass -TrainStateFixture with -RunFixtureTrain to write the final weights
   and Ranger optimizer state as a BOUNRNG1 train-state fixture.
   Pass -ResumeTrainStateFixture to restore a BOUNRNG1 train-state fixture and
   export/apply only the later teacher batches up to -TrainSteps.
   Pass -RunDirectTeacherTrain to also run the cuda-oxide root-loader bridge
   that reads teacher batches directly without intermediate train fixtures.
   When combined with -ResumeTrainStateFixture, the direct path also resumes
   from that BOUNRNG1 state and runs the remaining batches up to -TrainSteps.
   Pass -WeightsBin to initialise fixture/direct fresh runs from root
   weights.bin or bundled state.bin weights instead of deterministic weights.
   Pass -Output with -RunDirectTeacherTrain to write numbered cuda-oxide bridge
   checkpoints containing nn.bin, trained-forward.nnuef, state.boung, root
   state.bin, dataloader_pos.txt, and learn.log.
   Pass -SaveRate <N> with -Output to write a bridge checkpoint every N direct
   teacher batches. The default 0 keeps the historical final-only checkpoint.

The WSL nvJitLink shim is temporary. Ubuntu's CUDA 12.0 libnvJitLink exposes
versioned symbols, while the current cuda-oxide revision expects unversioned
symbols.
#>

[CmdletBinding()]
param(
    [string]$Teacher,
    [int]$BatchSize = 2,
    [int]$BufferMb = 1,
    [int]$LoaderThreads = 1,
    [int]$Threads = 1,
    [int]$ScoreDropAbs = 32000,
    [string]$WeightsBin,
    [string]$Fixture,
    [string]$WslDistro = "Ubuntu-24.04",
    [string]$CudaArch = "sm_89",
    [ValidateSet("sigmoid-mse", "wrm")]
    [string]$LossKind = "sigmoid-mse",
    [int]$TrainSteps = 1,
    [int]$SaveRate = 0,
    [switch]$SkipCudaBuild,
    [switch]$DebugReadback,
    [switch]$RunFixtureTrain,
    [string]$TrainedForwardFixture,
    [string]$TrainStateFixture,
    [string]$ResumeTrainStateFixture,
    [switch]$RunDirectTeacherTrain,
    [string]$DirectTrainedForwardFixture,
    [string]$DirectTrainStateFixture,
    [string]$Output
)

$ErrorActionPreference = "Stop"

function Convert-ToWslPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $resolved = (Resolve-Path -LiteralPath $Path).Path
    if ($resolved -match '^([A-Za-z]):\\(.*)$') {
        $drive = $Matches[1].ToLowerInvariant()
        $rest = $Matches[2] -replace '\\', '/'
        return "/mnt/$drive/$rest"
    }

    throw "Cannot convert path to WSL form: $resolved"
}

function Convert-TeacherSpecToWslPath {
    param([Parameter(Mandatory = $true)][string]$TeacherSpec)

    $parts = @($TeacherSpec -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_.Length -gt 0 })
    if ($parts.Count -eq 0) {
        throw "Teacher spec is empty"
    }
    return (($parts | ForEach-Object { Convert-ToWslPath $_ }) -join ",")
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][scriptblock]$Command
    )

    Write-Host "==> $Label"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

function Read-ExactBytes {
    param(
        [Parameter(Mandatory = $true)][System.IO.Stream]$Stream,
        [Parameter(Mandatory = $true)][int]$Count,
        [Parameter(Mandatory = $true)][string]$Name
    )

    $bytes = [byte[]]::new($Count)
    $offset = 0
    while ($offset -lt $Count) {
        $read = $Stream.Read($bytes, $offset, $Count - $offset)
        if ($read -eq 0) {
            throw "Unexpected EOF while reading $Name"
        }
        $offset += $read
    }
    return $bytes
}

function Read-NnueTrainStateCompletedSteps {
    param([Parameter(Mandatory = $true)][string]$Path)

    $resolved = (Resolve-Path -LiteralPath $Path).Path
    $stream = [System.IO.File]::OpenRead($resolved)
    try {
        $magic = [System.Text.Encoding]::ASCII.GetString((Read-ExactBytes $stream 8 "BOUNRNG1 magic"))
        if ($magic -ne "BOUNRNG1") {
            throw "Invalid BOUNRNG1 train-state fixture magic: $resolved"
        }

        $values = @()
        for ($i = 0; $i -lt 5; $i++) {
            $values += [System.BitConverter]::ToUInt64((Read-ExactBytes $stream 8 "BOUNRNG1 header value $i"), 0)
        }

        if ($values[4] -gt [int]::MaxValue) {
            throw "completed_steps is too large for this script: $($values[4])"
        }
        return [int]$values[4]
    }
    finally {
        $stream.Dispose()
    }
}

$scriptDir = Split-Path -Parent $PSCommandPath
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $scriptDir "..")).Path
$cudaRoot = Join-Path $repoRoot "cuda-oxide"

if ([string]::IsNullOrWhiteSpace($Teacher)) {
    $defaultTeacherDir = "C:\shogi\teacher\yane-distill-hcpe-20260508shuffled"
    if (Test-Path -LiteralPath $defaultTeacherDir) {
        $Teacher = (Get-ChildItem -LiteralPath $defaultTeacherDir -File -Filter "*.hcpe" |
            Sort-Object Name |
            Select-Object -First 1).FullName
    }
}

if ([string]::IsNullOrWhiteSpace($Teacher)) {
    throw "Specify -Teacher, or place .hcpe files under C:\shogi\teacher\yane-distill-hcpe-20260508shuffled"
}

if ($TrainSteps -lt 1) {
    throw "-TrainSteps must be >= 1"
}
if ($SaveRate -lt 0) {
    throw "-SaveRate must be >= 0"
}
if ($SaveRate -gt 0 -and [string]::IsNullOrWhiteSpace($Output)) {
    throw "-SaveRate requires -Output"
}

if (-not [string]::IsNullOrWhiteSpace($TrainedForwardFixture)) {
    $RunFixtureTrain = $true
}

if (-not [string]::IsNullOrWhiteSpace($TrainStateFixture)) {
    $RunFixtureTrain = $true
}

if (-not [string]::IsNullOrWhiteSpace($ResumeTrainStateFixture)) {
    $RunFixtureTrain = $true
}

if (-not [string]::IsNullOrWhiteSpace($DirectTrainedForwardFixture)) {
    $RunDirectTeacherTrain = $true
}

if (-not [string]::IsNullOrWhiteSpace($DirectTrainStateFixture)) {
    $RunDirectTeacherTrain = $true
}
if (-not [string]::IsNullOrWhiteSpace($Output)) {
    $RunDirectTeacherTrain = $true
}

if (-not (Test-Path -LiteralPath $Teacher)) {
    throw "Teacher file not found: $Teacher"
}

if (-not [string]::IsNullOrWhiteSpace($WeightsBin) -and -not (Test-Path -LiteralPath $WeightsBin)) {
    throw "Weights file not found: $WeightsBin"
}

if (-not [string]::IsNullOrWhiteSpace($WeightsBin) -and -not [string]::IsNullOrWhiteSpace($ResumeTrainStateFixture)) {
    throw "-WeightsBin cannot be combined with -ResumeTrainStateFixture; the train-state fixture already contains weights"
}

if (-not [string]::IsNullOrWhiteSpace($ResumeTrainStateFixture) -and -not (Test-Path -LiteralPath $ResumeTrainStateFixture)) {
    throw "Resume train-state fixture not found: $ResumeTrainStateFixture"
}

if (-not [string]::IsNullOrWhiteSpace($Output)) {
    New-Item -ItemType Directory -Force -Path $Output | Out-Null
}

if ([string]::IsNullOrWhiteSpace($Fixture)) {
    $Fixture = Join-Path $repoRoot "target\cuda-oxide-fixtures\nnue-halfkp-teacher-train-b$BatchSize.bin"
}

$resumeCompletedSteps = 0
if (-not [string]::IsNullOrWhiteSpace($ResumeTrainStateFixture)) {
    $resumeCompletedSteps = Read-NnueTrainStateCompletedSteps $ResumeTrainStateFixture
    if ($TrainSteps -le $resumeCompletedSteps) {
        throw "-TrainSteps ($TrainSteps) must be greater than completed_steps ($resumeCompletedSteps) in -ResumeTrainStateFixture"
    }
}

$fixtureDir = Split-Path -Parent $Fixture
if ([string]::IsNullOrWhiteSpace($fixtureDir)) {
    $fixtureDir = "."
}
$fixtureStem = [System.IO.Path]::GetFileNameWithoutExtension($Fixture)
$fixtureExt = [System.IO.Path]::GetExtension($Fixture)
$firstFixtureStep = if ([string]::IsNullOrWhiteSpace($ResumeTrainStateFixture)) { 0 } else { $resumeCompletedSteps }
$fixtureSpecs = @()
for ($step = $firstFixtureStep; $step -lt $TrainSteps; $step++) {
    $path = if ([string]::IsNullOrWhiteSpace($ResumeTrainStateFixture) -and $TrainSteps -eq 1 -and $step -eq 0) {
        $Fixture
    } else {
        Join-Path $fixtureDir "$fixtureStem-step$step$fixtureExt"
    }
    $fixtureSpecs += [pscustomobject]@{
        Step = $step
        Path = $path
    }
}
$fixturePaths = @($fixtureSpecs | ForEach-Object { $_.Path })

foreach ($path in $fixturePaths) {
    $fixtureDir = Split-Path -Parent $path
    New-Item -ItemType Directory -Force -Path $fixtureDir | Out-Null
}

if (-not [string]::IsNullOrWhiteSpace($TrainedForwardFixture)) {
    $trainedFixtureDir = Split-Path -Parent $TrainedForwardFixture
    if (-not [string]::IsNullOrWhiteSpace($trainedFixtureDir)) {
        New-Item -ItemType Directory -Force -Path $trainedFixtureDir | Out-Null
    }
    New-Item -ItemType File -Force -Path $TrainedForwardFixture | Out-Null
}

if (-not [string]::IsNullOrWhiteSpace($TrainStateFixture)) {
    $trainStateFixtureDir = Split-Path -Parent $TrainStateFixture
    if (-not [string]::IsNullOrWhiteSpace($trainStateFixtureDir)) {
        New-Item -ItemType Directory -Force -Path $trainStateFixtureDir | Out-Null
    }
    New-Item -ItemType File -Force -Path $TrainStateFixture | Out-Null
}

if (-not [string]::IsNullOrWhiteSpace($DirectTrainedForwardFixture)) {
    $directTrainedFixtureDir = Split-Path -Parent $DirectTrainedForwardFixture
    if (-not [string]::IsNullOrWhiteSpace($directTrainedFixtureDir)) {
        New-Item -ItemType Directory -Force -Path $directTrainedFixtureDir | Out-Null
    }
    New-Item -ItemType File -Force -Path $DirectTrainedForwardFixture | Out-Null
}

if (-not [string]::IsNullOrWhiteSpace($DirectTrainStateFixture)) {
    $directTrainStateFixtureDir = Split-Path -Parent $DirectTrainStateFixture
    if (-not [string]::IsNullOrWhiteSpace($directTrainStateFixtureDir)) {
        New-Item -ItemType Directory -Force -Path $directTrainStateFixtureDir | Out-Null
    }
    New-Item -ItemType File -Force -Path $DirectTrainStateFixture | Out-Null
}

$weightsExportArgs = @()
if (-not [string]::IsNullOrWhiteSpace($WeightsBin)) {
    $weightsExportArgs = @("--weights-bin", $WeightsBin)
}

foreach ($spec in $fixtureSpecs) {
    $step = [int]$spec.Step
    $outFixture = $spec.Path
    $fixtureWeightsArgs = if ([string]::IsNullOrWhiteSpace($ResumeTrainStateFixture) -and $step -eq 0) {
        $weightsExportArgs
    } else {
        @()
    }
    Invoke-Checked "export NNUE HalfKP teacher train fixture batch $step" {
        Set-Location $repoRoot
        $fixtureKindFlag = if ([string]::IsNullOrWhiteSpace($ResumeTrainStateFixture) -and $step -eq 0) {
            "--train-fixture"
        } else {
            "--batch-fixture"
        }
        cargo run -p bulletou_lib --example export_nnue_forward_fixture --release -- `
            --out $outFixture `
            $fixtureKindFlag `
            --case halfkp `
            $fixtureWeightsArgs `
            --teacher $Teacher `
            --batch-size $BatchSize `
            --batch-index $step `
            --buffer-mb $BufferMb `
            --loader-threads $LoaderThreads `
            --threads $Threads `
            --score-drop-abs $ScoreDropAbs
    }
}

$wslCudaRoot = Convert-ToWslPath $cudaRoot
$wslTeacher = Convert-TeacherSpecToWslPath $Teacher
$wslFixtures = @($fixturePaths | ForEach-Object { Convert-ToWslPath $_ })
$trainedForwardArg = ""
if (-not [string]::IsNullOrWhiteSpace($TrainedForwardFixture)) {
    $wslTrainedForwardFixture = Convert-ToWslPath $TrainedForwardFixture
    $trainedForwardArg = " --write-nnue-trained-forward-fixture `"$wslTrainedForwardFixture`""
}
$trainStateArg = ""
if (-not [string]::IsNullOrWhiteSpace($TrainStateFixture)) {
    $wslTrainStateFixture = Convert-ToWslPath $TrainStateFixture
    $trainStateArg = " --write-nnue-train-state-fixture `"$wslTrainStateFixture`""
}
$resumeTrainStateArg = ""
if (-not [string]::IsNullOrWhiteSpace($ResumeTrainStateFixture)) {
    $wslResumeTrainStateFixture = Convert-ToWslPath $ResumeTrainStateFixture
    $resumeTrainStateArg = " --nnue-train-state-fixture `"$wslResumeTrainStateFixture`""
}
$directTrainedForwardArg = ""
if (-not [string]::IsNullOrWhiteSpace($DirectTrainedForwardFixture)) {
    $wslDirectTrainedForwardFixture = Convert-ToWslPath $DirectTrainedForwardFixture
    $directTrainedForwardArg = " --write-nnue-trained-forward-fixture `"$wslDirectTrainedForwardFixture`""
}
$directTrainStateArg = ""
if (-not [string]::IsNullOrWhiteSpace($DirectTrainStateFixture)) {
    $wslDirectTrainStateFixture = Convert-ToWslPath $DirectTrainStateFixture
    $directTrainStateArg = " --write-nnue-train-state-fixture `"$wslDirectTrainStateFixture`""
}
$directWeightsArg = ""
if (-not [string]::IsNullOrWhiteSpace($WeightsBin)) {
    $wslWeightsBin = Convert-ToWslPath $WeightsBin
    $directWeightsArg = " --weights-bin `"$wslWeightsBin`""
}
$directOutputArg = ""
if (-not [string]::IsNullOrWhiteSpace($Output)) {
    $wslOutput = Convert-ToWslPath $Output
    $directOutputArg = " --output `"$wslOutput`""
}
$directSaveRateArg = ""
if ($SaveRate -gt 0) {
    $directSaveRateArg = " --save-rate $SaveRate"
}
$directResumeTrainStateArg = ""
$directTrainSteps = $TrainSteps
if (-not [string]::IsNullOrWhiteSpace($ResumeTrainStateFixture)) {
    $directResumeTrainStateArg = " --nnue-train-state-fixture `"$wslResumeTrainStateFixture`""
    $directTrainSteps = $TrainSteps - $resumeCompletedSteps
}
$debugFlag = if ($DebugReadback) { " --debug-readback" } else { "" }
$fixtureArgsList = @()
if (-not [string]::IsNullOrWhiteSpace($ResumeTrainStateFixture)) {
    $fixtureArgsList += $resumeTrainStateArg.Trim()
}
for ($i = 0; $i -lt $fixtureSpecs.Count; $i++) {
    $step = [int]$fixtureSpecs[$i].Step
    if ([string]::IsNullOrWhiteSpace($ResumeTrainStateFixture) -and $step -eq 0) {
        $fixtureArgsList += "--nnue-train-fixture `"$($wslFixtures[$i])`""
    } else {
        $fixtureArgsList += "--nnue-train-batch-fixture `"$($wslFixtures[$i])`""
    }
}
$fixtureArgs = $fixtureArgsList -join " "

$cudaEnv = @"
export CUDA_HOME=/usr
export CUDA_PATH=/usr
export CUDA_TOOLKIT_PATH=/usr
export CUDA_OXIDE_LIBDEVICE=/usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc
export LIBCLANG_PATH=/usr/lib/llvm-20/lib
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu
export CARGO_TARGET_DIR=/tmp/bulletou-cuda-target
"@

if (-not $SkipCudaBuild) {
    $buildCommand = @"
cd "$wslCudaRoot"
$cudaEnv
cargo oxide build --arch $CudaArch --features cuda -- --package bulletou-cuda-train --release
"@
    Invoke-Checked "cargo-oxide build ($WslDistro, $CudaArch)" {
        wsl -d $WslDistro -- bash -lc $buildCommand
    }
}

$shim = @'
#include <stddef.h>
#include <stdint.h>
typedef int nvJitLinkResult;
typedef int nvJitLinkInputType;
typedef struct nvJitLink* nvJitLinkHandle;
extern nvJitLinkResult __nvJitLinkCreate_12_0(nvJitLinkHandle *handle, uint32_t numOptions, const char **options);
extern nvJitLinkResult __nvJitLinkDestroy_12_0(nvJitLinkHandle *handle);
extern nvJitLinkResult __nvJitLinkAddData_12_0(nvJitLinkHandle handle, nvJitLinkInputType inputType, const void *data, size_t size, const char *name);
extern nvJitLinkResult __nvJitLinkAddFile_12_0(nvJitLinkHandle handle, nvJitLinkInputType inputType, const char *fileName);
extern nvJitLinkResult __nvJitLinkComplete_12_0(nvJitLinkHandle handle);
extern nvJitLinkResult __nvJitLinkGetLinkedCubinSize_12_0(nvJitLinkHandle handle, size_t *size);
extern nvJitLinkResult __nvJitLinkGetLinkedCubin_12_0(nvJitLinkHandle handle, void *cubin);
extern nvJitLinkResult __nvJitLinkGetLinkedPtxSize_12_0(nvJitLinkHandle handle, size_t *size);
extern nvJitLinkResult __nvJitLinkGetLinkedPtx_12_0(nvJitLinkHandle handle, char *ptx);
extern nvJitLinkResult __nvJitLinkGetErrorLogSize_12_0(nvJitLinkHandle handle, size_t *size);
extern nvJitLinkResult __nvJitLinkGetErrorLog_12_0(nvJitLinkHandle handle, char *log);
extern nvJitLinkResult __nvJitLinkGetInfoLogSize_12_0(nvJitLinkHandle handle, size_t *size);
extern nvJitLinkResult __nvJitLinkGetInfoLog_12_0(nvJitLinkHandle handle, char *log);
nvJitLinkResult nvJitLinkCreate(nvJitLinkHandle *handle, uint32_t numOptions, const char **options) { return __nvJitLinkCreate_12_0(handle, numOptions, options); }
nvJitLinkResult nvJitLinkDestroy(nvJitLinkHandle *handle) { return __nvJitLinkDestroy_12_0(handle); }
nvJitLinkResult nvJitLinkAddData(nvJitLinkHandle handle, nvJitLinkInputType inputType, const void *data, size_t size, const char *name) { return __nvJitLinkAddData_12_0(handle, inputType, data, size, name); }
nvJitLinkResult nvJitLinkAddFile(nvJitLinkHandle handle, nvJitLinkInputType inputType, const char *fileName) { return __nvJitLinkAddFile_12_0(handle, inputType, fileName); }
nvJitLinkResult nvJitLinkComplete(nvJitLinkHandle handle) { return __nvJitLinkComplete_12_0(handle); }
nvJitLinkResult nvJitLinkGetLinkedCubinSize(nvJitLinkHandle handle, size_t *size) { return __nvJitLinkGetLinkedCubinSize_12_0(handle, size); }
nvJitLinkResult nvJitLinkGetLinkedCubin(nvJitLinkHandle handle, void *cubin) { return __nvJitLinkGetLinkedCubin_12_0(handle, cubin); }
nvJitLinkResult nvJitLinkGetLinkedPtxSize(nvJitLinkHandle handle, size_t *size) { return __nvJitLinkGetLinkedPtxSize_12_0(handle, size); }
nvJitLinkResult nvJitLinkGetLinkedPtx(nvJitLinkHandle handle, char *ptx) { return __nvJitLinkGetLinkedPtx_12_0(handle, ptx); }
nvJitLinkResult nvJitLinkGetErrorLogSize(nvJitLinkHandle handle, size_t *size) { return __nvJitLinkGetErrorLogSize_12_0(handle, size); }
nvJitLinkResult nvJitLinkGetErrorLog(nvJitLinkHandle handle, char *log) { return __nvJitLinkGetErrorLog_12_0(handle, log); }
nvJitLinkResult nvJitLinkGetInfoLogSize(nvJitLinkHandle handle, size_t *size) { return __nvJitLinkGetInfoLogSize_12_0(handle, size); }
nvJitLinkResult nvJitLinkGetInfoLog(nvJitLinkHandle handle, char *log) { return __nvJitLinkGetInfoLog_12_0(handle, log); }
'@

$runCommand = @"
cat > /tmp/nvjitlink_shim.c &&
gcc -shared -fPIC -o /tmp/libnvJitLink_shim.so /tmp/nvjitlink_shim.c -L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/usr/lib/x86_64-linux-gnu -lnvJitLink &&
cd "$wslCudaRoot"
$cudaEnv
export LIBNVJITLINK_PATH=/tmp/libnvJitLink_shim.so
cargo run -p bulletou-cuda-train --features cuda --release -- --nnue-loss-ranger-step-smoke $fixtureArgs --loss-kind $LossKind$debugFlag
"@

if ([string]::IsNullOrWhiteSpace($ResumeTrainStateFixture)) {
    Invoke-Checked "NNUE loss Ranger step smoke with real teacher fixture" {
        $shim | wsl -d $WslDistro -- bash -lc $runCommand
    }
} else {
    Write-Host "==> skip CPU-golden loss smoke when restoring BOUNRNG1 state"
}

if ($RunFixtureTrain) {
    $fixtureTrainCommand = @"
cat > /tmp/nvjitlink_shim.c &&
gcc -shared -fPIC -o /tmp/libnvJitLink_shim.so /tmp/nvjitlink_shim.c -L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/usr/lib/x86_64-linux-gnu -lnvJitLink &&
cd "$wslCudaRoot"
$cudaEnv
export LIBNVJITLINK_PATH=/tmp/libnvJitLink_shim.so
cargo run -p bulletou-cuda-train --features cuda --release -- --nnue-fixture-train $fixtureArgs --loss-kind $LossKind$debugFlag$trainedForwardArg$trainStateArg
"@

    Invoke-Checked "NNUE fixture train loop with real teacher fixtures" {
        $shim | wsl -d $WslDistro -- bash -lc $fixtureTrainCommand
    }
}

if ($RunDirectTeacherTrain) {
    $directTeacherCommand = @"
cat > /tmp/nvjitlink_shim.c &&
gcc -shared -fPIC -o /tmp/libnvJitLink_shim.so /tmp/nvjitlink_shim.c -L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/usr/lib/x86_64-linux-gnu -lnvJitLink &&
cd "$wslCudaRoot"
$cudaEnv
export LIBNVJITLINK_PATH=/tmp/libnvJitLink_shim.so
cargo run -p bulletou-cuda-train --features cuda,root-loader --release -- --nnue-teacher-train --teacher "$wslTeacher"$directWeightsArg$directResumeTrainStateArg$directOutputArg --train-steps $directTrainSteps$directSaveRateArg --batch-size $BatchSize --buffer-mb $BufferMb --loader-threads $LoaderThreads --threads $Threads --score-drop-abs $ScoreDropAbs --loss-kind $LossKind$debugFlag$directTrainedForwardArg$directTrainStateArg
"@

    Invoke-Checked "NNUE direct teacher train loop" {
        $shim | wsl -d $WslDistro -- bash -lc $directTeacherCommand
    }
}

if ($RunDirectTeacherTrain) {
    Write-Host "OK: cuda-oxide NNUE teacher smoke completed"
} elseif ([string]::IsNullOrWhiteSpace($ResumeTrainStateFixture)) {
    Write-Host "OK: cuda-oxide NNUE teacher loss smoke completed"
} else {
    Write-Host "OK: cuda-oxide NNUE train-state resume smoke completed"
}
Write-Host "fixtures:"
foreach ($path in $fixturePaths) {
    Write-Host "  $path"
}
if (-not [string]::IsNullOrWhiteSpace($TrainedForwardFixture)) {
    Write-Host "trained forward fixture:"
    Write-Host "  $TrainedForwardFixture"
}
if (-not [string]::IsNullOrWhiteSpace($TrainStateFixture)) {
    Write-Host "train state fixture:"
    Write-Host "  $TrainStateFixture"
}
if (-not [string]::IsNullOrWhiteSpace($DirectTrainedForwardFixture)) {
    Write-Host "direct trained forward fixture:"
    Write-Host "  $DirectTrainedForwardFixture"
}
if (-not [string]::IsNullOrWhiteSpace($DirectTrainStateFixture)) {
    Write-Host "direct train state fixture:"
    Write-Host "  $DirectTrainStateFixture"
}
if (-not [string]::IsNullOrWhiteSpace($Output)) {
    Write-Host "direct bridge checkpoint output:"
    Write-Host "  $Output"
}
