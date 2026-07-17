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
    [string]$Fixture,
    [string]$WslDistro = "Ubuntu-24.04",
    [string]$CudaArch = "sm_89",
    [ValidateSet("sigmoid-mse", "wrm")]
    [string]$LossKind = "sigmoid-mse",
    [int]$TrainSteps = 1,
    [switch]$SkipCudaBuild,
    [switch]$DebugReadback
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

if (-not (Test-Path -LiteralPath $Teacher)) {
    throw "Teacher file not found: $Teacher"
}

if ([string]::IsNullOrWhiteSpace($Fixture)) {
    $Fixture = Join-Path $repoRoot "target\cuda-oxide-fixtures\nnue-halfkp-teacher-train-b$BatchSize.bin"
}

$fixturePaths = @()
if ($TrainSteps -eq 1) {
    $fixturePaths += $Fixture
} else {
    $fixtureDir = Split-Path -Parent $Fixture
    $fixtureStem = [System.IO.Path]::GetFileNameWithoutExtension($Fixture)
    $fixtureExt = [System.IO.Path]::GetExtension($Fixture)
    for ($step = 0; $step -lt $TrainSteps; $step++) {
        $fixturePaths += (Join-Path $fixtureDir "$fixtureStem-step$step$fixtureExt")
    }
}

foreach ($path in $fixturePaths) {
    $fixtureDir = Split-Path -Parent $path
    New-Item -ItemType Directory -Force -Path $fixtureDir | Out-Null
}

for ($step = 0; $step -lt $TrainSteps; $step++) {
    $outFixture = $fixturePaths[$step]
    Invoke-Checked "export NNUE HalfKP teacher train fixture batch $step" {
        Set-Location $repoRoot
        cargo run -p bulletou_lib --example export_nnue_forward_fixture --release -- `
            --out $outFixture `
            --train-fixture `
            --case halfkp `
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
$wslFixtures = @($fixturePaths | ForEach-Object { Convert-ToWslPath $_ })
$debugFlag = if ($DebugReadback) { " --debug-readback" } else { "" }
$fixtureArgs = ($wslFixtures | ForEach-Object { "--nnue-train-fixture `"$_`"" }) -join " "

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

Invoke-Checked "NNUE loss Ranger step smoke with real teacher fixture" {
    $shim | wsl -d $WslDistro -- bash -lc $runCommand
}

Write-Host "OK: cuda-oxide NNUE teacher loss smoke completed"
Write-Host "fixtures:"
foreach ($path in $fixturePaths) {
    Write-Host "  $path"
}
