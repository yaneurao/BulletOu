param(
    [string]$Teacher = "C:\shogi\teacher\yane-distill-hcpe-20260508shuffled\shuffled-001.hcpe",
    [string]$TestTeacher = "C:\shogi\teacher\test\yamaoka-floodgate.hcpe",
    [string]$TataraRoot = "C:\shogi\YaneuraOuWorks\tatara",
    [string]$WorkDir = "target\tatara-parity",
    [string]$WslDistro = "Ubuntu-24.04",
    [string]$CudaArch = "sm_89",
    [int]$TrainPositions = 128,
    [int]$TestPositions = 128,
    [int]$BatchSize = 64,
    [int]$BatchesPerSuperbatch = 1,
    [int]$Superbatches = 1,
    [int]$Threads = 8,
    [switch]$BuildBulletKernel,
    [switch]$BuildTataraKernel,
    [switch]$SkipBulletMetrics,
    [switch]$DebugHost
)

$ErrorActionPreference = "Stop"
$cargoRunProfileArgs = @()
$cargoRunProfile = ""
if (-not $DebugHost) {
    $cargoRunProfileArgs = @("--release")
    $cargoRunProfile = "--release "
}

function Resolve-FullPath([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Path))
}

function Convert-ToWslPath([string]$Path) {
    $full = [System.IO.Path]::GetFullPath($Path)
    if ($full -match '^([A-Za-z]):\\(.*)$') {
        $drive = $matches[1].ToLowerInvariant()
        $rest = $matches[2] -replace '\\', '/'
        return "/mnt/$drive/$rest"
    }
    return ($full -replace '\\', '/')
}

function Quote-Bash([string]$Text) {
    return "'" + ($Text -replace "'", "'\''") + "'"
}

function Quote-WindowsArgument([string]$Argument) {
    if ($Argument.Length -gt 0 -and $Argument -notmatch '[\s"]') {
        return $Argument
    }

    $result = '"'
    $backslashes = 0
    foreach ($ch in $Argument.ToCharArray()) {
        if ($ch -eq '\') {
            $backslashes += 1
            continue
        }
        if ($ch -eq '"') {
            $result += ('\' * ($backslashes * 2 + 1))
            $result += '"'
            $backslashes = 0
            continue
        }
        if ($backslashes -gt 0) {
            $result += ('\' * $backslashes)
            $backslashes = 0
        }
        $result += $ch
    }
    if ($backslashes -gt 0) {
        $result += ('\' * ($backslashes * 2))
    }
    $result += '"'
    return $result
}

function Join-ProcessArguments([string[]]$Arguments) {
    return (($Arguments | ForEach-Object { Quote-WindowsArgument $_ }) -join " ")
}

function Invoke-LoggedProcess([string]$Label, [string]$FileName, [string[]]$Arguments, [string]$LogPath) {
    Write-Host "== $Label =="
    $parent = Split-Path -Parent $LogPath
    if ($parent) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo.FileName = $FileName
    $process.StartInfo.Arguments = Join-ProcessArguments $Arguments
    $process.StartInfo.UseShellExecute = $false
    $process.StartInfo.RedirectStandardOutput = $true
    $process.StartInfo.RedirectStandardError = $true
    $process.StartInfo.CreateNoWindow = $true

    [void]$process.Start()
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $process.WaitForExit()

    $stdout = $stdoutTask.Result
    $stderr = $stderrTask.Result
    $combined = $stdout + $stderr
    $utf8NoBom = New-Object System.Text.UTF8Encoding -ArgumentList $false
    [System.IO.File]::WriteAllText($LogPath, $combined, $utf8NoBom)
    if ($stdout.Length -gt 0) {
        Write-Host $stdout -NoNewline
    }
    if ($stderr.Length -gt 0) {
        Write-Host $stderr -NoNewline
    }

    if ($process.ExitCode -ne 0) {
        throw "$Label failed with exit code $($process.ExitCode)"
    }
}

function Invoke-LoggedLocal([string]$Label, [string[]]$Command, [string]$LogPath) {
    $arguments = @()
    if ($Command.Length -gt 1) {
        $arguments = @($Command[1..($Command.Length - 1)])
    }
    Invoke-LoggedProcess -Label $Label -FileName $Command[0] -Arguments $arguments -LogPath $LogPath
}

function Invoke-LoggedWsl([string]$Label, [string]$BashCommand, [string]$LogPath) {
    Invoke-LoggedProcess -Label $Label -FileName "wsl" -Arguments @("-d", $WslDistro, "--", "bash", "-lc", $BashCommand) -LogPath $LogPath
}

$repoRoot = Resolve-FullPath "."
$tataraRootFull = [System.IO.Path]::GetFullPath($TataraRoot)
$workDirFull = Resolve-FullPath $WorkDir
$runId = Get-Date -Format "yyyyMMdd-HHmmss"
$runDir = Join-Path $workDirFull "parity-$runId"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$teacherPsv = Join-Path $runDir "teacher-$TrainPositions.psv"
$testPsv = Join-Path $runDir "test-$TestPositions.psv"
$tataraOut = Join-Path $runDir "tatara"
$bulletSpeedOut = Join-Path $runDir "bulletou-speed"
$bulletMetricsOut = Join-Path $runDir "bulletou-metrics"
$totalTrainBatches = $Superbatches * $BatchesPerSuperbatch
$requiredTrainPositions = [int64]$totalTrainBatches * [int64]$BatchSize
if ([int64]$TrainPositions -lt $requiredTrainPositions) {
    throw "TrainPositions=$TrainPositions is smaller than Superbatches*BatchesPerSuperbatch*BatchSize=$requiredTrainPositions."
}

Invoke-LoggedLocal "export train PSV" (@(
    "cargo", "run"
) + $cargoRunProfileArgs + @(
    "-p", "bulletou_lib", "--example", "export_teacher_psv", "--",
    "--teacher", $Teacher,
    "--out", $teacherPsv,
    "--positions", "$TrainPositions",
    "--buffer-mb", "1",
    "--loader-threads", "1"
)) (Join-Path $runDir "export-train.log")

Invoke-LoggedLocal "export test PSV" (@(
    "cargo", "run"
) + $cargoRunProfileArgs + @(
    "-p", "bulletou_lib", "--example", "export_teacher_psv", "--",
    "--teacher", $TestTeacher,
    "--out", $testPsv,
    "--positions", "$TestPositions",
    "--buffer-mb", "1",
    "--loader-threads", "1"
)) (Join-Path $runDir "export-test.log")

$tataraPtx = Join-Path $tataraRootFull "nnue_train.ptx"
$commonEnv = "export PATH=/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin " +
    "CUDA_HOME=/usr CUDA_PATH=/usr CUDA_TOOLKIT_PATH=/usr " +
    "CUDA_OXIDE_LIBDEVICE=/usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc " +
    "LIBCLANG_PATH=/usr/lib/llvm-20/lib " +
    "LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu " +
    "CUDA_OXIDE_TARGET=$CudaArch"

if (-not (Test-Path $tataraPtx)) {
    if (-not $BuildTataraKernel) {
        throw "Tatara kernel artifact not found: $tataraPtx. Re-run with -BuildTataraKernel."
    }
    $tataraWsl = Convert-ToWslPath $tataraRootFull
    $buildCmd = @(
        "cd $(Quote-Bash "$tataraWsl/bins/nnue_train")",
        $commonEnv,
        "cargo-oxide build --emit-nvvm-ir --arch sm_100",
        "cd $(Quote-Bash $tataraWsl)",
        "/usr/bin/llvm-link-20 nnue_train.ll /usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc -o nnue_train.linked.bc",
        "/usr/bin/llc-20 --mtriple=nvptx64-nvidia-cuda --mcpu=$CudaArch -O2 nnue_train.linked.bc -o nnue_train.ptx"
    ) -join " && "
    Invoke-LoggedWsl "build tatara kernel" $buildCmd (Join-Path $runDir "tatara-kernel-build.log")
}

$teacherPsvWsl = Convert-ToWslPath $teacherPsv
$testPsvWsl = Convert-ToWslPath $testPsv
$tataraRootWsl = Convert-ToWslPath $tataraRootFull
$tataraOutWsl = Convert-ToWslPath $tataraOut
$repoRootWsl = Convert-ToWslPath $repoRoot
$bulletCubin = Join-Path $repoRoot "cuda-oxide\target\cuda-oxide-artifacts\bulletou_cuda_train_bo015.cubin"
$bulletCubinWsl = Convert-ToWslPath $bulletCubin
$bulletSpeedOutWsl = Convert-ToWslPath $bulletSpeedOut
$bulletMetricsOutWsl = Convert-ToWslPath $bulletMetricsOut

if (-not (Test-Path $bulletCubin)) {
    if (-not $BuildBulletKernel) {
        throw "BulletOu CUDA artifact not found: $bulletCubin. Re-run with -BuildBulletKernel."
    }
}
if ($BuildBulletKernel) {
    # cuda-oxide atomic RMW currently needs the sm_100 NVVM IR route; direct
    # legacy NVVM IR for sm_89 cannot lower atomics.
    $bulletBuildEnv = "export PATH=/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin " +
        "CUDA_HOME=/usr CUDA_PATH=/usr CUDA_TOOLKIT_PATH=/usr " +
        "CUDA_OXIDE_LIBDEVICE=/usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc " +
        "LIBCLANG_PATH=/usr/lib/llvm-20/lib " +
        "LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu " +
        "CUDA_OXIDE_TARGET=sm_100"
    $bulletBuildCmd = @(
        "cd $(Quote-Bash "$repoRootWsl/cuda-oxide")",
        $bulletBuildEnv,
        "cargo-oxide build --emit-nvvm-ir --arch sm_100 --features cuda -- --package bulletou-cuda-train --release",
        "mkdir -p target/cuda-oxide-artifacts",
        "/usr/bin/llvm-link-20 bulletou_cuda_train.ll /usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc -o target/cuda-oxide-artifacts/bulletou_cuda_train_bo015_linked.bc",
        "/usr/bin/opt-20 -O2 target/cuda-oxide-artifacts/bulletou_cuda_train_bo015_linked.bc -o target/cuda-oxide-artifacts/bulletou_cuda_train_bo015_opt.bc",
        "/usr/bin/llc-20 --mtriple=nvptx64-nvidia-cuda --mcpu=$CudaArch -O2 target/cuda-oxide-artifacts/bulletou_cuda_train_bo015_opt.bc -o target/cuda-oxide-artifacts/bulletou_cuda_train_bo015.ptx",
        "ptxas -arch=$CudaArch target/cuda-oxide-artifacts/bulletou_cuda_train_bo015.ptx -o target/cuda-oxide-artifacts/bulletou_cuda_train_bo015.cubin"
    ) -join " && "
    Invoke-LoggedWsl "build BulletOu cuda-oxide kernel" $bulletBuildCmd (Join-Path $runDir "bulletou-kernel-build.log")
}

# This profile matches BulletOu's current `--loss-kind wrm` constants:
# nnue2score=600, prediction offset/scaling=270/340, target offset/scaling=270/380,
# and nnue-pytorch exponent 2.5.
$tataraArgs = @(
    "cd $(Quote-Bash $tataraRootWsl)",
    $commonEnv,
    "cargo run ${cargoRunProfile}--bin nnue-train --",
    "--data $(Quote-Bash $teacherPsvWsl)",
    "--test-data $(Quote-Bash $testPsvWsl)",
    "--test-positions $TestPositions",
    "--output $(Quote-Bash $tataraOutWsl)",
    "--net-id tatara-parity",
    "--feature-set halfkp",
    "--superbatches $Superbatches",
    "--batches-per-superbatch $BatchesPerSuperbatch",
    "--batch-size $BatchSize",
    "--threads $Threads",
    "--save-rate $Superbatches",
    "--scale 600",
    "--win-rate-model",
    "--wrm-in-offset 270",
    "--wrm-target-offset 270",
    "--wrm-in-scaling 340",
    "--wrm-target-scaling 380",
    "--wrm-nnue2score 600",
    "--loss-pow-exp 2.5",
    "--lr-schedule constant",
    "--lr 0.01",
    "simple --arch 256x2-32-32"
)
$tataraCmd = (@($tataraArgs[0], $tataraArgs[1], (($tataraArgs[2..($tataraArgs.Length - 1)]) -join " "))) -join " && "
Invoke-LoggedWsl "tatara parity smoke" $tataraCmd (Join-Path $runDir "tatara.log")

$bulletBasePrefix = @(
    "cd $(Quote-Bash "$repoRootWsl/cuda-oxide")",
    $commonEnv
)
$bulletBaseArgs = @(
    "cargo run ${cargoRunProfile}-p bulletou-cuda-train --features cuda,root-loader --",
    "--nnue-teacher-train",
    "--teacher $(Quote-Bash $teacherPsvWsl)",
    "--test-teacher $(Quote-Bash $testPsvWsl)",
    "--test-positions $TestPositions",
    "--test-batch-size $BatchSize",
    "--train-steps $totalTrainBatches",
    "--batches-per-superbatch $BatchesPerSuperbatch",
    "--batch-size $BatchSize",
    "--buffer-mb 1",
    "--loader-threads 1",
    "--threads $Threads",
    "--score-drop-abs 0",
    "--loss-kind wrm",
    "--learning-rate 0.01",
    "--lr-schedule fixed",
    "--optimizer-weight-decay 0",
    "--ptx $(Quote-Bash $bulletCubinWsl)"
)

$bulletSpeedCmd = ($bulletBasePrefix + (($bulletBaseArgs) -join " ")) -join " && "
Invoke-LoggedWsl "BulletOu cuda-oxide speed smoke" $bulletSpeedCmd (Join-Path $runDir "bulletou-speed.log")

if (-not $SkipBulletMetrics) {
    $bulletMetricsCmd = ($bulletBasePrefix + (($bulletBaseArgs + @(
        "--save-rate $Superbatches",
        "--output $(Quote-Bash $bulletMetricsOutWsl)"
    )) -join " ")) -join " && "
    Invoke-LoggedWsl "BulletOu cuda-oxide metrics smoke" $bulletMetricsCmd (Join-Path $runDir "bulletou-metrics.log")
}

Write-Host ""
Write-Host "Parity smoke artifacts:"
Write-Host "  run_dir        $runDir"
Write-Host "  host_profile   $(if ($DebugHost) { 'debug' } else { 'release' })"
Write-Host "  threads        $Threads"
Write-Host "  train_psv      $teacherPsv"
Write-Host "  test_psv       $testPsv"
Write-Host "  tatara_log     $(Join-Path $runDir "tatara.log")"
Write-Host "  bullet_speed   $(Join-Path $runDir "bulletou-speed.log")"
if (-not $SkipBulletMetrics) {
    Write-Host "  bullet_metrics $(Join-Path $runDir "bulletou-metrics.log")"
}
