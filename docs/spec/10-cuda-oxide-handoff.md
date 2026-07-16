# 10. cuda-oxide 高速化 引き継ぎメモ

この文書は、CUDA Toolkit / NVIDIA GPU がある環境で BulletOu の
cuda-oxide 高速化作業を再開するための引き継ぎ資料である。

## 目的

BulletOu の既存 Bullet backend は残したまま、NNUE / SFNN 専用の
cuda-oxide backend を追加し、tatara 相当の学習速度を目指す。

重要な方針:

- 既存 `--backend bullet` を壊さない。
- cuda-oxide 関連の依存は nested workspace `cuda-oxide/` に閉じる。
- まず fp32 forward の CPU / GPU 数値一致を確認する。
- 数値一致前に train loop へ接続しない。

## 現在位置

作業 branch:

```bash
git checkout shogi-support
```

直近の関連 commit:

```text
b3a1ff1 Document NNUE cuda forward verification
7f4f652 Add NNUE cuda forward launcher
9b40b76 Add cuda oxide NNUE forward kernels
f47404a Use cuda oxide kernel names as base labels
b6b36e6 Add NNUE fast forward trace
035cf62 Document NNUE cuda forward subtasks
```

完了済み:

- root 側に `FastBatchHost` と CPU scalar NNUE forward golden を追加済み。
- nested `cuda-oxide/` workspace を作成済み。
- `bulletou-cuda-oxide-runtime` に NNUE weight / workspace / batch layout を追加済み。
- `bulletou-cuda-train` binary crate に cuda-oxide `#[kernel]` 定義を追加済み。
- `launch_nnue_forward` を追加済み。

未完了:

- CUDA feature の compile 検証。
- cargo-oxide で生成される kernel artifact の loader 接続。
- `--nnue-forward-smoke` の実装。
- CPU golden と GPU output の数値一致確認。
- loss / backward / optimizer kernel。
- train loop への接続。

## 重要ファイル

| path | 内容 |
|---|---|
| `docs/spec/08-cuda-oxide-speedup-plan.md` | 調査結果と全体方針 |
| `docs/spec/09-cuda-oxide-todo.md` | 作業チケット |
| `cuda-oxide/README.md` | nested workspace の概要 |
| `cuda-oxide/Cargo.toml` | cuda-oxide dependency pin |
| `cuda-oxide/rust-toolchain.toml` | nightly / toolchain pin |
| `cuda-oxide/crates/runtime/src/nnue.rs` | NNUE host/device layout |
| `cuda-oxide/bins/bulletou-cuda-train/src/kernels/nnue.rs` | NNUE CUDA kernel 定義 |
| `cuda-oxide/bins/bulletou-cuda-train/src/nnue_forward.rs` | host launch sequence |
| `crates/bulletou_lib/src/value/fast_nnue.rs` | CPU scalar golden |
| `crates/bulletou_lib/src/value/fast_batch.rs` | fixed batch layout |

## CUDA 環境で最初に確認すること

CUDA Toolkit が見える環境で実行する。

例:

```bash
cd /path/to/BulletOu/cuda-oxide

export CUDA_HOME=/usr/local/cuda
export CUDA_TOOLKIT_PATH="$CUDA_HOME"

# RTX 4090 なら sm_89。Ampere なら sm_80 / sm_86、Turing なら sm_75。
export CUDA_OXIDE_TARGET=sm_89

cargo check
cargo test -p bulletou-cuda-oxide-runtime
cargo check -p bulletou-cuda-train --features cuda
```

現在の非 CUDA 環境では、最後のコマンドは以下で止まることを確認済み:

```text
cuda-bindings: could not find cuda.h in the CUDA toolkit at `/usr/local/cuda`.
```

このエラーは CUDA Toolkit 不在が原因であり、現時点では BulletOu 側の
compile error まで到達していない。

## cargo-oxide の準備

cuda-oxide dependency は rev `b5d35e0` に pin している。
`cargo-oxide` も同じ rev に揃える。

```bash
cargo install --git https://github.com/NVlabs/cuda-oxide.git \
  --rev b5d35e0 \
  --force \
  cargo-oxide
```

必要なら診断:

```bash
cargo-oxide doctor
```

kernel artifact 生成の初期候補:

```bash
cd /path/to/BulletOu/cuda-oxide/bins/bulletou-cuda-train
CUDA_OXIDE_TARGET=sm_89 cargo-oxide build --emit-nvvm-ir --arch sm_89
```

tatara では `.ll` を生成し、起動時に `.ll -> .ptx` へ変換して
`CudaModule` として load する。BulletOu 側はまだこの artifact loader を
移植していない。

## 次に実装する順番

### 1. CUDA feature compile を通す

まず以下を通す。

```bash
cd cuda-oxide
cargo check -p bulletou-cuda-train --features cuda
```

想定される修正対象:

- `cuda_launch!` の kernel path / module path
- `DisjointSlice` の helper 関数渡し
- `slice(...)` / `slice_mut(...)` 引数の借用形
- `cuda-device` 側で使えない標準ライブラリ API

この段階では実行できなくてもよい。compile が通れば次に進む。

### 2. kernel artifact loader を追加する

現状の `bulletou-cuda-train` は smoke 用の `smoke/noop.ptx` を直接 load する。
NNUE kernel では、cargo-oxide が生成した `bulletou-cuda-train.ll` または
`.ptx` を load する必要がある。

tatara の参考箇所:

```text
tatara/crates/gpu-runtime/src/kernel_loader.rs
tatara/bins/nnue_train/src/kernel_module.rs
```

移植方針:

- `cuda-oxide/crates/runtime/src/kernel_loader.rs` を追加する。
- `manifest_dir` を呼び出し側 binary から渡せる形にする。
- 探索対象は、bin crate dir と workspace root の両方。
- `.ll` があれば `.ptx` へ変換して load する。
- まずは tatara の実装を保守的に移植し、後で整理する。

### 3. `--nnue-forward-smoke` を実装する

CLI 例:

```bash
cd cuda-oxide
CUDA_HOME=/usr/local/cuda CUDA_OXIDE_TARGET=sm_89 \
  cargo run -p bulletou-cuda-train --features cuda -- \
  --nnue-forward-smoke --device 0
```

smoke の内容:

- tiny shape を使う。
- 固定 weight / 固定 sparse batch を作る。
- CPU scalar forward を計算する。
- `NnueForwardDeviceWeights` / `NnueForwardDeviceBatch` / `NnueForwardWorkspace`
  を作る。
- `launch_nnue_forward` を呼ぶ。
- `workspace.output.to_host_vec()` で GPU output を戻す。
- 絶対誤差 `1e-5` 以下なら pass。

注意:

- root crate の CPU golden は nested `cuda-oxide/` から直接参照していない。
  smoke 用には tiny CPU reference を nested 側に書いてよい。
- 最初は output だけ比較する。
- 不一致なら L0 / concat / L1 / L2 の中間 buffer を戻す debug mode を追加する。

### 4. 1 batch NNUE forward を固定 arch で通す

tiny shape が通ったら、`NNUE_HALFKP_256x2_32_32` の固定 shape で通す。

合格条件:

- 同じ `FastBatchHost` 相当の sparse index。
- 同じ owned weights。
- CPU scalar golden と GPU output が `1e-5` 以内。

### 5. 速度比較はまだしない

この時点の kernel は correctness-first の単純実装であり、tatara のような
fused / tiled / FP16 / async ring 実装ではない。

positions/sec 比較は以下の後で行う:

- forward 数値一致
- loss kernel
- backward kernel
- optimizer kernel
- input upload / loss readback ring

## 既知の設計上の注意

### `#[kernel]` は binary crate 側に置く

cuda-oxide は binary crate から到達可能な `#[kernel]` を artifact 化する。
そのため、kernel 定義はここに置いている:

```text
cuda-oxide/bins/bulletou-cuda-train/src/kernels/
```

runtime crate に kernel entry point を移すと、cargo-oxide の収集対象から外れる
可能性がある。

### raw kernel name resolve は使わない

`#[kernel]` の PTX symbol は Rust 名そのままではない。
`module.load_function("nnue_sparse_l0_crelu")` のような raw resolve ではなく、
`cuda_launch!` に kernel path を渡して、生成された PTX name helper を使う。

### forward launcher はまだ train loop から未使用

`launch_nnue_forward` は `#[allow(dead_code)]` が付いている。
数値一致前に trainer へ接続してはいけない。

### 現時点では速くなっていない

現在の実装は高速化そのものではなく、高速化するための土台である。
元の BulletOu より速くなるのは、loss / backward / optimizer / async ring まで
入ってからである。

## トラブルシュート

### `cuda.h` が見つからない

CUDA Toolkit が見えていない。

```bash
export CUDA_HOME=/usr/local/cuda
export CUDA_TOOLKIT_PATH="$CUDA_HOME"
ls "$CUDA_HOME/include/cuda.h"
```

### `cargo-oxide` と crate の rev がずれる

`cargo-oxide` を rev `b5d35e0` で入れ直す。

```bash
cargo install --git https://github.com/NVlabs/cuda-oxide.git \
  --rev b5d35e0 \
  --force \
  cargo-oxide
```

### kernel artifact が見つからない

まず bin crate 側で artifact を生成する。

```bash
cd cuda-oxide/bins/bulletou-cuda-train
CUDA_OXIDE_TARGET=sm_89 cargo-oxide build --emit-nvvm-ir --arch sm_89
```

その後、loader が探す path と実際に生成された `.ll` / `.ptx` の場所を確認する。

### CUDA 実機で compile が通ったら

次の作業として、この文書の「次に実装する順番」の 2 から進める。
特に `--nnue-forward-smoke` の実装と、CPU/GPU output 一致が最優先である。
