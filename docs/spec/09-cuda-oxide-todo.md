# 09. cuda-oxide 高速化 TODO

tatara 同等速度を目標にした実装チケット。

この TODO は、既存 Bullet backend を壊さずに NNUE / SFNN 専用 cuda-oxide backend を
段階的に追加するための作業順である。各項目は小さい commit 単位に分割し、完了時に
status を更新する。

## TODO 一覧

| id | status | 内容 | 完了条件 |
|---|---|---|---|
| CO-001 | done | TODO 起票 | このファイルと README へのリンクを追加する |
| CO-002 | done | fixed-layout batch adapter | 既存 dataloader から `FastBatchHost` を直接列挙できる |
| CO-003 | done | cuda-oxide crate 境界の作成 | 既存 workspace を巻き込まず、専用 crate / binary の置き場所を作る |
| CO-004 | done | PTX smoke loader | 生成済み PTX を load し、kernel symbol resolve と最小 kernel launch を行う |
| CO-005 | done | CPU reference test harness | fast backend kernel と既存 Bullet backend の 1 batch 出力比較を作る |
| CO-006 | done | minimal NNUE forward | `NNUE_HALFKP_256x2_32_32` の 1 batch forward を cuda-oxide で一致させる。CPU golden と所有重みレイアウトは追加済み |
| CO-007 | done | SFNN forward | `SFNN_halfka2_1024_7_64_k3k3` の forward を cuda-oxide で一致させる |
| CO-008 | todo | loss kernel | target transform / sigmoid / loss reduction を fused kernel 化する |
| CO-009 | todo | backward kernel | dense backward と sparse FT backward を実装する |
| CO-010 | todo | optimizer kernel | Ranger / RAdam update を fused kernel 化する |
| CO-011 | todo | async rings | input upload ring と loss readback ring を入れる |
| CO-012 | todo | checkpoint compatibility | `nn.bin` / log / checkpoint layout を既存と揃え、state backend marker を入れる |
| CO-013 | todo | speed benchmark | 同一 teacher / seed / schedule で existing Bullet backend と positions/sec を比較する |

## 作業原則

- 既存 `--backend bullet` は常に動く状態を保つ。
- cuda-oxide dependency は既存 workspace root に直接入れない。
- `cuda-oxide/` nested workspace の default build は CUDA Toolkit なしで通る状態を保つ。
- 数値が変わる高速化は opt-in にする。
- 速度比較は fp32 baseline の 1 batch 数値一致後に行う。
- KPPT / KPP_KKPT は今回の cuda-oxide 高速化対象外とする。

## CO-006 minimal NNUE forward 内訳

- done: `FastBatchHost` sparse padding を `-1` sentinel に統一。
- done: `FastBatchHost` から 1 sample の `stm` / `nstm` sparse slice を取り出す API を追加。
- done: `NNUE_HALFKP_256x2_32_32` の CPU scalar golden forward を追加。
- done: root 側に owned weight layout と workspace layout を追加。
- done: nested `cuda-oxide` runtime 側に weight / workspace / launch plan layout を追加。
- done: nested `cuda-oxide` runtime 側に forward kernel set resolve 境界を追加。
- done: `nnue_sparse_l0_crelu` kernel 定義を追加。WSL2 Ubuntu 24.04 + RTX 4090 で feature `cuda` の compile と tiny fixed case の L0 出力比較を確認。
- done: `nnue_concat_l0` / `nnue_dense_l1_crelu` / `nnue_dense_l2_crelu` / `nnue_dense_output` の kernel 定義を追加。WSL2 Ubuntu 24.04 + RTX 4090 で compile / launch を確認。
- done: host launch sequence を追加。WSL2 Ubuntu 24.04 + RTX 4090 で tiny fixed case の CPU golden 比較を確認。
- done: `bulletou-cuda-train --nnue-forward-smoke` CLI と tiny fixed weight / sparse batch の CPU golden 比較を追加。
- done: `--nnue-forward-case halfkp` を追加し、`NNUE_HALFKP_256x2_32_32` の実 shape / max_active=38 / 決定論的 synthetic weight + sparse batch で同じ launch sequence を CPU golden と比較。
- done: `--write-nnue-forward-fixture` / `--nnue-forward-fixture` を追加。root workspace へ依存せず、root 側 exporter から `FastBatchHost` + `NnueForwardOwnedWeights` を渡すための little-endian fixture 境界を用意。
- done: root 側 `write_nnue_forward_fixture` / `write_nnue_forward_fixture_file` を追加。`FastBatchHost` + `NnueForwardWeights` から同じ fixture 形式を書き出せる。
- done: root example `export_nnue_forward_fixture --teacher` を追加。既存 `expand_teacher` / `infer_data_format` / `HcpeDataLoader` / `Hcpe3DataLoader` / `ShogiPackLoader` / `DirectSequentialDataLoader` と `DefaultDataLoader::prepare(ShogiHalfKP)` を通して、実 teacher の先頭 1 batch を `FastBatchHost` fixture にできる。
- done: root 側 `FastBatchHost` / `NnueForwardOwnedWeights` から nested cuda-oxide 実行へ流す橋渡しを作り、synthetic ではなく既存データ経路の 1 batch で CPU golden と比較する。
- note: 現 Windows native 環境では RTX 4090 と CUDA Toolkit v13.1 は見える。`cargo-oxide` と Python wheel 由来の `libclang.dll` は導入済み。CUDA 13.1 headers と Python wheel の CUDA 12.9 headers の双方で `cargo check -p bulletou-cuda-train --features cuda` を試したが、pin済み cuda-oxide rev の `cuda-core` が Windows bindgen 生成の `u32` flag / enum と合わず E0308 で停止する。実機 launch 検証は WSL2 Ubuntu 24.04 で進める。

### 2026-07-16 WSL2 / RTX 4090 検証結果

- WSL2 Ubuntu 24.04 を導入し、`.wslconfig` に `networkingMode=mirrored` / `dnsTunneling=true` / `autoProxy=true` を追加して apt の外向き通信を復旧。
- WSL2 側で `nvidia-smi` が RTX 4090 を認識することを確認。
- 導入済み: `rustup`, `nightly-2026-04-03`, `cargo-oxide` rev `b5d35e0`, LLVM/Clang 20, Ubuntu `nvidia-cuda-toolkit` 12.0。
- `cargo check` と `cargo check -p bulletou-cuda-train --features cuda` は WSL2 で成功。
- `cargo oxide setup` と `cargo oxide build --arch sm_89 --features cuda -- --package bulletou-cuda-train` は成功。
- `cargo oxide doctor` は Ubuntu CUDA 12.0 の `libnvJitLink.so` が `nvJitLinkCreate` 未バージョン名シンボルを出さないため赤を残す。ただし現 CO-006 tiny forward kernel は libdevice math を使わず、PTX 生成と実行は成功。
- `cargo run -p bulletou-cuda-train --features cuda -- --nnue-forward-smoke --ptx ./bulletou_cuda_train.ptx --debug-readback` は成功。
  - output max_abs diff: `0.00000011920929`
  - `stm_l0` / `nstm_l0` / `combined` / `hidden1` / `hidden2`: max_abs diff `0`

### 2026-07-17 WSL2 / RTX 4090 検証結果

- `--nnue-forward-case halfkp` を追加し、`NNUE_HALFKP_256x2_32_32` の実 shape (`input=125388 l1=256 l2=32 l3=32`) と `max_active=38` で forward smoke を実行。
- `cargo check`, `cargo test`, `cargo check -p bulletou-cuda-train --features cuda`, `cargo oxide build --arch sm_89 --features cuda -- --package bulletou-cuda-train` は成功。
- `cargo run -p bulletou-cuda-train --features cuda -- --nnue-forward-smoke --nnue-forward-case tiny --ptx ./bulletou_cuda_train.ptx --debug-readback` は成功。
- `cargo run -p bulletou-cuda-train --features cuda -- --nnue-forward-smoke --nnue-forward-case halfkp --ptx ./bulletou_cuda_train.ptx --debug-readback` は成功。
  - output max_abs diff: `0.0000000009313226`
  - `stm_l0` / `nstm_l0` / `combined`: max_abs diff `0`
  - `hidden1` max_abs diff: `0.0000000037252903`
  - `hidden2` max_abs diff: `0.0000000018626451`
- tiny fixture write/read roundtrip は成功。fixture size: `204` bytes。
- halfkp fixture write/read roundtrip は成功。fixture size: 約 `123M`。
- root 側 exporter unit test は成功。tiny fixture の magic / header / byte length (`204`) を確認。
- root example `export_nnue_forward_fixture` を追加。root 側の `FastBatchHost` + `NnueForwardOwnedWeights` から `tiny` / `halfkp` fixture を出力でき、HalfKP weights は `optimiser_state/weights.bin` と bundled `state.bin` (`nnue/weights/*`) の両方から読める。
- `cargo check -p bulletou_lib --example export_nnue_forward_fixture` は成功。
- root で出力した tiny fixture (`target/bulletou-root-tiny.nnuef`) を WSL2/cuda-oxide の `--nnue-forward-fixture` で実行し成功。output max_abs diff: `0.00000011920929`、debug readback buffers は許容誤差内。
- root で出力した HalfKP fixture (`target/bulletou-root-halfkp.nnuef`) を WSL2/cuda-oxide の `--nnue-forward-fixture` で実行し成功。output max_abs diff: `0.0000000009313226`、`stm_l0` / `nstm_l0` / `combined` max_abs diff: `0`、`hidden1` max_abs diff: `0.0000000037252903`、`hidden2` max_abs diff: `0.0000000018626451`。
- `export_nnue_forward_fixture --teacher` の loader 配線は `cargo check -p bulletou_lib --example export_nnue_forward_fixture` で型検証済み。
- user-provided HCPE teacher `C:\shogi\teacher\yane-distill-hcpe-20260508shuffled\shuffled-001.hcpe` から batch_size=2 の HalfKP fixture (`target/bulletou-teacher-halfkp.nnuef`) を export し、WSL2/cuda-oxide の `--nnue-forward-fixture` で実行し成功。output max_abs diff: `0.0000000027939677`、`stm_l0` / `nstm_l0` / `combined` max_abs diff: `0`、`hidden1` max_abs diff: `0.000000007450581`、`hidden2` max_abs diff: `0.0000000018626451`。
- actual BulletOu checkpoint weight `checkpoints/NNUE_HALFKP-256x2-32-32-6sb-cos/0031/state.bin` と user-provided HCPE teacher `C:\shogi\teacher\yane-distill-hcpe-20260508shuffled\shuffled-001.hcpe` から batch_size=2 の HalfKP fixture (`target/bulletou-teacher-realweights-halfkp.nnuef`) を export し、WSL2/cuda-oxide の `--nnue-forward-fixture` で実行し成功。CPU output は `[-1.3658719, 4.570574]`、GPU comparison は output max_abs diff: `0.00000011920929`、`stm_l0` / `nstm_l0` / `combined` max_abs diff: `0`、`hidden1` max_abs diff: `0.00000011920929`、`hidden2` max_abs diff: `0.00000011920929`。

### CO-006 CUDA 実機検証

このリポジトリの通常 CI / 通常開発環境では CUDA Toolkit がないことがあるため、
feature `cuda` の検証は CUDA Toolkit が入った環境で行う。

最初の確認:

```bash
cd cuda-oxide
CUDA_HOME=/usr/local/cuda cargo check -p bulletou-cuda-train --features cuda
```

次に追加する検証コマンド:

```bash
cd cuda-oxide
CUDA_HOME=/usr/local/cuda cargo run -p bulletou-cuda-train --features cuda -- \
  --nnue-forward-smoke --device 0
```

Full HalfKP shape:

```bash
cd cuda-oxide
CUDA_HOME=/usr/local/cuda cargo run -p bulletou-cuda-train --features cuda -- \
  --nnue-forward-smoke --nnue-forward-case halfkp --device 0 --debug-readback
```

合格条件:

- tiny shape の固定 weight / 固定 sparse batch を作る。
- CPU scalar golden と `launch_nnue_forward` の GPU output を比較する。
- 絶対誤差 `1e-5` 以下で一致する。
- L0 / concat / L1 / L2 / output のどこで不一致になったかを切り分けられるよう、
  必要なら中間 buffer を host に戻す debug flag を用意する。

### 2026-07-17 CO-007 SFNN forward validation

- Added root-side `fast_sfnn` scalar CPU golden for `SFNN_halfka2_1024_7_64_k3k3`: sparse FT, CReLU, pairwise-mul, LayerStack L1-L3, and PSQT skip.
- Added cuda-oxide SFNN shape / weight / batch / workspace layout.
- Added `bulletou-cuda-train --sfnn-forward-smoke` and kernels: `sfnn_sparse_l0_crelu`, `sfnn_pairwise_concat`, `sfnn_stacked_l1`, `sfnn_l2_input`, `sfnn_stacked_l2_crelu`, `sfnn_stacked_l3_output`.
- WSL2 Ubuntu 24.04 + RTX 4090: `cargo check -p bulletou-cuda-train --features cuda` succeeded.
- WSL2 Ubuntu 24.04 + RTX 4090: `cargo oxide build --arch sm_89 --features cuda -- --package bulletou-cuda-train` succeeded.
- Tiny SFNN smoke: output and debug readback (`stm_l0`, `nstm_l0`, `combined`, `l1`, `l2_input`, `l2`) all had max_abs diff `0`.
- Full synthetic `SFNN_halfka2_1024_7_64_k3k3` smoke: output max_abs diff `0.0000000037252903`; `l1` / `l2_input` max_abs diff `0.0000000018626451`; all within tolerance.
- Added root exporter `export_sfnn_forward_fixture` and fixture format `BOUSFWD1`.
- Exported a HalfKa2 fixture from user-provided HCPE teacher `C:\shogi\teacher\yane-distill-hcpe-20260508shuffled\shuffled-001.hcpe`: `target/bulletou-teacher-sfnn-halfka2.sfnnf`, batch_size=2, buckets `[8, 5]`, CPU output `[0.031887285, 0.035274245]`.
- Ran that teacher fixture through WSL2/cuda-oxide `--sfnn-forward-fixture`: output max_abs diff `0`; `l1` / `l2_input` / `l2` max_abs diff `0.0000000018626451` or less; L0/combined max_abs diff `0`.

### 2026-07-17 CO-008 scalar loss first slice

- Added root-side scalar CPU golden `fast_loss` for weighted sigmoid-MSE:
  `entry_weight * (sigmoid(output) - target)^2`, plus `weighted_sum` and
  `mean = weighted_sum / batch_size`.
- Added cuda-oxide runtime layout for scalar loss buffers:
  `outputs`, `targets`, `entry_weights`, `per_sample`, `weighted_sum`, and
  `mean`.
- Added CUDA kernel `loss_sigmoid_mse_reduce` and host launcher. Correctness
  baseline uses one launched thread per sample for `per_sample`; thread 0 also
  computes `weighted_sum` and `mean`.
- `f32::exp()` / `f32::exp_m1()` currently route through `std` in this
  cuda-oxide revision and are rejected by device collection. The kernel uses
  `core::intrinsics::expf32` behind the crate's CUDA-only nightly feature gate.
- Because libdevice math makes cuda-oxide emit `bulletou_cuda_train.ll` instead
  of PTX, the runtime loader now accepts `.ll` artifacts and builds/loads a
  cubin through cuda-host's LTOIR pipeline. Generated `.cubin`, `.ltoir`,
  `.options`, and `.target` files are ignored by git.
- WSL2 Ubuntu 24.04 + RTX 4090 validation:
  - `cargo check -p bulletou-cuda-train` succeeded.
  - `cargo test -p bulletou-cuda-oxide-runtime loss` succeeded.
  - `cargo check -p bulletou-cuda-train --features cuda` succeeded.
  - `cargo oxide build --arch sm_89 --features cuda -- --package bulletou-cuda-train --release` succeeded.
  - `--loss-smoke --loss-case tiny --debug-readback`: sum max_abs diff
    `0.0000000037252903`; mean max_abs diff `0.0000000009313226`;
    per_sample max_abs diff `0`.
  - `--loss-smoke --loss-case weighted --debug-readback`: sum max_abs diff
    `0.0000000009313226`; mean max_abs diff `0.00000000023283064`;
    per_sample max_abs diff `0`.
- Environment note: Ubuntu noble's `libnvjitlink12` package exposes only
  versioned symbols such as `__nvJitLinkCreate_12_0`, while this cuda-oxide
  revision expects `nvJitLinkCreate`. For the above smoke runs, a temporary
  `/tmp/libnvJitLink_shim.so` was used via `LIBNVJITLINK_PATH`. A proper CUDA
  Toolkit nvJitLink install or a local cuda-oxide/nvjitlink-sys fix should
  replace this shim.
- Remaining CO-008/CO-009 work: WRM value loss / target transform variants,
  parallel reduction, and backward gradients.

### 2026-07-17 CO-008 WRM value loss extension

- Added `ScalarValueLossKind::NnuePytorchWrm` to the root CPU golden. The
  formula matches `examples/bulletou.rs`: `scorenet = output * 600`,
  `q = sigmoid((scorenet - 270) / 340)`,
  `qm = sigmoid((-scorenet - 270) / 340)`,
  `prediction = (1 + q - qm) * 0.5`, and
  `abs(prediction - target)^2.5`.
- Added CUDA kernel `loss_nnue_pytorch_wrm_reduce`, using
  `core::intrinsics::expf32` and `core::intrinsics::powf32` for libdevice
  lowering.
- Added `--loss-kind sigmoid-mse|wrm` to `bulletou-cuda-train --loss-smoke`.
- Validation:
  - `cargo test -p bulletou_lib fast_loss` succeeded.
  - `cargo check -p bulletou-cuda-train` succeeded.
  - `cargo test -p bulletou-cuda-oxide-runtime loss` succeeded.
  - `cargo check -p bulletou-cuda-train --features cuda` succeeded.
  - `cargo oxide build --arch sm_89 --features cuda -- --package bulletou-cuda-train --release` succeeded.
  - `--loss-kind wrm --loss-case tiny --debug-readback`: sum/mean/per_sample
    max_abs diff `0`.
  - `--loss-kind wrm --loss-case weighted --debug-readback`: sum/mean max_abs
    diff `0`; per_sample max_abs diff `0.0000000000000008881784`.
  - Regression `--loss-kind sigmoid-mse --loss-case weighted --debug-readback`
    remained within tolerance: sum max_abs diff `0.0000000009313226`;
    mean max_abs diff `0.00000000023283064`; per_sample max_abs diff `0`.

### 2026-07-17 CO-009 entry: scalar loss output gradients

- Extended the scalar loss CPU golden with `mean_output_gradients`, i.e.
  `d(mean_loss) / d(network_output)` for each sample. This is the seed buffer
  needed by later dense / sparse backward kernels.
- Extended the cuda-oxide scalar loss workspace with a
  `mean_output_gradients` device buffer.
- Updated both loss kernels to write gradients while computing per-sample loss:
  - sigmoid-MSE:
    `entry_weight * 2 * (sigmoid(output) - target) * sigmoid(output) *
    (1 - sigmoid(output)) / batch_size`
  - WRM: derivative of the existing WRM prediction transform and
    `abs(prediction - target)^2.5`, also divided by `batch_size`.
- Validation:
  - `cargo test -p bulletou_lib fast_loss` succeeded.
  - `cargo check -p bulletou-cuda-train` succeeded.
  - `cargo test -p bulletou-cuda-oxide-runtime loss` succeeded.
  - `cargo check -p bulletou-cuda-train --features cuda` succeeded.
  - `cargo oxide build --arch sm_89 --features cuda -- --package bulletou-cuda-train --release` succeeded.
  - `--loss-kind sigmoid-mse --loss-case weighted --debug-readback`:
    `mean_grad` max_abs diff `0.000000000014551915`.
  - `--loss-kind wrm --loss-case weighted --debug-readback`:
    `mean_grad` max_abs diff `0.0000000000000035527137`.

### 2026-07-17 CO-009 dense output backward first slice

- Added cuda-oxide runtime layout for a minimal scalar-output dense backward:
  `DenseOutputBackwardLayout { batch_size, input_len }`.
- Added CUDA kernel `dense_output_backward` for affine output layers:
  - `input_gradients[sample, row] = output_gradient[sample] * weight[row]`
  - `weight_gradients[row] = sum_s output_gradient[s] * input[s, row]`
  - `bias_gradient = sum_s output_gradient[s]`
- Added host launcher and CLI smoke:
  `bulletou-cuda-train --dense-output-backward-smoke`.
- Validation:
  - `cargo check -p bulletou-cuda-train` succeeded.
  - `cargo test -p bulletou-cuda-oxide-runtime backward` succeeded.
  - `cargo check -p bulletou-cuda-train --features cuda` succeeded.
  - `cargo oxide build --arch sm_89 --features cuda -- --package bulletou-cuda-train --release` succeeded.
  - `--dense-output-backward-smoke --debug-readback`: input_grad max_abs
    diff `0`; weight_grad max_abs diff `0.0000000037252903`; bias_grad
    max_abs diff `0`.
- Remaining CO-009 work: hidden dense layers with activation derivatives,
  stacked SFNN layout support, and sparse feature-transformer gradient
  accumulation.
