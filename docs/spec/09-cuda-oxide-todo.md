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

### 2026-07-17 CO-009 dense CReLU backward slice

- Added cuda-oxide runtime layout for hidden dense layers with CReLU
  activation derivatives:
  `DenseCReluBackwardLayout { batch_size, input_dim, output_dim }`.
- Added CUDA kernel `dense_crelu_backward`:
  - gates the upstream gradient with the post-CReLU activation
    (`0 < activation < 1` passes; saturated `0` / `1` stops);
  - computes `input_gradients[sample, input]`;
  - accumulates `weight_gradients[input, output]`;
  - accumulates `bias_gradients[output]`.
- Added host launcher and CLI smoke:
  `bulletou-cuda-train --dense-crelu-backward-smoke`.
- Validation:
  - `cargo check -p bulletou-cuda-train` succeeded.
  - `cargo test -p bulletou-cuda-oxide-runtime backward` succeeded.
  - `cargo check -p bulletou-cuda-train --features cuda` succeeded.
  - `cargo oxide build --arch sm_89 --features cuda -- --package bulletou-cuda-train --release` succeeded.
  - `--dense-crelu-backward-smoke --debug-readback`: input_grad max_abs
    diff `0`; weight_grad max_abs diff `0.000000029802322`; bias_grad
    max_abs diff `0`.
- Remaining CO-009 work: map this generic CReLU backward into the NNUE / SFNN
  stacked layer shapes, then add sparse feature-transformer gradient
  accumulation.

### 2026-07-17 CO-009 NNUE dense-stack backward smoke

- Added `bulletou-cuda-train --nnue-dense-backward-smoke`.
- The smoke runs the existing NNUE forward launch first, then chains the
  generic backward kernels across the dense stack:
  `output -> hidden2 CReLU -> hidden1 CReLU -> combined`.
- Extended the smoke with `nnue_l0_crelu_backward`, which splits
  `combined_grad` back into stm/nstm L0 gradients and applies the L0 CReLU
  derivative gate.
- Extended the smoke with `nnue_l0_sparse_backward`, which computes shared
  sparse feature-transformer `l0w` / `l0b` gradients from stm/nstm sparse
  indices and L0 pre-activation gradients.
- Implementation note: cuda-oxide rev `b5d35e0` rejected `DeviceAtomicF32`
  atomic RMW during legacy NVVM IR lowering, so this correctness baseline uses
  a race-free scan kernel: one thread owns each `l0w[feature,row]` or
  `l0b[row]` gradient element and scans the sparse batch.
- Added CPU scalar golden for the same dense-stack backward path, including
  output, L2, and L1 weight/bias gradients plus intermediate activation
  gradients, L0 stm/nstm pre-activation gradients, and sparse L0 weight/bias
  gradients.
- Validation:
  - `cargo check -p bulletou-cuda-train` succeeded.
  - `cargo test -p bulletou-cuda-oxide-runtime backward` succeeded.
  - `cargo check -p bulletou-cuda-train --features cuda` succeeded.
  - `cargo oxide build --arch sm_89 --features cuda -- --package bulletou-cuda-train --release` succeeded.
  - `--nnue-dense-backward-smoke --debug-readback` tiny case succeeded:
    max_abs diff `0` for all compared buffers except `combined_grad` and
    `stm_l0_grad`, both max_abs diff `0.0000000004656613`, and `l0w_grad`
    max_abs diff `0.0000000037252903`.
  - `--nnue-dense-backward-smoke --nnue-forward-case halfkp --debug-readback`
    succeeded for `NNUE_HALFKP_256x2_32_32`: largest observed max_abs diff
    was `0.0000000004656613` (`outw_grad`); `l0w_grad` max_abs diff
    `0.00000000000009947598`, `l0b_grad` max_abs diff
    `0.00000000000017053026`.
- Remaining CO-009 work: actual optimizer integration / gradient buffer
  plumbing and the analogous SFNN stacked dense/backward path.

### 2026-07-17 CO-009 SFNN output backward first slice

- Added cuda-oxide runtime layout for SFNN stacked L3 output backward:
  `SfnnStackedL3BackwardLayout { batch_size, l2_size, l1_out, num_stacks }`.
- Added CUDA kernel `sfnn_stacked_l3_backward`:
  - computes `l2_gradients[sample, row]`;
  - writes the L1 skip-connection gradient at `l1_hidden`;
  - accumulates stacked `l3w_gradients[row, stack]`;
  - accumulates stacked `l3b_gradients[stack]`.
- Added host launcher and CLI smoke:
  `bulletou-cuda-train --sfnn-output-backward-smoke`.
- Validation:
  - `cargo check -p bulletou-cuda-train` succeeded.
  - `cargo test -p bulletou-cuda-oxide-runtime backward` succeeded.
  - `cargo check -p bulletou-cuda-train --features cuda` succeeded.
  - `cargo oxide build --arch sm_89 --features cuda -- --package bulletou-cuda-train --release` succeeded.
  - `--sfnn-output-backward-smoke --debug-readback` tiny case succeeded:
    `l2_grad`, `l1_grad`, `l3w_grad`, and `l3b_grad` max_abs diff `0`.
  - `--sfnn-output-backward-smoke --sfnn-forward-case halfka2 --debug-readback`
    succeeded for `SFNN_halfka2_1024_7_64_k3k3`: all compared buffers
    max_abs diff `0`.
- Remaining CO-009 work: optimizer update kernel / trainer loop integration.

### 2026-07-17 CO-009 SFNN backward workspace ownership

- Added cuda-oxide runtime workspace ownership for the SFNN backward pass:
  `SfnnBackwardWorkspaceLayout { shape, batch_size, max_active }` and
  `SfnnBackwardWorkspace`.
- The workspace now owns all reusable SFNN gradient buffers after the scalar
  output seed:
  - intermediate gradients: `l2_gradients`, `l1_gradients`,
    `l2_input_gradients`, `combined_gradients`, `stm_l0_gradients`,
    `nstm_l0_gradients`, `stm_l0_pre_gradients`, `nstm_l0_pre_gradients`;
  - parameter gradients: `l0w/l0b`, `l1w/l1b`, `l2w/l2b`, and `l3w/l3b`.
- Updated `--sfnn-dense-backward-smoke` to allocate this workspace once and
  pass its buffers through the existing backward launch chain instead of
  scattering `DeviceBuffer::zeroed` calls through the smoke function.
- The output seed remains outside the backward workspace so the real trainer
  can pass loss-produced `mean_output_gradients` directly into the backward
  chain.
- Validation:
  - `cargo check -p bulletou-cuda-train` succeeded.
  - `cargo test -p bulletou-cuda-oxide-runtime backward` succeeded.
  - `cargo check -p bulletou-cuda-train --features cuda` succeeded.
  - `cargo oxide build --arch sm_89 --features cuda -- --package bulletou-cuda-train --release` succeeded.
  - `--sfnn-dense-backward-smoke --debug-readback` tiny case succeeded with
    the same comparison results as before the refactor.
  - `--sfnn-dense-backward-smoke --sfnn-forward-case halfka2 --debug-readback`
    succeeded for `SFNN_halfka2_1024_7_64_k3k3`; the largest observed max_abs
    diff remained `0.00000000005820766` (`comb_grad`).
- Remaining CO-009 work: optimizer update kernel / trainer loop integration.

### 2026-07-17 CO-009 SFNN sparse L0 CReLU backward smoke

- Added cuda-oxide runtime layout for SFNN sparse L0 backward:
  `SfnnL0SparseBackwardLayout { batch_size, max_active, input_size, ft_size }`.
- Added CUDA kernel `sfnn_l0_sparse_backward`:
  - applies the CReLU derivative gate to pairwise-produced `stm_l0_grad` and
    `nstm_l0_grad`, producing `stm_l0_pre` / `nstm_l0_pre` diagnostic buffers;
  - accumulates shared sparse feature-transformer `l0w_grad`;
  - accumulates shared sparse feature-transformer `l0b_grad`;
  - uses the same race-free scan strategy as the NNUE L0 sparse correctness
    baseline because cuda-oxide rev `b5d35e0` does not yet lower the needed
    atomic RMW path.
- Extended `--sfnn-dense-backward-smoke` to cover the full current SFNN
  forward stack:
  `stacked L3 output backward -> stacked L2 CReLU backward -> L2-input
  transform backward -> stacked L1 backward -> pairwise backward -> sparse L0
  CReLU backward`.
- Validation:
  - `cargo check -p bulletou-cuda-train` succeeded.
  - `cargo test -p bulletou-cuda-oxide-runtime backward` succeeded.
  - `cargo check -p bulletou-cuda-train --features cuda` succeeded.
  - `cargo oxide build --arch sm_89 --features cuda -- --package bulletou-cuda-train --release` succeeded.
  - `--sfnn-dense-backward-smoke --debug-readback` tiny case succeeded:
    all compared buffers max_abs diff `0` except `l0b_grad`, whose max_abs
    diff was `0.000000029802322`.
  - `--sfnn-dense-backward-smoke --sfnn-forward-case halfka2 --debug-readback`
    succeeded for `SFNN_halfka2_1024_7_64_k3k3`: `stm_l0_pre` max_abs diff
    `0.000000000014551915`, `nstm_l0_pre` max_abs diff
    `0.000000000007275958`, `l0w_grad` max_abs diff
    `0.000000000014551915`, `l0b_grad` max_abs diff
    `0.000000000021827873`, and all previous compared buffers remained within
    tolerance.
- Remaining CO-009 work: gradient-buffer ownership/plumbing and optimizer
  integration in the trainer.

### 2026-07-17 CO-009 SFNN pairwise backward smoke

- Added cuda-oxide runtime layout for SFNN pairwise-concat backward:
  `SfnnPairwiseBackwardLayout { batch_size, ft_size }`.
- Added CUDA kernel `sfnn_pairwise_backward`:
  - maps `combined_grad[0..ft_size/2]` through the stm pairwise products to
    `stm_l0_grad`;
  - maps `combined_grad[ft_size/2..]` through the nstm pairwise products to
    `nstm_l0_grad`;
  - uses the same `(127 / 128)` scale as forward.
- Extended `--sfnn-dense-backward-smoke` to chain:
  `stacked L3 output backward -> stacked L2 CReLU backward -> L2-input
  transform backward -> stacked L1 backward -> pairwise backward`.
- Validation:
  - `cargo check -p bulletou-cuda-train` succeeded.
  - `cargo test -p bulletou-cuda-oxide-runtime backward` succeeded.
  - `cargo check -p bulletou-cuda-train --features cuda` succeeded.
  - `cargo oxide build --arch sm_89 --features cuda -- --package bulletou-cuda-train --release` succeeded.
  - `--sfnn-dense-backward-smoke --debug-readback` tiny case succeeded:
    all compared buffers max_abs diff `0`.
  - `--sfnn-dense-backward-smoke --sfnn-forward-case halfka2 --debug-readback`
    succeeded for `SFNN_halfka2_1024_7_64_k3k3`: `stm_l0_grad` max_abs diff
    `0.000000000014551915`, `nstm_l0_grad` max_abs diff
    `0.000000000007275958`, and all previous compared buffers remained within
    tolerance.
- Remaining CO-009 work: SFNN sparse L0 CReLU backward and integration into
  trainer gradient buffers.

### 2026-07-17 CO-009 SFNN stacked L1 backward smoke

- Added cuda-oxide runtime layout for generic SFNN stacked affine backward:
  `SfnnStackedAffineBackwardLayout { batch_size, input_dim, output_dim,
  num_stacks }`.
- Added CUDA kernel `sfnn_stacked_affine_backward`:
  - computes `input_gradients[sample, in_col]` through the active stack;
  - accumulates stacked `weight_gradients[in_col, stack, out_col]`;
  - accumulates stacked `bias_gradients[stack, out_col]`.
- Extended `--sfnn-dense-backward-smoke` to chain:
  `stacked L3 output backward -> stacked L2 CReLU backward -> L2-input
  transform backward -> stacked L1 backward`.
- The new L1 step produces `combined_grad`, `l1w_grad`, and `l1b_grad`
  buffers against the CPU scalar golden.
- Validation:
  - `cargo check -p bulletou-cuda-train` succeeded.
  - `cargo test -p bulletou-cuda-oxide-runtime backward` succeeded.
  - `cargo check -p bulletou-cuda-train --features cuda` succeeded.
  - `cargo oxide build --arch sm_89 --features cuda -- --package bulletou-cuda-train --release` succeeded.
  - `--sfnn-dense-backward-smoke --debug-readback` tiny case succeeded:
    all compared buffers max_abs diff `0`.
  - `--sfnn-dense-backward-smoke --sfnn-forward-case halfka2 --debug-readback`
    succeeded for `SFNN_halfka2_1024_7_64_k3k3`: `comb_grad` max_abs diff
    `0.00000000005820766`, `l1w_grad` max_abs diff
    `0.00000000000017053026`, `l1b_grad` max_abs diff
    `0.000000000007275958`, and all previous compared buffers remained within
    tolerance.
- Remaining CO-009 work: SFNN pairwise/L0 backward and integration into
  trainer gradient buffers.

### 2026-07-17 CO-009 SFNN L2-input transform backward smoke

- Added cuda-oxide runtime layout for the weightless SFNN L2-input transform
  backward:
  `SfnnL2InputBackwardLayout { batch_size, l1_hidden }`.
- Added CUDA kernel `sfnn_l2_input_backward`:
  - maps gradients from `l2_input[0..l1_hidden]`, the squared/absolute branch,
    back to `l1[0..l1_hidden]` with derivative
    `2 * l1_value * (127 / 128)` when the branch is inside CReLU range;
  - maps gradients from `l2_input[l1_hidden..]`, the linear CReLU branch,
    back to `l1[0..l1_hidden]`;
  - adds into the existing `l1_gradients` buffer while preserving the final
    `l1_hidden` skip-connection column already written by L3 backward.
- Extended `--sfnn-dense-backward-smoke` to chain:
  `stacked L3 output backward -> stacked L2 CReLU backward -> L2-input
  transform backward`.
- Validation:
  - `cargo check -p bulletou-cuda-train` succeeded.
  - `cargo test -p bulletou-cuda-oxide-runtime backward` succeeded.
  - `cargo check -p bulletou-cuda-train --features cuda` succeeded.
  - `cargo oxide build --arch sm_89 --features cuda -- --package bulletou-cuda-train --release` succeeded.
  - `--sfnn-dense-backward-smoke --debug-readback` tiny case succeeded:
    all compared buffers max_abs diff `0`.
  - `--sfnn-dense-backward-smoke --sfnn-forward-case halfka2 --debug-readback`
    succeeded for `SFNN_halfka2_1024_7_64_k3k3`: `l1_grad` max_abs diff
    `0.000000000007275958`, `l2_in_grad` max_abs diff
    `0.000000000014551915`, `l2w_grad` max_abs diff
    `0.0000000000018189894`, and all other compared buffers max_abs diff `0`.
- Remaining CO-009 work: SFNN stacked L1 backward, pairwise/L0 backward, and
  integration into trainer gradient buffers.

### 2026-07-17 CO-009 SFNN L2 CReLU backward smoke

- Added cuda-oxide runtime layout for SFNN stacked CReLU backward:
  `SfnnStackedCReluBackwardLayout { batch_size, input_dim, output_dim,
  num_stacks }`.
- Added CUDA kernel `sfnn_stacked_crelu_backward`:
  - computes `input_gradients[sample, in_col]` through the active stack;
  - applies the CReLU derivative gate from the post-CReLU activation;
  - accumulates stacked `weight_gradients[in_col, stack, out_col]`;
  - accumulates stacked `bias_gradients[stack, out_col]`.
- Added host launcher and extended the SFNN backward smoke. The preferred CLI
  flag is now `bulletou-cuda-train --sfnn-dense-backward-smoke`; the previous
  `--sfnn-output-backward-smoke` remains as an alias.
- The smoke now runs SFNN forward, then chains:
  `stacked L3 output backward -> stacked L2 CReLU backward`.
- Validation:
  - `cargo check -p bulletou-cuda-train` succeeded.
  - `cargo test -p bulletou-cuda-oxide-runtime backward` succeeded.
  - `cargo check -p bulletou-cuda-train --features cuda` succeeded.
  - `cargo oxide build --arch sm_89 --features cuda -- --package bulletou-cuda-train --release` succeeded.
  - `--sfnn-dense-backward-smoke --debug-readback` tiny case succeeded:
    `l2_grad`, `l1_grad`, `l3w_grad`, `l3b_grad`, `l2_in_grad`, `l2w_grad`,
    and `l2b_grad` max_abs diff `0`.
  - `--sfnn-dense-backward-smoke --sfnn-forward-case halfka2 --debug-readback`
    succeeded for `SFNN_halfka2_1024_7_64_k3k3`: `l2_in_grad` max_abs diff
    `0.000000000014551915`, `l2w_grad` max_abs diff
    `0.0000000000018189894`, and all other compared buffers max_abs diff `0`.
- Remaining CO-009 work: SFNN L2-input transform backward, stacked L1
  backward, pairwise/L0 backward, and integration into trainer gradient
  buffers.

### 2026-07-17 CO-010 AdamW update smoke

- Added cuda-oxide runtime layout/params for a fused AdamW-style update:
  `AdamWUpdateLayout { len }` and `AdamWUpdateParams`.
- Added CUDA kernel `adamw_update`:
  - applies `gradient_factor`;
  - applies decoupled weight decay before the gradient step;
  - updates momentum and velocity buffers;
  - divides by `sqrt(velocity) + epsilon`;
  - clamps the updated weight into `[min_weight, max_weight]`.
- Added host launcher and CLI:
  `bulletou-cuda-train --adamw-update-smoke`.
- The tiny smoke uses 7 parameters to exercise a non-round vector length and
  clamp behavior.
- Validation:
  - `cargo check -p bulletou-cuda-train` succeeded.
  - `cargo test -p bulletou-cuda-oxide-runtime optimizer` succeeded.
  - `cargo check -p bulletou-cuda-train --features cuda` succeeded in WSL2
    Ubuntu-24.04.
  - `cargo oxide build --arch sm_89 --features cuda -- --package
    bulletou-cuda-train --release` succeeded in WSL2 Ubuntu-24.04.
  - `--adamw-update-smoke --debug-readback` succeeded on RTX 4090:
    `weights`, `momentum`, and `velocity` all had max_abs diff `0`.
- Remaining CO-010 work: wire this update into the real trainer parameter
  groups and reconcile the existing optimizer variant naming (Ranger/RAdam vs
  current AdamW baseline).

### 2026-07-17 CO-010 RAdam update smoke

- Added cuda-oxide runtime layout/params for a fused RAdam update:
  `RAdamUpdateLayout { len }` and `RAdamUpdateParams`.
- Added host-side RAdam step-size calculation matching the existing
  `crates/trainer/src/optimiser/radam.rs` formula.
- Added CUDA kernel `radam_update`:
  - applies `gradient_factor`;
  - uses `learning_rate * step_size` as the effective rate, matching existing
    RAdam weight decay behavior;
  - updates momentum and velocity buffers;
  - conditionally divides by `sqrt(velocity) + epsilon` when the rectified
    branch is active;
  - clamps the updated weight into `[min_weight, max_weight]`.
- Added host launcher and CLI:
  `bulletou-cuda-train --radam-update-smoke`.
- The smoke runs two 7-parameter cases: `step=1` for the warmup/no-denominator
  branch and `step=6` for the rectified denominator branch.
- Validation:
  - `cargo check -p bulletou-cuda-train` succeeded.
  - `cargo test -p bulletou-cuda-oxide-runtime optimizer` succeeded.
  - `cargo check -p bulletou-cuda-train --features cuda` succeeded in WSL2
    Ubuntu-24.04.
  - `cargo oxide build --arch sm_89 --features cuda -- --package
    bulletou-cuda-train --release` succeeded in WSL2 Ubuntu-24.04.
  - `--radam-update-smoke --debug-readback` succeeded on RTX 4090:
    warmup/no-denominator and rectified-denominator cases both had max_abs diff
    `0` for `weights`, `momentum`, and `velocity`.
- Remaining CO-010 work: add Lookahead/Ranger smoke, then wire optimizer state
  buffers into the real trainer path.

### 2026-07-17 CO-010 Ranger Lookahead smoke

- Added cuda-oxide runtime layout/params for the Ranger lookahead step:
  `RangerLookaheadLayout { len }` and `RangerLookaheadParams { alpha }`.
- Added CUDA kernel `ranger_lookahead` matching the existing
  `crates/trainer/src/optimiser/ranger.rs` formula:
  `new = alpha * fast + (1 - alpha) * slow`, then write `new` to both fast and
  slow parameter buffers.
- Added host launcher and CLI:
  `bulletou-cuda-train --ranger-lookahead-smoke`.
- The smoke uses a 7-parameter tiny case with `alpha=0.35` to exercise a
  non-round vector length and non-default interpolation factor.
- Validation:
  - `cargo check -p bulletou-cuda-train` succeeded.
  - `cargo test -p bulletou-cuda-oxide-runtime optimizer` succeeded.
  - `cargo check -p bulletou-cuda-train --features cuda` succeeded in WSL2
    Ubuntu-24.04.
  - `cargo oxide build --arch sm_89 --features cuda -- --package
    bulletou-cuda-train --release` succeeded in WSL2 Ubuntu-24.04.
  - `--ranger-lookahead-smoke --debug-readback` succeeded on RTX 4090:
    `weights` and `slow_params` both had max_abs diff `0`.
- Remaining CO-010 work: add full Ranger chain smoke (`RAdam update ->
  conditional Lookahead`), then wire optimizer state buffers into the real
  trainer path.

### 2026-07-17 CO-010 Ranger update chain smoke

- Added cuda-oxide runtime layout/params for the composed Ranger update:
  `RangerUpdateLayout { len }` and `RangerUpdateParams { radam, lookahead, k }`.
- Added host launcher `launch_ranger_update`:
  - always launches the RAdam update for the current step;
  - launches `ranger_lookahead` only when `step % k == 0`.
- Added CLI:
  `bulletou-cuda-train --ranger-update-smoke`.
- The smoke runs six 7-parameter RAdam steps with `k=3`, so Lookahead fires on
  steps 3 and 6. It compares final `weights`, `momentum`, `velocity`, and
  `slow_params` against a CPU scalar golden.
- Validation:
  - `cargo check -p bulletou-cuda-train` succeeded.
  - `cargo test -p bulletou-cuda-oxide-runtime optimizer` succeeded.
  - `cargo check -p bulletou-cuda-train --features cuda` succeeded in WSL2
    Ubuntu-24.04.
  - `cargo oxide build --arch sm_89 --features cuda -- --package
    bulletou-cuda-train --release` succeeded in WSL2 Ubuntu-24.04.
  - `--ranger-update-smoke --debug-readback` succeeded on RTX 4090:
    Lookahead fired on steps `[3, 6]`; `weights` and `slow_params` max_abs diff
    `0`, `momentum` max_abs diff `0.00000000011641532`, and `velocity`
    max_abs diff `0.0000000000009094947`.
- Remaining CO-010 work: wire optimizer state buffers into the real trainer
  path.

### 2026-07-17 CO-010 optimizer state buffers

- Added cuda-oxide runtime ownership structs for optimizer state buffers:
  - `OptimizerStateLayout { len }`;
  - `MomentumVelocityDeviceState` for AdamW/RAdam-style `momentum` and
    `velocity` buffers;
  - `RangerOptimizerState` for Ranger `momentum`, `velocity`, and
    `slow_params` buffers.
- Added host-state validation helpers for loading state buffers from checkpoint
  or fixture data before uploading to CUDA.
- Refactored `--ranger-update-smoke` to allocate and pass
  `RangerOptimizerState` instead of naked `DeviceBuffer` fields, matching the
  ownership boundary needed by the future trainer path.
- Validation:
  - `cargo check -p bulletou-cuda-train` succeeded.
  - `cargo test -p bulletou-cuda-oxide-runtime optimizer` succeeded.
  - `cargo check -p bulletou-cuda-train --features cuda` succeeded in WSL2
    Ubuntu-24.04.
  - `cargo oxide build --arch sm_89 --features cuda -- --package
    bulletou-cuda-train --release` succeeded in WSL2 Ubuntu-24.04.
  - `--ranger-update-smoke --debug-readback` succeeded on RTX 4090 through the
    `RangerOptimizerState` wrapper path: `weights` and `slow_params` max_abs
    diff `0`, `momentum` max_abs diff `0.00000000011641532`, and `velocity`
    max_abs diff `0.0000000000009094947`.
- Remaining CO-010 work: wire these state objects to real NNUE/SFNN parameter
  groups.

### 2026-07-17 CO-010 NNUE/SFNN optimizer state bundles

- Added runtime layouts that mirror the real network parameter groups:
  - `NnueOptimizerStateLayout` maps NNUE `l0w`, `l0b`, `l1w`, `l1b`,
    `l2w`, `l2b`, `outw`, and `outb` to per-tensor
    `OptimizerStateLayout`s.
  - `SfnnOptimizerStateLayout` maps SFNN `l0w`, `l0b`, `l1w`, `l1b`,
    `l2w`, `l2b`, `l3w`, and `l3b` to per-tensor
    `OptimizerStateLayout`s.
- Added CUDA-gated ownership bundles:
  - `NnueRangerOptimizerStates`;
  - `SfnnRangerOptimizerStates`.
- Added `RangerOptimizerState::zeroed_with_host_slow_params` so real trainer
  initialization can zero `momentum`/`velocity` while copying `slow_params`
  from the current weights. This avoids the Lookahead state accidentally
  starting from all zeros when weights are loaded or randomly initialized.
- Added optimizer layout unit tests for tiny NNUE/SFNN shapes, including total
  parameter count and total Ranger state count.
- Validation:
  - `cargo check -p bulletou-cuda-train` succeeded.
  - `cargo test -p bulletou-cuda-oxide-runtime optimizer` succeeded with 25
    optimizer tests.
  - `cargo check -p bulletou-cuda-train --features cuda` succeeded in WSL2
    Ubuntu-24.04.
  - `cargo oxide build --arch sm_89 --features cuda -- --package
    bulletou-cuda-train --release` succeeded in WSL2 Ubuntu-24.04.
- Remaining CO-010 work: bind these per-tensor state bundles to actual
  NNUE/SFNN gradient buffers and launch the Ranger update over each parameter
  group in the real trainer path.

### 2026-07-17 CO-010 SFNN Ranger update launcher

- Refactored `launch_ranger_update` to take `RangerOptimizerState` directly
  instead of separate `momentum`, `velocity`, and `slow_params` buffers.
  The launcher now verifies the state length matches the update layout before
  dispatching kernels.
- Added `launch_sfnn_ranger_update`, which wires SFNN forward weights,
  `SfnnBackwardWorkspace` gradient buffers, and `SfnnRangerOptimizerStates`
  together for all 8 SFNN parameter groups:
  `l0w`, `l0b`, `l1w`, `l1b`, `l2w`, `l2b`, `l3w`, and `l3b`.
- Added shape guards so SFNN weights, gradients, and optimizer states must
  agree before any per-tensor update launches.
- Updated the Ranger chain smoke path to pass the `RangerOptimizerState`
  wrapper directly.
- Validation:
  - `cargo check -p bulletou-cuda-train` succeeded.
  - `cargo check -p bulletou-cuda-train --features cuda` succeeded in WSL2
    Ubuntu-24.04.
  - `--ranger-update-smoke --debug-readback` succeeded on RTX 4090 through the
    `RangerOptimizerState` launcher path: `weights` and `slow_params` max_abs
    diff `0`, `momentum` max_abs diff `0.00000000011641532`, and `velocity`
    max_abs diff `0.0000000000009094947`.
  - `cargo oxide build --arch sm_89 --features cuda -- --package
    bulletou-cuda-train --release` succeeded in WSL2 Ubuntu-24.04.
- Remaining CO-010 work: add the analogous NNUE gradient/state launcher shape,
  then connect these launchers to the real trainer loop.
