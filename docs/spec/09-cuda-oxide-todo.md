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
| CO-006 | in-progress | minimal NNUE forward | `NNUE_HALFKP_256x2_32_32` の 1 batch forward を cuda-oxide で一致させる。CPU golden と所有重みレイアウトは追加済み |
| CO-007 | todo | SFNN forward | `SFNN_halfka2_1024_7_64_k3k3` の forward を cuda-oxide で一致させる |
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
- in-progress: `nnue_sparse_l0_crelu` kernel 定義を追加済み。CUDA Toolkit 環境で feature `cuda` の compile と CPU golden の L0 出力比較が必要。
- in-progress: `nnue_concat_l0` / `nnue_dense_l1_crelu` / `nnue_dense_l2_crelu` / `nnue_dense_output` の kernel 定義を追加済み。CUDA Toolkit 環境で compile / launch 検証が必要。
- in-progress: host launch sequence を追加済み。CUDA Toolkit 環境で feature `cuda` の compile と 1 batch 最終 output の CPU golden 比較が必要。
- done: `bulletou-cuda-train --nnue-forward-smoke` CLI と tiny fixed weight / sparse batch の CPU golden 比較を追加。
- blocked: 現 Windows 環境では CUDA Toolkit v13.1 と RTX 4090 は見えるが、`libclang.dll` と `cargo-oxide` が未導入。`cargo check -p bulletou-cuda-train --features cuda` は bindgen の libclang 探索で停止する。実機 launch 検証には LLVM/libclang と cargo-oxide の導入が必要。

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

合格条件:

- tiny shape の固定 weight / 固定 sparse batch を作る。
- CPU scalar golden と `launch_nnue_forward` の GPU output を比較する。
- 絶対誤差 `1e-5` 以下で一致する。
- L0 / concat / L1 / L2 / output のどこで不一致になったかを切り分けられるよう、
  必要なら中間 buffer を host に戻す debug flag を用意する。
