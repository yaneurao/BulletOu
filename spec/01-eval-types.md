# 01. `--eval-type` 仕様

`bulletou --eval-type <X>` で選択できる学習ターゲットの一覧。やねうら王エンジンが実際に load できる組合せだけを公開する。

## 公開 eval-type 一覧

| `--eval-type` | family | 出力ファイル (per save dir) | engine 側で load 可能か |
|---|---|---|---|
| `KPPT` | KPPT | `KK_synthesized.bin` + `KKP_synthesized.bin` + `KPP_synthesized.bin` (3 ファイル組) | ○ (3 ファイル必須) |
| `KPP_KKPT` | KPPT (factorised) | 同上だが KPP のみ手番チャンネルなしの int16 (約半サイズ) | ○ |
| `NNUE_HALFKP` | NNUE | `nn.bin` 単独 | ○ |
| `NNUE_KP` | NNUE | `nn.bin` 単独 | ○ |

すべての eval-type で、save dir には別途 `state.bin` (resume 用) と `learn.log` (loss snapshot) が一緒に書かれる。詳細は [04-checkpoint-layout.md](04-checkpoint-layout.md)。

## 内部 helper として存在するが CLI に公開しない要素

KPPT family は内部的に「KK 単独学習」「KKP 単独学習」「KPP 単独学習」を順番に走らせる構造を持つ (`run_kppt_kk` / `run_kppt_kkp` / `run_kppt_kpp` という Rust 関数として存在する) が、これらは **`--eval-type` の値としては公開しない**。

理由: やねうら王の KPPT エンジンは `KK_synthesized.bin` + `KKP_synthesized.bin` + `KPP_synthesized.bin` の 3 ファイル組を要求するので、単一 component だけ出力されたディレクトリは engine が load できず、ユーザー視点で価値が無いため。3 component を統合した `KPPT` / `KPP_KKPT` のみが engine-loadable。

## `--arch` 依存

| `--eval-type` | `--arch` を使うか | 現状サポートする `--arch` 値 |
|---|---|---|
| `KPPT` / `KPP_KKPT` | 使わない (固定 architecture) | n/a |
| `NNUE_HALFKP` / `NNUE_KP` | 使う | `256x2-32-32` (default) / `384x2-8-96` / `512x2-8-64` / `768x2-16-64` / `1024x2-8-32` / `1024x2-8-64` (やねうら王が `NNUE_halfkp_*` で配布している全 preset と一致) |

`--arch` の値は `<L1>x2-<L2>-<L3>` 表記 (Stockfish 慣習に準拠)。`x2` は dual-perspective を意味する固定リテラル。

将来 `--arch` の値域を増やす場合は、`bullet_lib::value::nnue_save` の量子化スケール (qa=127 ClippedReLU 前提) と SIMD パディング規約 (pad32) を踏まえて L0 / L1 サイズを選ぶこと。詳細は [02-nnue-binary.md](02-nnue-binary.md)。

## デフォルト `--output` 規約

`--output` 省略時は以下のように自動命名:

| eval-type | デフォルト `--output` |
|---|---|
| `KPPT` | `checkpoints/KPPT` |
| `KPP_KKPT` | `checkpoints/KPP_KKPT` |
| `NNUE_HALFKP` | `checkpoints/NNUE_HALFKP-256x2-32-32` |
| `NNUE_KP` | `checkpoints/NNUE_KP-256x2-32-32` |

NNUE 系は `<eval-type>-<arch>`、KPPT 系は `<eval-type>` のみ。`<eval-type>` と `<arch>` はそれぞれ CLI でユーザーが入力する値 (SCREAMING_SNAKE_CASE / `256x2-32-32` 表記) をそのまま使う。

## アクティベーション

すべての KPPT / NNUE eval-type で **ClippedReLU** を使う。SqrClippedReLU (SCReLU) は使わない。歴史的経緯は [05-activation-history.md](05-activation-history.md) を参照。
