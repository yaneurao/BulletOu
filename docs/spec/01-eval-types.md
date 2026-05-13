# 01. `--eval-type` 仕様

`bulletou --eval-type <X>` で選択できる学習ターゲットの一覧。やねうら王エンジンが実際に load できる組合せだけを公開する。

## 公開 eval-type 一覧

| `--eval-type` | family | 出力ファイル (per save dir) | engine 側で load 可能か |
|---|---|---|---|
| `KPPT` | KPPT | `KK_synthesized.bin` + `KKP_synthesized.bin` + `KPP_synthesized.bin` (3 ファイル組) | ○ (3 ファイル必須) |
| `KPP_KKPT` | KPPT (factorised) | 同上だが KPP のみ手番チャンネルなしの int16 (約半サイズ) | ○ |
| `NNUE_HALFKP` | NNUE | `nn.bin` 単独 | ○ |
| `NNUE_KP` | NNUE | `nn.bin` 単独 | ○ |
| `NNUE_HALFKPE9` | NNUE | `nn.bin` 単独 | ○ |
| `NNUE_HALFKPVM` | NNUE | `nn.bin` 単独 | ○ |
| `SFNN_HALFKA1HM` | SFNN-1536 (LayerStacks=9) | `nn.bin` 単独 | ○ (やねうら王 `SFNNwoP1536` ビルド) |
| `SFNN_HALFKA2HM` | SFNN-1536 (LayerStacks=9) | `nn.bin` 単独 | ○ (同上、これが標準) |

すべての eval-type で、save dir には別途 `state.bin` (resume 用) と `learn.log` (loss snapshot) が一緒に書かれる。詳細は [04-checkpoint-layout.md](04-checkpoint-layout.md)。

## 内部 helper として存在するが CLI に公開しない要素

KPPT family は内部的に「KK 単独学習」「KKP 単独学習」「KPP 単独学習」を順番に走らせる構造を持つ (`run_kppt_kk` / `run_kppt_kkp` / `run_kppt_kpp` という Rust 関数として存在する) が、これらは **`--eval-type` の値としては公開しない**。

理由: やねうら王の KPPT エンジンは `KK_synthesized.bin` + `KKP_synthesized.bin` + `KPP_synthesized.bin` の 3 ファイル組を要求するので、単一 component だけ出力されたディレクトリは engine が load できず、ユーザー視点で価値が無いため。3 component を統合した `KPPT` / `KPP_KKPT` のみが engine-loadable。

## `--arch` 依存

| `--eval-type` | `--arch` を使うか | 現状サポートする `--arch` 値 |
|---|---|---|
| `KPPT` / `KPP_KKPT` | 使わない (固定 architecture) | n/a |
| `NNUE_HALFKP` / `NNUE_KP` / `NNUE_HALFKPE9` / `NNUE_HALFKPVM` | 使う | `256x2-32-32` (default)<br>`384x2-8-96`<br>`512x2-8-64`<br>`768x2-16-64`<br>`1024x2-8-32`<br>`1024x2-8-64`<br>(やねうら王が `NNUE_halfkp_*` で配布している全 preset と一致) |
| `SFNN_HALFKA1HM` / `SFNN_HALFKA2HM` | 使う | `1536x2-15-32` のみ engine-loadable (やねうら王同梱 SFNN ビルドは 1 preset のみ)。他の preset を指定すると学習は通るが nn.bin は engine に load できない (ablation 用途) |

`--arch` の値は `<L1>x2-<L2>-<L3>` 表記 (Stockfish 慣習に準拠)。`x2` は dual-perspective を意味する固定リテラル。SFNN の `1536x2-15-32` は `(ft_size=1536, l1_hidden=15, l2_size=32)` にマップされ、`l1_hidden + 1` の PSQT shortcut neuron は内部で自動付加される (`fc_0` の出力次元は実際には 16)。

将来 `--arch` の値域を増やす場合は、`bulletou_lib::value::nnue_save` の量子化スケール (qa=127 ClippedReLU 前提) と SIMD パディング規約 (pad32) を踏まえて L0 / L1 サイズを選ぶこと。詳細は [02-nnue-binary.md](02-nnue-binary.md)。

## `--layerstack` 依存

SFNN family のみが消費する flag。バケット選択ロジック (どのサブネットを使うか) を決め、暗黙的に **LayerStacks 数** (= バケット数) も決まる:

| `--eval-type` | `--layerstack` を使うか | 現状サポートする値 |
|---|---|---|
| `SFNN_HALFKA1HM` / `SFNN_HALFKA2HM` | 使う | `king3-by-king3` (= 自玉段 3 区分 × 敵玉段 3 区分 = 9 stacks、やねうら王 `stack_index_for_nnue` 互換) |
| その他すべて | 使わない (= 単一 NN) | n/a |

`bulletou_lib::game::outputs::ShogiLayerStackBucket9` には `Ply9` / `Progress8*` 等の他バケットモードが実装済みだが、これらは engine 側のバケット選択ロジックと一致しないため CLI に公開しない (= `examples/shogi_layerstack.rs` 経路で実験用にのみアクセス可能)。

## デフォルト `--output` 規約

`--output` 省略時は以下のように自動命名:

| eval-type | デフォルト `--output` |
|---|---|
| `KPPT` | `checkpoints/KPPT` |
| `KPP_KKPT` | `checkpoints/KPP_KKPT` |
| `NNUE_HALFKP` | `checkpoints/NNUE_HALFKP-256x2-32-32` |
| `NNUE_KP` | `checkpoints/NNUE_KP-256x2-32-32` |
| `NNUE_HALFKPE9` | `checkpoints/NNUE_HALFKPE9-256x2-32-32` |
| `NNUE_HALFKPVM` | `checkpoints/NNUE_HALFKPVM-256x2-32-32` |
| `SFNN_HALFKA1HM` | `checkpoints/SFNN_HALFKA1HM-1536x2-15-32-king3-by-king3` |
| `SFNN_HALFKA2HM` | `checkpoints/SFNN_HALFKA2HM-1536x2-15-32-king3-by-king3` |

NNUE 系は `<eval-type>-<arch>`、SFNN 系は `<eval-type>-<arch>-<layerstack>`、KPPT 系は `<eval-type>` のみ。`<eval-type>` / `<arch>` / `<layerstack>` はそれぞれ CLI でユーザーが入力する値をそのまま使う。

## アクティベーション

KPPT および NNUE_HALFKP / NNUE_KP / NNUE_HALFKPE9 / NNUE_HALFKPVM では **ClippedReLU のみ**。SFNN-1536 family は **ClippedReLU + SqrClippedReLU の pair** (`fc_0` 出力に対し CReLU と SqrCReLU を別々に適用してから concat) を使う。歴史的経緯は [05-activation-history.md](05-activation-history.md) を参照。
