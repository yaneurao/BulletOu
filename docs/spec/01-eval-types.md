# 01. `--eval-type` 仕様

`bulletou --eval-type <X>` で選択できる学習ターゲットの一覧。やねうら王エンジンが実際に load できる組合せだけを公開する。

## 公開 eval-type 一覧

| `--eval-type` | family | 出力ファイル (per save dir) | engine 側で load 可能か |
|---|---|---|---|
| `KPPT` | KPPT | `KK_synthesized.bin` + `KKP_synthesized.bin` + `KPP_synthesized.bin` (3 ファイル組) | ○ (3 ファイル必須) |
| `KPP_KKPT` | KPPT (factorised) | 同上だが KPP のみ手番チャンネルなしの int16 (約半サイズ) | ○ |
| `NNUE_HALFKP` | NNUE | `nn.bin` 単独 | ○ |
| `NNUE_KP` | NNUE | `nn.bin` 単独 | ○ |
| `NNUE_KA2` | NNUE | `nn.bin` 単独 | ○ (やねうら王 `YANEURAOU_ENGINE_NNUE_ka2_*` ビルド) |
| `NNUE_HALFKPE9` | NNUE | `nn.bin` 単独 | ○ |
| `NNUE_HALFKPVM` | NNUE | `nn.bin` 単独 | ○ |
| `SFNN_HALFKA1HM` | SFNN-1536 (LayerStacks=9) | `nn.bin` 単独 | ○ |
| `SFNN_HALFKA2HM` | SFNN-1536 (LayerStacks=9) | `nn.bin` 単独 | ○ (同上、これが標準) |
| `SFNN_HALFKA2` | SFNN (LayerStacks=9) | `nn.bin` 単独 | ○ (やねうら王 `YANEURAOU_ENGINE_SFNN_halfka2_*_k3k3` ビルド) |
| `SFNN_KA2` | SFNN-1536 (LayerStacks=9) | `nn.bin` 単独 | ○ (やねうら王 `YANEURAOU_ENGINE_SFNN_ka2_*_k3k3` ビルド) |

すべての eval-type で、save dir には別途 `state.bin` (resume 用) と `learn.log` (loss snapshot) が一緒に書かれる。詳細は [04-checkpoint-layout.md](04-checkpoint-layout.md)。

## 内部 helper として存在するが CLI に公開しない要素

KPPT family は内部的に「KK 単独学習」「KKP 単独学習」「KPP 単独学習」を順番に走らせる構造を持つが、これらは **`--eval-type` の値としては公開しない**。

理由: やねうら王の KPPT エンジンは `KK_synthesized.bin` + `KKP_synthesized.bin` + `KPP_synthesized.bin` の 3 ファイル組を要求するので、単一 component だけ出力されたディレクトリは engine が load できず、ユーザー視点で価値が無いため。3 component を統合した `KPPT` / `KPP_KKPT` のみが engine-loadable。

## `--arch` 依存

| `--eval-type` | `--arch` を使うか | 現状サポートする `--arch` 値 |
|---|---|---|
| `KPPT` / `KPP_KKPT` | 使わない (固定 architecture) | n/a |
| `NNUE_HALFKP` / `NNUE_KP` / `NNUE_KA2` / `NNUE_HALFKPE9` / `NNUE_HALFKPVM` | 使う | `NNUE_<feature>_<L1>x2_<L2>_<L3>` |
| `SFNN_HALFKA1HM` / `SFNN_HALFKA2HM` / `SFNN_HALFKA2` / `SFNN_KA2` | 使う | `SFNN_<feature>_<FT>_<H1>_<H2>_k3k3` |

`--arch` の値は、やねうら王 Makefile の `YANEURAOU_ENGINE_` prefix を除いた architecture 名。SFNN の `SFNN_halfkahm2_1536_15_32_k3k3` は `(ft_size=1536, l1_hidden=15, l2_size=32)` と k3k3(king3-by-king3) LayerStacks にマップされ、`l1_hidden + 1` の PSQT shortcut neuron は内部で自動付加される (`fc_0` の出力次元は実際には 16)。

将来 `--arch` の値域を増やす場合は、`bulletou_lib::value::nnue_save` の量子化スケール (qa=127 ClippedReLU 前提) と SIMD パディング規約 (pad32) を踏まえて L0 / L1 サイズを選ぶこと。詳細は [02-nnue-binary.md](02-nnue-binary.md)。

## LayerStack suffix

SFNN family では `--arch` の末尾 suffix がバケット選択ロジック (どのサブネットを使うか) を決め、暗黙的に **LayerStacks 数** (= バケット数) も決まる。LayerStack 専用の別 flag は使わない:

| `--eval-type` | 現状サポートする suffix |
|---|---|
| `SFNN_*` | `k3k3(king3-by-king3)` (= 自玉段 3 区分 × 敵玉段 3 区分 = 9 stacks、やねうら王 `stack_index_for_nnue` 互換) |
| その他すべて | n/a |

`bulletou_lib::game::outputs::ShogiLayerStackBucket9` には `Ply9` / `Progress8*` 等の他バケットモードが実装済みだが、これらは engine 側のバケット選択ロジックと一致しないため CLI に公開しない (= `examples/shogi_layerstack.rs` 経路で実験用にのみアクセス可能)。

## デフォルト `--output` 規約

`--output` 省略時は以下のように自動命名:

| eval-type | デフォルト `--output` |
|---|---|
| `KPPT` | `checkpoints/KPPT` |
| `KPP_KKPT` | `checkpoints/KPP_KKPT` |
| `NNUE_HALFKP` | `checkpoints/NNUE_HALFKP-NNUE_halfkp_256x2_32_32` |
| `NNUE_KP` | `checkpoints/NNUE_KP-NNUE_kp_256x2_32_32` |
| `NNUE_HALFKPE9` | `checkpoints/NNUE_HALFKPE9-NNUE_halfkpe9_256x2_32_32` |
| `NNUE_HALFKPVM` | `checkpoints/NNUE_HALFKPVM-NNUE_halfkpvm_256x2_32_32` |
| `SFNN_HALFKA1HM` | `checkpoints/SFNN_HALFKA1HM-SFNN_halfkahm1_1536_15_32_k3k3` |
| `SFNN_HALFKA2HM` | `checkpoints/SFNN_HALFKA2HM-SFNN_halfkahm2_1536_15_32_k3k3` |

NNUE / SFNN 系は `<eval-type>-<arch>`、KPPT 系は `<eval-type>` のみ。`<eval-type>` / `<arch>` はそれぞれ CLI でユーザーが入力する値をそのまま使う。

## アクティベーション

KPPT および NNUE_HALFKP / NNUE_KP / NNUE_HALFKPE9 / NNUE_HALFKPVM / NNUE_SHARDKP では **ClippedReLU のみ**。SFNN-1536 family は **ClippedReLU + SqrClippedReLU の pair** (`fc_0` 出力に対し CReLU と SqrCReLU を別々に適用してから concat) を使う。歴史的経緯は [05-activation-history.md](05-activation-history.md) を参照。

## Experimental: NNUE_SHARDKP

`NNUE_SHARDKP` is an experimental cuda-cpp NNUE target imported from the
`shardKP` branch. It uses the dense-L0 prototype input
`ShogiShardKp`, where each K+P feature is expanded to one common connection
plus six shard connection IDs.

- default arch: `NNUE_shardkp_c256_s128x64_f6_16_16`
- input dims: `1710 * 7 = 11970`
- max active: `40 * 7 = 280`
- L1 dims: `256 + 128 * 64 = 8448`
- activation: ClippedReLU
- default output: `checkpoints/NNUE_SHARDKP-NNUE_shardkp_c256_s128x64_f6_16_16`

The emitted `nn.bin` is BulletOu-internal experimental output and requires an
engine build that implements the same ShardKP feature hash/layout.
