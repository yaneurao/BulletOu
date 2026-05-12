# 05. 活性化関数の歴史 (ClippedReLU vs SqrClippedReLU)

NNUE 系 architecture を再実装する際に **「いつ何の活性化関数が入ったか」を取り違えてはいけない** ので、やねうら王側の変遷を時系列で記録しておく。

## 結論

| `--eval-type` | architecture file | 活性化関数 |
|---|---|---|
| `NNUE_HALFKP` (256x2-32-32) | `halfkp_256x2-32-32.h` | **ClippedReLU** |
| `NNUE_KP` (256x2-32-32) | `kp_256x2-32-32.h` | **ClippedReLU** |
| (将来) SFNNwoPSQT-1536 | `sfnnwop-1536.h` | **SqrClippedReLU** (一部層) + ClippedReLU (他層) |

## ClippedReLU の定義

```cpp
// やねうら王 source/eval/nnue/layers/clipped_relu.h より
output[i] = max(0, min(127, input[i] >> kWeightScaleBits));
```

- 出力 range: `0..127`
- 量子化 `qa = 127` (L0 出力スケール)
- 2 乗していない素直な ReLU + クリップ

## SqrClippedReLU (= SCReLU) の定義

```cpp
// 概念的に
clipped = max(0, min(127, input[i] >> shift));
output[i] = clipped * clipped;  // (※ 内部表現の関係で 127 にスケールダウンする shift が入る)
```

- 出力は 2 乗されてから 127 スケールに戻される
- 量子化 `qa = 255` (L0 内部スケール) + L1 bias scale が `127 × qb` (出力が 127 に正規化されるため)

ClippedReLU と完全に別物。`(x^2)` 項のおかげで微小値領域の勾配が CReLU より大きく、表現力が上がる傾向がある (= Stockfish が SCReLU に置き換えた経緯)。

## やねうら王での導入時点

### 2018-05-15: 那須さん PR #75 — NNUE 評価関数追加 (commit 7a310543)

`halfkp_256x2-32-32.h` および `k-p_256x2-32-32.h` が追加された時点のネットワーク定義:

```cpp
using HiddenLayer1 = ClippedReLU<AffineTransform<InputLayer, 32>>;
using HiddenLayer2 = ClippedReLU<AffineTransform<HiddenLayer1, 32>>;
using OutputLayer = AffineTransform<HiddenLayer2, 1>;
```

**この時点では ClippedReLU のみ**。SqrClippedReLU は存在しない。

### 2026-01-31: PR #311 — SFNNwoPSQT-1536 NNUE アーキテクチャの追加 (commit 61d757e1)

`sfnnwop-1536.h` が追加。この PR で SqrClippedReLU が初めて導入される。`halfkp_256x2-32-32.h` 等の既存 architecture は **修正されず ClippedReLU のまま**。

## 設計上の含意

| 状況 | 活性化関数 |
|---|---|
| `halfkp_256x2-32-32` / `kp_256x2-32-32` を学習する | ClippedReLU 一択 (engine が CReLU で推論するので学習時も CReLU でないと一致しない) |
| 1024x2 / 512x2 系の HalfKP variant を後で追加する | architecture file (`halfkp_1024x2-8-32.h` 等) を読んで活性化を確認すること。ClippedReLU のまま増やしただけのものと、SCReLU 化されたものが混在し得る |
| `SFNNwoPSQT-1536` を学習する | architecture file の各層を見て CReLU と SCReLU を **層ごと正しく使い分ける** |

## BulletOu 側の実装契約

- `bullet_lib::value::nnue_save::Activation` enum: `Crelu` / `Screlu` を持つ
- `l1_bias_scale(activation, pairwise, qa, qb)` がスケールを返す (CReLU/SCReLU で値が違う、特に SCReLU は qa=255 でも 127 にスケールダウン)
- `examples/bulletou.rs` の `run_halfkp` / `run_kp` は network builder で `.crelu()` を呼ぶ、`save_format` で `qa=127` を使う

**間違って `.screlu()` で halfkp を学習すると、生成された `nn.bin` を engine が load しても CReLU で推論されるため、学習時と推論時の活性化が乖離して大きく弱くなる**。

(本セッションでも私が最初に SCReLU で実装してしまい、yaneurao さんに指摘されて修正した経緯あり。commit `5bee81c` 参照。)

## 出典

- やねうら王 commit `7a310543` (2018-05-15): "Add NNUE evaluation functions (#75)"
- やねうら王 commit `61d757e1` (2026-01-31): "SFNNwoPSQT-1536 NNUE アーキテクチャの追加 (#311)"
- `source/eval/nnue/layers/clipped_relu.h` および `sqr_clipped_relu.h`
- `source/eval/nnue/architectures/halfkp_256x2-32-32.h` / `kp_256x2-32-32.h` / `sfnnwop-1536.h`
