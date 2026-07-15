# NNUE HalfKP 学習

<a href="../../en/shogi/halfkp.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

[リファレンス目次へ戻る](../README.md)

`--eval-type NNUE_HALFKP` は、やねうら王が長年採用している古典的な HalfKP NNUE を学習する。dual-perspective HalfKP feature transformer + 全層 ClippedReLU の 4 層構成。

活性化関数の歴史的経緯 (なぜ SCReLU ではなく ClippedReLU か) は [`spec/05-activation-history.md`](../../spec/05-activation-history.md) を参照。

## アーキテクチャ

L1 / L2 / L3 サイズは `--arch NNUE_halfkp_<L1>x2_<L2>_<L3>` で指定する (`L1` は 32 の倍数)。やねうら王が実エンジンとして配布している共通サイズ: `256x2-32-32` (default)、`384x2-8-96`、`512x2-8-64`、`768x2-16-64`、`1024x2-8-32`、`1024x2-8-64`。CLI では古い短縮形 `256x2-32-32` ではなく `NNUE_halfkp_256x2_32_32` のように書く。詳細は [§4.3](../tutorial/4-train.md#43---arch-を指定する) 参照。以下はデフォルト構成の図:

```
HalfKP 疎入力 (125,388 次元 × 自他 2 perspective)
        │
        │  L0 affine + ClippedReLU       ← 両 perspective で重み共有
        ▼
   accumulator (256 次元 × 2 perspective = 連結して 512 次元)
        │
        │  L1 affine (512 → 32) + ClippedReLU
        ▼
        │  L2 affine (32 → 32) + ClippedReLU
        ▼
        │  Out affine (32 → 1)
        ▼
      eval (centipawn ベースのスカラー)
```

## 実際の使い方

### 必要なもの

- BulletOu をビルド済み (`cargo build --release --features device-cuda --example bulletou`)
- 学習データ (`.hcpe` / `.hcpe3` / `.pack` / `.psv` のいずれか)

### コマンド

```bash
./target/release/examples/bulletou \
    --eval-type NNUE_HALFKP \
    --teacher /path/to/train.hcpe \
    --output checkpoints/my-halfkp
```

`--superbatches` も `--max-epochs` も省略しているので、教師データを 1 周 (dataloader が EOF を返すまで) で学習が終了する。複数 epoch 回したい場合は `--max-epochs 3` のように指定する。デフォルトの `step_gamma` は LR を継続し、`step` / `cos` を明示した場合は epoch 境界で warm restart する。

### 保存レイアウト

```
checkpoints/my-halfkp/
├── summary-learn.log                  ← トップレベルの通算ログ (全 run / resume を連結)
├── 0001/
│   ├── nn.bin                         ← やねうら王 / Stockfish (nnue-pytorch) 互換の NNUE バイナリ
│   ├── state.bin                      ← resume 用の重み + Adam moments
│   └── learn.log                      ← この save 時点の学習ログ snapshot
├── 0002/
│   ├── ...
├── ...
└── 000N/                              ← 最新 (= 最後に保存された) save
    ├── nn.bin
    ├── state.bin
    └── learn.log
```

最新の `000N/nn.bin` をやねうら王の HalfKP エンジンの eval ファイルとして指定すれば対局可能 (`state.bin` は engine からは無視される)。

### `nn.bin` フォーマット

中身は nnue-pytorch / Stockfish のバイナリ形式 (nnue-pytorch の `serialize.py` の出力と byte 単位で同一)。layout:

- ヘッダー: `NNUE_VERSION` = `0x7AF32F16` (u32 LE)、`network_hash` (u32 LE)、`desc_len` (u32 LE)、`description` (UTF-8 bytes)
- Feature Transformer レイヤーハッシュ (u32 LE)
- L0 biases (i16 × L1)
- L0 weights (i16 × INPUT × L1)
- Network レイヤーハッシュ (u32 LE)
- L1: biases (i32 × L2)、weights (i8 × L2 × pad32(L1×2), row-major)
- L2: biases (i32 × L3)、weights (i8 × L3 × pad32(L2), row-major)
- Output: biases (i32 × 1)、weights (i8 × 1 × pad32(L3), row-major)

`pad32(n) = ceil(n/32) * 32` で各層の入力次元を SIMD 用に 32 バイトアライン。量子化: L0 は `qa = 127` (ClippedReLU の出力レンジ 0..127)、L1-Out は `qb = 64` で i8 重み。

### 主要 CLI フラグ

| フラグ | 意味 | デフォルト |
|---|---|---|
| `--eval-type` | `NNUE_HALFKP` | (必須) |
| `--arch` | `NNUE_halfkp_256x2_32_32`<br>`NNUE_halfkp_384x2_8_96`<br>`NNUE_halfkp_512x2_8_64`<br>`NNUE_halfkp_768x2_16_64`<br>`NNUE_halfkp_1024x2_8_32`<br>`NNUE_halfkp_1024x2_8_64` | `NNUE_halfkp_256x2_32_32` |
| `--teacher` | 教師ファイル (`.hcpe` / `.hcpe3` / `.pack` / `.psv`)、またはそれらが入ったディレクトリ、カンマ区切りで併用可 | (必須) |
| `--output` | チェックポイント親ディレクトリ | `checkpoints/<eval-type>-<arch>` (例: `checkpoints/NNUE_HALFKP-NNUE_halfkp_256x2_32_32`) |
| `--max-epochs` | epoch を最大何回実行するか。省略時は `step` / `cos` では 1、`plateau` では final loss の改善が止まるまで | 省略 |
| `--superbatches` | epoch あたりの superbatch 数の上限 | 上限なし |
| `--positions-per-superbatch` | superbatch あたりの目標局面数。実効値は `batch-size` の倍数へ切り捨て | 100000000 |
| `--save-rate` | N superbatch ごとに save | 1 |
| `--lr` / `--lr-schedule` / `--lr-min` | LR スケジューラ (`step_gamma` = bullet-shogi 互換 StepLR、`step` = geometric、`cos` = cosine、`plateau` = validation loss が改善しないときだけ減衰、詳細は [§6.1](../tutorial/6-tune.md#61-学習スケジュール)) | 0.001 / `step_gamma` / 0.00001 |
| `--lambda` | 教師 eval と対局結果 (WDL = Win/Draw/Loss) のブレンド比 (やねうら王内蔵学習器の `lambda` と同じ慣例): `λ × 教師eval + (1−λ) × 対局結果`。`λ=1.0` で純 eval、`λ=0.0` で純 WDL | 1.0 |

loss は `sigmoid(eval).squared_error(target)` に固定。活性化関数は ClippedReLU に固定 (2018 年オリジナル準拠)。必要になったら CLI フラグ化する余地はある。
