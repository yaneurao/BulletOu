# NNUE HalfKP 学習

<a href="../../en/shogi/halfkp.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

[リファレンス目次へ戻る](../README.md)

`--arch NNUE_halfkp_256x2_32_32` は、やねうら王が長年採用している古典的な HalfKP NNUE を学習する。dual-perspective HalfKP feature transformer + 全層 ClippedReLU の 4 層構成。

活性化関数の歴史的経緯 (なぜ SCReLU ではなく ClippedReLU か) は [`spec/05-activation-history.md`](../../spec/05-activation-history.md) を参照。

## アーキテクチャ

L1 / L2 / L3 サイズは `--arch NNUE_halfkp_<L1>x2_<L2>_<L3>` で指定する (`L1` は 32 の倍数)。やねうら王が実エンジンとして配布している共通サイズ: `256x2-32-32` (default)、`384x2-8-96`、`512x2-8-64`、`768x2-16-64`、`1024x2-8-32`、`1024x2-8-64`。CLI では `NNUE_halfkp_256x2_32_32` のように書く。基本のコマンド形は [チュートリアル: 学習を走らせる](../tutorial/3-train.md) を参照。以下はデフォルト構成の図:

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

- BulletOu をビルド済み (`cargo build --release --features cuda-cpp-backend --example bulletou`)
- 学習データ (`.hcpe` / `.hcpe3` / `.pack` / `.psv` / `.bin` のいずれか)

### コマンド

```bash
./target/release/examples/bulletou \
    --arch NNUE_halfkp_256x2_32_32 \
    --teacher /path/to/train.hcpe \
    --output checkpoints/my-halfkp
```

`--superbatches` も `--max-epochs` も省略しているので、教師データを 1 周 (dataloader が EOF を返すまで) で学習が終了する。複数 epoch 回したい場合は `--superbatches` で epoch 長を決めたうえで `--max-epochs 3` のように指定する。`step` / `geometric` / `cos` は epoch 境界で `--lr` に戻る。

### 保存レイアウト

```
checkpoints/my-halfkp/
├── summary-learn.log                  ← トップレベルの通算ログ (全 run / resume を連結)
├── 0001/
│   ├── nn.bin                         ← やねうら王 / Stockfish (nnue-pytorch) 互換の NNUE バイナリ
│   ├── state.bin                      ← resume 用の重み + Ranger optimizer state
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
| `--arch` | `NNUE_halfkp_256x2_32_32`<br>`NNUE_halfkp_384x2_8_96`<br>`NNUE_halfkp_512x2_8_64`<br>`NNUE_halfkp_768x2_16_64`<br>`NNUE_halfkp_1024x2_8_32`<br>`NNUE_halfkp_1024x2_8_64` | (required; target `NNUE_HALFKP` is inferred) |
| `--teacher` | 教師ファイル (`.hcpe` / `.hcpe3` / `.pack` / `.psv` / `.bin`)、またはそれらが入ったディレクトリ、カンマ区切りで併用可 | (必須) |
| `--output` | チェックポイント親ディレクトリ | `checkpoints/<target>-<arch>` (例: `checkpoints/NNUE_HALFKP-NNUE_halfkp_256x2_32_32`) |
| `--max-epochs` | epoch を最大何回実行するか。省略時は固定上限なし | 省略 |
| `--superbatches` | epoch あたりの superbatch 数の上限 | 上限なし |
| `--batch-size` | 1 gradient step あたりの局面数。省略時は tatara に合わせて 65536 | 65536 |
| `--positions-per-superbatch` | superbatch あたりの目標局面数。実効値は `batch-size` の倍数へ切り捨て | 100000000 |
| `--save-rate` | N superbatch ごとに save。デフォルトでは epoch 末尾も save | 20 |
| `--save-epoch-end` / `--no-save-epoch-end` | epoch 末尾の暗黙 save を有効/無効にする | on |
| `--lr` / `--lr-schedule` / `--lr-min` | LR スケジューラ (`step` = tatara/bullet-shogi 互換 StepLR、`geometric` = geometric、`cos` = cosine、`plateau` = validation loss が改善しないときだけ減衰。詳細は [応用編: 学習設定を調整する](../advanced/tuning.md)) | 0.000875 / `step` / 0.00001 |
| `--lambda` | 教師 eval と対局結果 (WDL = Win/Draw/Loss) のブレンド比 (やねうら王内蔵学習器の `lambda` と同じ慣例): `λ × 教師eval + (1−λ) × 対局結果`。`λ=1.0` で純 eval、`λ=0.0` で純 WDL | 1.0 |
| `--scale` | sigmoid loss の target で使う eval-to-score sigmoid scale。省略時は固定値 290 | 省略 |

loss は `sigmoid(eval).squared_error(target)` に固定。活性化関数は ClippedReLU に固定 (2018 年オリジナル準拠)。必要になったら CLI フラグ化する余地はある。
