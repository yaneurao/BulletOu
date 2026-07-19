# 5. `nerf` コマンド

<a href="../../en/reference/5-nerf.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

`bulletou nerf` は、学習済み評価関数ファイルに再現可能なランダム摂動を加える実験用の後処理コマンド。

このコマンドは学習器本体の機能ではなく、生成済み `nn.bin` を意図的に弱くしたい場合の補助ツールとして用意している。現時点では SFNN 系 `nn.bin` のみに対応しているが、コマンド自体は SFNN 専用という位置づけではない。

## 対応形式

現時点で対応しているのは、SFNNwoPSQT 系の `nn.bin` レイアウト:

- Feature Transformer が LEB128 圧縮されている
- LayerStacks を持つ
- 後段の `fc0` / `fc1` / `fc2` の i8 重みを持つ

通常の `NNUE_HALFKP` / `NNUE_KP` / `NNUE_KA2` / `NNUE_HALFKPE9` / `NNUE_HALFKPVM` の標準 NNUE レイアウトには未対応。

## コマンド例

`SFNN-HalfKA2-1024-7-64` の `nn.bin` を弱くする例:

```bash
cargo run -p bulletou_lib --release --example bulletou -- nerf \
  --input nn.bin \
  --output nn-nerf.bin \
  --arch SFNN_halfka2_1024_7_64_k3k3 \
  --layers fc2,fc1 \
  --count 1000 \
  --seed 1
```

## オプション

| オプション | 説明 |
|---|---|
| `--input` | 入力 `nn.bin` |
| `--output` | 出力 `nn.bin`。`--input` と同じパスは指定不可 |
| `--arch` | `YANEURAOU_ENGINE_` prefix を除いた、やねうら王 architecture 名。例: `SFNN_halfka2_1024_7_64_k3k3` |
| `--layers` | 変更対象。`fc0` / `fc1` / `fc2` / `all` をカンマ区切りで指定 |
| `--count` | ランダムな `+1` / `-1` 摂動の試行回数。同じ重みが複数回選ばれることがある |
| `--seed` | 乱数 seed。同じ入力・同じ seed なら同じ出力になる |

`--layers` の既定値は `fc2,fc1`。Feature Transformer、bias、hash、SIMD padding 部分の重みは変更しない。

## 変更内容

指定された候補重みからランダムに 1 個を選び、`+1` または `-1` を加える、という操作を `--count` 回行う。同じ重みが複数回選ばれることもあるので、`--count` は候補重み数を超えてよい。複数回選ばれた重みでは変化が累積したり、`+1` と `-1` が打ち消し合ったりする。値は i8 の範囲に clamp するため、すでに `127` または `-128` にある重みでは変化しない場合がある。

実行後は、候補数、摂動試行回数、実際に変化した回数、飽和により変化しなかった回数を表示する。
