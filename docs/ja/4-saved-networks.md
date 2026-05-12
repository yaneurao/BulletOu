# 学習済みネットワーク

[English](../en/4-saved-networks.md) / **日本語**

## チェックポイントのレイアウト

チェックポイントが `<out_dir>/<checkpoint_name>` ディレクトリに保存されると、その中には以下が含まれる:

- `raw.bin` — ネットワークの生の浮動小数点パラメータ (`f32`)
- `quantised.bin` — 量子化済みネットワーク。64 バイトの倍数になるようパディングされる
- `optimiser_state/` — optimizer の内部状態

量子化が失敗した (整数オーバーフロー等) 場合は、量子化済みネットワークは保存されないが、学習自体は影響を受けない。

## チェックポイントの読み込み

既存のチェックポイントを `trainer: Trainer` に読み込むには `trainer.load_from_checkpoint()` を使う。
重みだけを読み込むには `trainer.load_weights_from_file(<checkpoint_path>/optimiser_state/weights.bin)` を使う。

## `SavedFormat` のレイアウト

`f32`, `i16` などのプリミティブ型は常に **little-endian** で書き出される (現代のほぼ全てのハードウェアでの標準)。

各重みには形状 `M × N` が紐づいており、ファイルには **column-major (列優先)** で書き出される。

つまり、次の 2 × 3 行列:

```
[1, 2, 3]
[4, 5, 6]
```

は `[1, 4, 2, 5, 3, 6]` の順で書き出される。

`affine` 層

```rust
let affine = builder.new_affine("affine", input_size, output_size);
```

について、weight の形状は `output_size × input_size`、bias の形状は `output_size × 1` であることに注意。

この層を連続して、`i16` に factor `256` で量子化して保存するには、以下の `SavedFormat` エントリを追加する:

```rust
SavedFormat::id("affinew").quantise::<i16>(256),
SavedFormat::id("affineb").quantise::<i16>(256),
```

column-major ではなく row-major で保存したい場合 (例: 推論性能の都合) は、weight を転置する必要がある:

```rust
SavedFormat::id("affinew").transpose().quantise::<i16>(256)
```

`.quantise::<T>(Q)` のデフォルト動作は `quantised_value = truncate(float_value * Q)`。これだと望ましくないことが多いので、`.round()` を加えて `quantised_value = round(float_value * Q)` に変えることができる:

```rust
SavedFormat::id("affinew").round().quantise::<i16>(256),
```

任意の変換を `SavedFormat::transform` のチェーンで重ね掛けできる。これの例は [input buckets の example](https://github.com/yaneurao/BulletOu/blob/shogi-support/examples/progression/4_multi_layer.rs#L47) で見られる (input factoriser のマージに使われている)。

`.round` と `.quantise::<T>` の **配置順は関係ない**。これらは常にファイル書き出しの直前で適用される。それ以外の変換は指定された順序で適用される (`.transpose` は内部的に `.transform` を使っているだけ)。
