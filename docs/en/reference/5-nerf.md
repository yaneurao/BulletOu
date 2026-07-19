# 5. `nerf` Command

<a href="../../ja/reference/5-nerf.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

`bulletou nerf` is an experimental post-processing command that adds reproducible random perturbations to a trained evaluation file.

This is an auxiliary tool for intentionally weakening an already generated `nn.bin`; it is not part of the training pipeline itself. The current implementation supports only SFNN-style `nn.bin` files, but the command is not intended to be SFNN-specific forever.

## Supported Formats

The current implementation supports the SFNNwoPSQT-style `nn.bin` layout:

- LEB128-compressed Feature Transformer
- LayerStacks
- i8 weights in the downstream `fc0` / `fc1` / `fc2` layers

The standard NNUE layouts used by `NNUE_HALFKP` / `NNUE_KP` / `NNUE_KA2` / `NNUE_HALFKPE9` / `NNUE_HALFKPVM` are not supported yet.

## Example

Example for weakening an `SFNN-HalfKA2-1024-7-64` `nn.bin`:

```bash
cargo run -p bulletou_lib --release --example bulletou -- nerf \
  --input nn.bin \
  --output nn-nerf.bin \
  --arch SFNN_halfka2_1024_7_64_k3k3 \
  --layers fc2,fc1 \
  --count 1000 \
  --seed 1
```

## Options

| Option | Description |
|---|---|
| `--input` | Input `nn.bin` |
| `--output` | Output `nn.bin`; it must differ from `--input` |
| `--arch` | YaneuraOu architecture name without the `YANEURAOU_ENGINE_` prefix, for example `SFNN_halfka2_1024_7_64_k3k3` |
| `--layers` | Target layers: comma-separated `fc0` / `fc1` / `fc2` / `all` |
| `--count` | Number of random `+1` / `-1` mutation attempts. The same weight may be selected multiple times |
| `--seed` | RNG seed. The same input and seed produce the same output |

The default `--layers` value is `fc2,fc1`. The command does not modify the Feature Transformer, biases, hashes, or SIMD padding weights.

## Mutation

The command performs `--count` mutation attempts. Each attempt randomly selects one candidate weight and adds either `+1` or `-1` to that i8 value. The same weight may be selected multiple times, so `--count` may exceed the candidate count. Repeated selections can accumulate or cancel each other out. Values are clamped to the i8 range, so an already saturated `127` or `-128` weight may remain unchanged.

After running, the command prints the candidate count, mutation-attempt count, changed count, and saturated no-op count.
