# Training a progress classifier from complete games

<a href="../../ja/advanced/progress-training.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-Japanese-2563EB?style=flat-square"></a>

SFNN architectures containing `progress4`, `progress8`, or another `progressN` token use `progress.bin` to divide positions into opening, middlegame, and endgame-like buckets. This page explains how to train only that classifier from complete games, independently of the evaluation-network loss.

## Training target

Suppose one game contains `L` evaluable positions. The target for position index `i` is:

```text
target = i / (L - 1)
```

- first position: `0`
- last evaluable position: `1`
- positions in between: their relative location in the game

A `.pack` file has no evaluable board after its terminal marker, so the position before the final stored move is assigned `1`. The learned weights are exported as bias-free, headerless f64 `progress.bin` parameters and converted to an integer in `0..255` at runtime.

## File format and inference

The format matches tatara / [YaneuraOu PR #326](https://github.com/yaneurao/YaneuraOu/pull/326): **headerless `f64 little-endian[81][1548]`**, 125,388 weights, exactly **1,003,104 bytes**. There is no bias, hash or bucket count. The index is `king_square * 1548 + BonaPiece`; active non-king features from both king perspectives contribute to the sum.

The separate `progress-train` command learns:

```text
z = Σ w[active_KP_feature]
prediction = sigmoid(z)
loss = (prediction - i / (L - 1))²
```

NNUE training, counting and validation use `round(w * 65536)` (clamped to i32). Fixed sigmoid thresholds in the source convert the integer sum to `0..255`; thresholds are not stored in the file.

`nn.bin` does **not** embed progress data. Its layout is `NNUE header → FeatureTransformer → stack networks`, without a progress-specific hash. Checkpoint saves also write a sibling `progress.bin`, containing the effective integer weights divided by 65536 as exact f64 values, so reloading preserves bucket assignments.

Deploy both files with an engine supporting external progress files and the matching architecture. The old embedded-progress engine format is not supported. BulletOu's quantized diagnostics load `progress.bin` from the same directory as `nn.bin` for progress architectures.

### Migrating the current checkpoint

Normal file loading accepts only the new format. A one-time converter is available:

```powershell
.\convert_progress_format.ps1 `
  -InputPath C:\path\to\progress.bin `
  -OutputPath C:\path\to\progress.bin
```

An in-place conversion backs up the source as `.q16-with-bias.bak`. Instead of discarding the old bias, it adds `bias_q16 / 4` to rook-related weights: two rooks (including dragons and held rooks) contribute four KP terms across both perspectives. Integer sums remain exact when the bias is divisible by four. **This guarantee does not apply to rook-odds games.** Fresh `progress-train` runs do not use this migration or any fixed offset.

To continue the current older `state.bin`, its bias-prefixed progress record is migrated the same way. Value-network weights, their optimizer state and the dataloader cursor are preserved. Use the converted `--sfnn-progress-bin` and normal `--resume`. Newly saved progress records contain weights only. `export-progress-bin` now accepts `--state-bin` only; `nn.bin` no longer contains a classifier to extract.

## Why `.pack` is required

The target requires both ends of each game. Use a YaneuraOu `.pack` file because it preserves complete game boundaries.

PSV, HCPE, and fixed-record `.bin` files contain independent positions. Once they are shuffled, the relative location of a position inside its original game cannot be reconstructed, so they cannot supply this target.

Numbers embedded in the filename are not interpreted as game counts. The command scans the file and reports the actual game and position counts.

## Command

```powershell
.\target\release\examples\bulletou.exe progress-train `
  --teacher C:\path\to\games.pack `
  --output C:\path\to\progress.bin `
  --epochs 5 `
  --batch-size 4096 `
  --lr 0.0002
```

| Option | Meaning |
|---|---|
| `--teacher` | One `.pack` file containing complete games |
| `--output` | Destination headerless f64 `progress.bin` |
| `--epochs` | Number of full training passes over the file |
| `--batch-size` | Positions averaged into one Adam update |
| `--lr` | Adam learning rate; no hidden multiplier is applied |
| `--validation-game-stride` | Hold out every Nth game; default 20, or 0 to disable validation |
| `--max-games` | Maximum games to scan; 0 means the entire file |
| `--overwrite` | Replace an existing output file |

Each epoch reports MSE, MAE, mean predictions for the first and last positions, and the resulting `progress4` / `progress8` bucket distributions. The epoch with the lowest validation MSE is exported.

## Sharing one classifier between progress bucket counts

`progress.bin` produces a bucket-count-independent scalar in `0..255`. An architecture maps it to a bucket with:

```text
bucket = min(progress_0_255 * bucket_count / 256,
             bucket_count - 1)
```

One `progress.bin` can therefore be shared by `progress4`, `progress8`, `progress16`, and other `progressN` architectures. `count.bin`, however, stores occurrence counts for one concrete architecture and must be regenerated when the bucket layout changes.

## Using it for counts and SFNN training

Create the architecture-specific count file with the trained classifier:

```powershell
.\target\release\examples\bulletou.exe bucket-count `
  --teacher D:\teacher `
  --arch SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4 `
  --progress-bin C:\path\to\progress.bin `
  --output D:\counts\count.bin
```

Load and freeze the same classifier during SFNN training:

```powershell
--sfnn-progress-bin C:\path\to\progress.bin
```

Normal SFNN training always keeps this classifier fixed. Evaluation loss never updates it, so progress-bucket assignment stays aligned with count collection and validation caches can be reused. To retrain the classifier, run `progress-train` separately.

For HalfKA2 / KA2 training, the CPU preparation threads calculate the progress bucket from the already decoded board. Each batch reaches the GPU with its buckets finalized, without storing per-position progress feature indices or recomputing buckets in the training consumer. Both normal training and worker mode use this automatically; no extra option or VRAM is needed.
