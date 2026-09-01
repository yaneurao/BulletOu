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

A `.pack` file has no evaluable board after its terminal marker, so the position before the final stored move is assigned `1`. The learned logits are exported as q16 `progress.bin` parameters and converted to an integer in `0..255` at runtime.

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
| `--output` | Destination q16 `progress.bin` |
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
--sfnn-progress-bin C:\path\to\progress.bin `
--sfnn-freeze-progress
```

This keeps progress-bucket assignment aligned between count collection and SFNN training.

