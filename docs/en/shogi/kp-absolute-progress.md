# Game-Progress Estimation Using KP-Absolute Features

<a href="../../ja/shogi/kp-absolute-progress.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

This document explains the concept, specification, and usage notes of **KP-Absolute** (KP絶対) based logistic regression, one of the game-progress estimation methods adopted by BulletOu (and its upstream `bullet-shogi`).

For the CLI specification and command examples of the training tools (`shogi_progress_kpabs_train` / `shogi_progress_kpabs_train_cuda`), see [`shogi_progress_kpabs_train.md`](shogi_progress_kpabs_train.md).

---

## 1. Background

This is the method adopted by YaneuraOu/tanuki- at WCSC27 (2017):
**"Logistic regression with KP-Absolute features"**, faithfully reproduced in BulletOu (via bullet-shogi).

> Progress estimation uses logistic regression, the same approach that Gekisashi and Gikou had used.
> KP-Absolute is used as the feature.

**References:**
- [yaneurao/YaneuraOu](https://github.com/yaneurao/YaneuraOu) `old_engines/eval/progress/progress.h` — definition of `Tanuki::Progress` class (which contains the `weights_[SQ_NB][Eval::fe_end]` member)
- [yaneurao/YaneuraOu](https://github.com/yaneurao/YaneuraOu) `old_engines/eval/progress/progress.cpp` — `Tanuki::Progress::Estimate`
- [nodchip/nnue-pytorch](https://github.com/nodchip/nnue-pytorch) `tanuki_progress.cpp` — `Tanuki::Progress::Estimate`

> Throughout this document, `yaneurao/YaneuraOu` (official YaneuraOu) and
> `nodchip/nnue-pytorch` (Tanuki team's shogi-oriented derivative of nnue-pytorch)
> refer to the repositories at the URLs above.

---

## 2. What is KP-Absolute?

### Concept

KP-Absolute treats the following combination as a single feature index:

```
KP-Absolute index = king square (sq_k) × fe_end + piece BonaPiece (bp)
```

- **K**: the king's square on the board (0..80, 81 squares)
- **P**: the BonaPiece index of a non-king piece (0..1547)

Each index has an associated real-valued weight in a table. Summing the weights of the active indices and applying sigmoid produces a progress score in `[0.0, 1.0]`.

### YaneuraOu implementation

```cpp
// yaneurao/YaneuraOu : old_engines/eval/progress/progress.cpp / Tanuki::Progress::Estimate
double Progress::Estimate(const Position& pos) {
    Square sq_bk = pos.king_square(BLACK);
    Square sq_wk = Inv(pos.king_square(WHITE));   // mirror the white king into black's perspective
    const auto& list0 = pos.eval_list()->piece_list_fb();  // BonaPiece list from black's perspective
    const auto& list1 = pos.eval_list()->piece_list_fw();  // BonaPiece list from white's perspective

    double sum = 0.0;
    for (int i = 0; i < PIECE_NO_KING; ++i) {   // 38 pieces (kings excluded)
        sum += weights_[sq_bk][list0[i]];         // black-king perspective
        sum += weights_[sq_wk][list1[i]];         // white-king perspective
    }
    return 1.0 / (1.0 + exp(-sum));   // sigmoid
}
```

**Reference:** `yaneurao/YaneuraOu`'s `old_engines/eval/progress/progress.cpp`.

### Bucket quantisation

The formula that quantises progress `p ∈ [0.0, 1.0]` into a LayerStack bucket index is:

```
bucket = clamp(floor(p * 8), 0, 7)
```

Key points:

- **Uses floor**, not ceiling. `p` is divided into 8 equal-width intervals: `[0, 1/8)` → 0, `[1/8, 2/8)` → 1, ..., `[7/8, 1)` → 7.
- **The output range is the 8 values `{0, 1, ..., 7}`**. The "8" in `progress8kpabs` refers to this 8-way split.
- **Edge case `p = 1.0`**: `floor(p * 8) = 8`, but `clamp(0, 7)` pushes it down to 7, keeping the bucket count at 8.
- The lower clamp at 0 is rarely triggered (sigmoid output is always in `[0, 1]`), but it serves as a safety net against numerical errors and invalid positions.

Implementation: the `OutputBuckets::bucket` method of `ShogiProgressKPAbs` in `crates/bullet_lib/src/game/outputs.rs`, namely `((p * 8.0).floor() as i32).clamp(0, 7) as u8`.

#### Relationship between LayerStack's 9 buckets and Progress8KPAbs's 8 buckets

The `ShogiLayerStackBucket9` enum has `BUCKETS = 9`, so the **total LayerStack capacity is 9 buckets**, but the `Progress8KPAbs` variant uses only **buckets 0..7 (8 of them)**, with index 8 unused. The bucket count of `progress8kpabs` (8) and the LayerStack bucket capacity (9) are intentionally different numbers.

---

## 3. BonaPiece Index Layout

BulletOu's BonaPiece definition conforms to YaneuraOu's and is **fully compatible**.

**Reference:** the module-level doc comment at the top of `crates/bullet_lib/src/shogi/bona_piece.rs`.

### Hand (in-hand) piece region (indices 1..89)

| Piece | Black (f_hand_*) | White (e_hand_*) | Count |
|---|---|---|---|
| Pawn | 1..19 | 20..38 | 18 each |
| Lance | 39..43 | 44..48 | 4 each |
| Knight | 49..53 | 54..58 | 4 each |
| Silver | 59..63 | 64..68 | 4 each |
| Gold | 69..73 | 74..78 | 4 each |
| Bishop | 79..81 | 82..84 | 2 each |
| Rook | 85..87 | 88..89 | 2 each |

`fe_hand_end = 90`

### On-board piece region (indices 90..1547)

81 squares per side for each piece type.

| Piece | Black (f_*) | White (e_*) |
|---|---|---|
| Pawn | 90..170 | 171..251 |
| Lance | 252..332 | 333..413 |
| Knight | 414..494 | 495..575 |
| Silver | 576..656 | 657..737 |
| Gold / Pro-Pawn / Pro-Lance / Pro-Knight / Pro-Silver | 738..818 | 819..899 |
| Bishop | 900..980 | 981..1061 |
| Horse (promoted Bishop) | 1062..1142 | 1143..1223 |
| Rook | 1224..1304 | 1305..1385 |
| Dragon (promoted Rook) | 1386..1466 | 1467..1547 |

`FE_OLD_END = fe_end = 1548`

> **Note:** Pro-Pawn, Pro-Lance, Pro-Knight, and Pro-Silver share the **same** index range as Gold.
> `PIECE_BASE[ProPawn][is_friend] == F_GOLD or E_GOLD`
>
> **Reference:** the `PIECE_BASE` table in `crates/bullet_lib/src/shogi/bona_piece.rs`.

### Square index (`Square::index`)

```
Square::index() = file * 9 + rank   (0-indexed)
  file: 0 = file 1, 8 = file 9
  rank: 0 = rank 1, 8 = rank 9
```

Mirroring to the white perspective:

```rust
// crates/bullet_lib/src/shogi/types.rs
pub const fn inverse(self) -> Self {
    Square(80 - self.0)   // 0 ↔ 80, 1 ↔ 79, ...
}
```

**Reference:** `Square::inverse` in `crates/bullet_lib/src/shogi/types.rs`.

---

## 4. KP-Absolute Index Formula

### Black perspective

```
black_index = sq_bk * FE_END + bp_black
```

- `sq_bk` = `board.black_king_sq.index()` — used as-is
- `bp_black` = `BonaPiece::from_piece_square(piece, sq, Color::Black).value()` 
  or `BonaPiece::from_hand_piece(Color::Black, owner, pt, count).value()`

### White perspective

```
white_index = sq_wk_inv * FE_END + bp_white
```

- `sq_wk_inv` = `board.white_king_sq.inverse().index()` — **inverse() is required**
- `bp_white` = `BonaPiece::from_piece_square(piece, sq, Color::White).value()`
  or `BonaPiece::from_hand_piece(Color::White, owner, pt, count).value()`

### Total parameter count

```
81 (squares) × 1548 (fe_end) = 125,388
```

Black and white perspectives **share the same weight table** (same as the YaneuraOu implementation).

**Reference:** `Tanuki::Progress::Estimate` in `nodchip/nnue-pytorch`'s `tanuki_progress.cpp`.

---

## 5. Coefficient File (`progress.bin`) Specification

### Binary layout

| Item | Value |
|---|---|
| Filename | `progress.bin` (arbitrary) |
| Size | `81 × 1548 × 8 = 1,003,104 bytes` (about 1 MB) |
| Encoding | Sequence of little-endian `f64` |
| Layout | `weights[sq_k][bp]` (row-major, `sq_k` outer) |

```
offset = (sq_k * 1548 + bp) * 8  bytes
```

**Reference:** `Tanuki::Progress::Load` / `Tanuki::Progress::Save` in `yaneurao/YaneuraOu`'s `old_engines/eval/progress/progress.cpp`.

> Inside the estimation loop, weights are converted to `f32` for speed (accumulating `f64` weights would be slower).
> The file itself remains `f64`.

### Training and saving in nodchip/nnue-pytorch

`nodchip/nnue-pytorch`'s `train_progress.py` outputs `progress.bin` in the same format.

```python
# nodchip/nnue-pytorch : progress_tools.py
def export_progress_weights(weights: torch.Tensor, path: str):
    # weights.shape = [KP_ABS_NUM_WEIGHTS] (f32 → save as f64)
    w64 = weights.double().numpy()
    w64.reshape(81, 1548).tofile(path)
```

**Reference:** `progress_tools.py` in `nodchip/nnue-pytorch`.

Teacher label generation:

```python
# nodchip/nnue-pytorch : progress_tools.py
def build_linear_targets(length: int) -> torch.Tensor:
    # Linearly maps (position index / total_moves) within a game to 0.0..1.0.
    return torch.linspace(0.0, 1.0, steps=length)
```

Loss function: `MSE`. Optimiser: `Adam` (β1=0.9, β2=0.999, lr=2e-4).

**Reference:** `train_progress.py` in `nodchip/nnue-pytorch`,
and `Tanuki::Progress::Learn` in `yaneurao/YaneuraOu`'s `old_engines/eval/progress/progress.cpp`.

---

## 6. Comparison of Progress Estimators in BulletOu

BulletOu (via bullet-shogi) contains multiple progress-estimator implementations, switchable for LayerStack bucket selection.

| Aspect | `ShogiProgressBucket8` (coeff_v1) | `ShogiProgressBucket8GikouLite` (coeff_v2) | `ShogiProgressKPAbs` |
|---|---|---|---|
| Features | Hand-crafted counts (on-board piece count, hand piece count, promoted count, king's rank) | v1 features + Chebyshev distance from king (3 levels) | KP-Absolute indices (enumerated dynamically) |
| Feature count | 6 | 34 | Dynamic (varies with king position and piece count) |
| Parameter count | A few (features + bias) | A few dozen (features + bias) | 125,388 |
| Standardisation | z-score | z-score | None |
| Coefficient file | JSON | JSON | Binary `progress.bin` (`f64`, 1,003,104 bytes) |
| Training script | External | External | `train_progress.py` in `nodchip/nnue-pytorch`, or BulletOu's `shogi_progress_kpabs_train` / `shogi_progress_kpabs_train_cuda` |
| Expressiveness | Low (local details are washed out) | Medium | High (memorises king position × every piece placement) |
| Origin | Hand-crafted, inspired by Gikou's progress features | Extension of the above | From YaneuraOu/tanuki- |

Implementations are concentrated in `crates/bullet_lib/src/game/outputs.rs`. The choice of which to use during LayerStack training/inference is selected via `BucketMode` in `examples/shogi_layerstack.rs`.

---

## 7. Implementation Notes

### 7-1. Skipping `BonaPiece::ZERO`

`BonaPiece::from_piece_square` returns `BonaPiece::ZERO` for kings and empty squares. To avoid accumulating into the weight table at `index = 0`, the KP-Absolute summation loop must check `bp != BonaPiece::ZERO`.

**Reference:** `BonaPiece::from_piece_square` in `crates/bullet_lib/src/shogi/bona_piece.rs` (returns `BonaPiece::ZERO` for kings and empty squares).

### 7-2. Inverting the white king

Always invert the white king's square before use:

```rust
// WRONG: board.white_king_sq.index()
// OK:    board.white_king_sq.inverse().index()
```

Corresponds to `Inv(pos.king_square(WHITE))` in YaneuraOu.

**Reference:** `Tanuki::Progress::Estimate` in `yaneurao/YaneuraOu`'s `old_engines/eval/progress/progress.cpp` (`Inv(pos.king_square(WHITE))`).

### 7-3. Enumerate hand pieces one by one

YaneuraOu's `piece_list` already expands each in-hand piece into its own BonaPiece. The BulletOu side must replicate this by calling `BonaPiece::from_hand_piece(perspective, owner, pt, count)` in a loop from 1..n for each piece kind.

### 7-4. Status of progress in YaneuraOu proper

YaneuraOu's `old_engines/eval/readme.txt` says:

```
progress/
    For training progress. Progress itself turned out somewhat lacklustre, so it ended up unused.
    Might revive it later.
```

**Reference:** `old_engines/eval/readme.txt` in `yaneurao/YaneuraOu`.

While integration into the YaneuraOu mainline was shelved, `nodchip/nnue-pytorch` keeps an independent implementation as `tanuki_progress.cpp`, and the training script is maintained.

---

## 8. Related Files

### BulletOu

| File | Contents |
|---|---|
| `crates/bullet_lib/src/shogi/bona_piece.rs` | BonaPiece definition and index calculation |
| `crates/bullet_lib/src/shogi/packed_sfen.rs` | PackedSfenValue / ShogiBoard definitions |
| `crates/bullet_lib/src/shogi/types.rs` | `Square::inverse`, `BOARD_PIECE_TYPES`, etc. |
| `crates/bullet_lib/src/game/outputs.rs` | All progress-estimator implementations (`ShogiProgressBucket8` / `ShogiProgressBucket8GikouLite` / `ShogiProgressKPAbs`) |
| `examples/shogi_layerstack.rs` | LayerStack training; impl switching via `BucketMode` |
| `examples/shogi_progress_kpabs_train.rs` | `progress.bin` trainer (CPU version) |
| `examples/shogi_progress_kpabs_train_cuda.rs` | `progress.bin` trainer (CUDA + parallel reader) |
| `docs/en/shogi/shogi_progress_kpabs_train.md` / `docs/ja/shogi/shogi_progress_kpabs_train.md` | CLI spec and command examples |

### yaneurao/YaneuraOu (<https://github.com/yaneurao/YaneuraOu>)

| File | Contents |
|---|---|
| `old_engines/eval/progress/progress.h` | Definition of `Tanuki::Progress` class and the `weights_[SQ_NB][Eval::fe_end]` member |
| `old_engines/eval/progress/progress.cpp` | `Estimate` / `Learn` / `Load` / `Save` implementations |
| `source/evaluate.h` | `BonaPiece` enum and `fe_end` definition |

### nodchip/nnue-pytorch (<https://github.com/nodchip/nnue-pytorch>)

A shogi-oriented derivative of [official-stockfish/nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch) by the Tanuki team.

| File | Contents |
|---|---|
| `tanuki_progress.cpp` | C++ progress estimator (for the data loader) |
| `tanuki_progress.h` | Header of the above |
| `train_progress.py` | Training script (PyTorch) |
| `progress_tools.py` | `build_linear_targets`, `export_progress_weights` |
| `progress_bucket_viz.py` | `progress_to_bucket` implementation |
