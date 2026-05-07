# KP絶対を用いた進行度推定の実装ガイド

YaneuraOu/tanuki- と同方式の KP絶対（KP Absolute）ベースの進行度推定を
bullet-shogi の `OutputBuckets` として実装するためのリファレンスドキュメント。

学習ツール（`shogi_progress_kpabs_train` / `shogi_progress_kpabs_train_cuda`）の
CLI 仕様・コマンド例は [`shogi_progress_kpabs_train.md`](shogi_progress_kpabs_train.md) を参照。

---

## 1. 背景と目的

### 既存実装の課題

現在 bullet-shogi には 2 種類の進行度推定実装がある。

| 実装 | 特徴量数 | 特徴量の性質 |
|------|---------|------------|
| `ShogiProgressBucket8` (coeff_v1) | 6個 | 盤上駒数・持ち駒数・成り駒数・玉の段 |
| `ShogiProgressBucket8GikouLite` (coeff_v2) | 34個 | v1 + 玉からの Chebyshev 距離 3段階 |

どちらも「意味的に解釈可能な手作り特徴量」であり、技巧（Gikou）の進行度特徴量を参考にした設計。

**リファレンス元:** `crates/bullet_lib/src/game/outputs.rs`

### 目標

YaneuraOu/tanuki- が WCSC27（2017）で採用した方式：
**「KP絶対を用いたロジスティック回帰」** を実装する。

> 進行度の推定には激指・技巧等が使用した、ロジスティック回帰を使用する。
> 特徴量には KP絶対を用いる。

**リファレンス元:**
- [yaneurao/YaneuraOu](https://github.com/yaneurao/YaneuraOu) `old_engines/eval/progress/progress.h` の `Tanuki::Progress` クラス定義（`weights_[SQ_NB][Eval::fe_end]` メンバを含む）
- [yaneurao/YaneuraOu](https://github.com/yaneurao/YaneuraOu) `old_engines/eval/progress/progress.cpp` の `Tanuki::Progress::Estimate`
- [nodchip/nnue-pytorch](https://github.com/nodchip/nnue-pytorch) `tanuki_progress.cpp` の `Tanuki::Progress::Estimate`

> 以降の本文では `yaneurao/YaneuraOu`（公式 YaneuraOu）と
> `nodchip/nnue-pytorch`（Tanuki チームの将棋向け nnue-pytorch フォーク）の
> リポジトリを上記 URL のものとして扱う。

---

## 2. KP絶対とは

### 概念

KP絶対（KP Absolute）とは、以下の組み合わせを 1 個の特徴インデックスとして扱う手法。

```
KP絶対インデックス = 玉の位置 (sq_k) × fe_end + 駒の BonaPiece (bp)
```

- **K**：玉（King）の盤上の位置（0〜80、81升）
- **P**：玉以外の駒の BonaPiece インデックス（0〜1547）

各インデックスに対応した実数重みをテーブルとして持ち、該当する重みを合算して sigmoid を適用することで進行度（0.0〜1.0）を得る。

### YaneuraOu の実装

```cpp
// yaneurao/YaneuraOu : old_engines/eval/progress/progress.cpp / Tanuki::Progress::Estimate
double Progress::Estimate(const Position& pos) {
    Square sq_bk = pos.king_square(BLACK);
    Square sq_wk = Inv(pos.king_square(WHITE));   // 後手玉を先手視点に反転
    const auto& list0 = pos.eval_list()->piece_list_fb();  // 先手視点 BonaPiece リスト
    const auto& list1 = pos.eval_list()->piece_list_fw();  // 後手視点 BonaPiece リスト

    double sum = 0.0;
    for (int i = 0; i < PIECE_NO_KING; ++i) {   // 38駒（玉除く）
        sum += weights_[sq_bk][list0[i]];         // 先手玉視点
        sum += weights_[sq_wk][list1[i]];         // 後手玉視点
    }
    return 1.0 / (1.0 + exp(-sum));   // sigmoid
}
```

**リファレンス元:** `yaneurao/YaneuraOu` の `old_engines/eval/progress/progress.cpp`

---

## 3. BonaPiece インデックス体系

bullet-shogi の BonaPiece 定義は YaneuraOu に準拠しており、**完全に互換**。

**リファレンス元:** `crates/bullet_lib/src/shogi/bona_piece.rs` 冒頭のモジュール docコメント

### 手駒領域（インデックス 1〜89）

| 駒種 | 先手 (f_hand_*) | 後手 (e_hand_*) | 枚数 |
|------|----------------|----------------|------|
| 歩 | 1〜19 | 20〜38 | 各18枚 |
| 香 | 39〜43 | 44〜48 | 各4枚 |
| 桂 | 49〜53 | 54〜58 | 各4枚 |
| 銀 | 59〜63 | 64〜68 | 各4枚 |
| 金 | 69〜73 | 74〜78 | 各4枚 |
| 角 | 79〜81 | 82〜84 | 各2枚 |
| 飛 | 85〜87 | 88〜89 | 各2枚 |

`fe_hand_end = 90`

### 盤上駒領域（インデックス 90〜1547）

各駒種につき先手・後手それぞれ 81 マス分。

| 駒種 | 先手 (f_*) | 後手 (e_*) |
|------|-----------|-----------|
| 歩 | 90〜170 | 171〜251 |
| 香 | 252〜332 | 333〜413 |
| 桂 | 414〜494 | 495〜575 |
| 銀 | 576〜656 | 657〜737 |
| 金/成歩/成香/成桂/成銀 | 738〜818 | 819〜899 |
| 角 | 900〜980 | 981〜1061 |
| 馬 | 1062〜1142 | 1143〜1223 |
| 飛 | 1224〜1304 | 1305〜1385 |
| 龍 | 1386〜1466 | 1467〜1547 |

`FE_OLD_END = fe_end = 1548`

> **注意:** 成歩・成香・成桂・成銀は金と**同じ**インデックス範囲を使う。
> `PIECE_BASE[ProPawn][is_friend] == F_GOLD or E_GOLD`
>
> **リファレンス元:** `crates/bullet_lib/src/shogi/bona_piece.rs` の `PIECE_BASE` テーブル

### マスのインデックス（Square::index）

```
Square::index() = file * 9 + rank   (0-indexed)
  file: 0=1筋, 8=9筋
  rank: 0=1段, 8=9段
```

後手視点への反転:

```rust
// crates/bullet_lib/src/shogi/types.rs
pub const fn inverse(self) -> Self {
    Square(80 - self.0)   // 0 ↔ 80、1 ↔ 79、...
}
```

**リファレンス元:** `crates/bullet_lib/src/shogi/types.rs` の `Square::inverse`

---

## 4. KP絶対インデックスの計算式

### 先手視点（黒視点）

```
black_index = sq_bk * FE_END + bp_black
```

- `sq_bk` = `board.black_king_sq.index()` ←そのまま
- `bp_black` = `BonaPiece::from_piece_square(piece, sq, Color::Black).value()`
  または `BonaPiece::from_hand_piece(Color::Black, owner, pt, count).value()`

### 後手視点（白視点）

```
white_index = sq_wk_inv * FE_END + bp_white
```

- `sq_wk_inv` = `board.white_king_sq.inverse().index()` ←**反転が必要**
- `bp_white` = `BonaPiece::from_piece_square(piece, sq, Color::White).value()`
  または `BonaPiece::from_hand_piece(Color::White, owner, pt, count).value()`

### 総パラメータ数

```
81 (マス数) × 1548 (fe_end) = 125,388
```

先手視点・後手視点で**同じ重みテーブルを共有する**（YaneuraOu 実装と同じ）。

**リファレンス元:** `nodchip/nnue-pytorch` の `tanuki_progress.cpp` の `Tanuki::Progress::Estimate`

---

## 5. 実装設計

### 5-1. 重みテーブル構造体

`OutputBuckets` トレイトは `Copy + Default + 'static` を要求する。

```rust
// crates/bullet_lib/src/game/outputs.rs : OutputBuckets trait
pub trait OutputBuckets<T>: Send + Sync + Copy + Default + 'static { ... }
```

重みは 81 × 1548 = **125,388 要素**（約 500 KB）ある。
この重みを bucket struct 自身に持たせると `Copy` 制約と相性が悪く、
コピーコストも大きい。

#### 解決策: `OnceLock<Box<[f32]>>` に 1 プロセス 1 セットだけ保持する

実装では、重み本体は process-global な `OnceLock<Box<[f32]>>` に保持し、
`ShogiProgressKPAbs` 自体は **ゼロサイズ型**として `Copy + Default` を満たす。
これにより `Box::leak` は不要で、bucket 値のコピーも軽い。

```rust
pub const SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS: usize = 81 * FE_OLD_END;

static SHOGI_PROGRESS_KP_ABS_WEIGHTS: OnceLock<Box<[f32]>> = OnceLock::new();
static SHOGI_PROGRESS_KP_ABS_ZERO_WEIGHTS: [f32; SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS] =
    [0.0; SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS];

#[derive(Clone, Copy, Default)]
pub struct ShogiProgressKPAbs;
```

### 5-2. 進行度推定関数

```rust
impl ShogiProgressKPAbs {
    fn weights() -> &'static [f32] {
        SHOGI_PROGRESS_KP_ABS_WEIGHTS
            .get()
            .map_or(&SHOGI_PROGRESS_KP_ABS_ZERO_WEIGHTS, |weights| weights.as_ref())
    }

    /// 局面の進行度を 0.0..=1.0 で返す
    ///
    /// # アルゴリズム
    /// 先手玉視点と後手玉視点（盤反転）それぞれについて
    /// 玉以外の全駒の KP絶対インデックスの重みを合算し、sigmoid を適用。
    pub fn progress(&self, pos: &PackedSfenValue) -> f32 {
        let board = pos.decode();  // ShogiBoard
        if !board.black_king_sq.is_valid() || !board.white_king_sq.is_valid() {
            return 0.5;  // 無効局面では中立値
        }

        let weights = Self::weights();

        // 玉位置
        let sq_bk = board.black_king_sq.index();          // そのまま
        let sq_wk = board.white_king_sq.inverse().index(); // 後手玉は反転

        let mut sum = 0.0f32;

        // --- 盤上駒（玉以外）---
        for &pt in &BOARD_PIECE_TYPES {
            for color in [Color::Black, Color::White] {
                for sq in board.pieces(color, pt) {
                    let piece = Piece::new(color, pt);

                    // 先手視点 BonaPiece
                    let bp_b = BonaPiece::from_piece_square(piece, sq, Color::Black);
                    if bp_b != BonaPiece::ZERO {
                        sum += weights[sq_bk * FE_OLD_END + bp_b.value() as usize];
                    }

                    // 後手視点 BonaPiece
                    let bp_w = BonaPiece::from_piece_square(piece, sq, Color::White);
                    if bp_w != BonaPiece::ZERO {
                        sum += weights[sq_wk * FE_OLD_END + bp_w.value() as usize];
                    }
                }
            }
        }

        // --- 持ち駒 ---
        for owner in [Color::Black, Color::White] {
            let hand = if owner == Color::Black { board.black_hand } else { board.white_hand };
            for &pt in &HAND_PIECE_TYPES {
                let count = hand.count(pt);
                for c in 1..=count {
                    // 先手視点
                    let bp_b = BonaPiece::from_hand_piece(Color::Black, owner, pt, c);
                    if bp_b != BonaPiece::ZERO {
                        sum += weights[sq_bk * FE_OLD_END + bp_b.value() as usize];
                    }
                    // 後手視点
                    let bp_w = BonaPiece::from_hand_piece(Color::White, owner, pt, c);
                    if bp_w != BonaPiece::ZERO {
                        sum += weights[sq_wk * FE_OLD_END + bp_w.value() as usize];
                    }
                }
            }
        }

        // sigmoid
        1.0 / (1.0 + (-sum).exp())
    }
}
```

### 5-3. OutputBuckets 実装

`ShogiProgressKPAbs` 自体はゼロサイズなので、既存の Progress 系と同様に
直接 `OutputBuckets<PackedSfenValue>` を実装できる。

```rust
impl OutputBuckets<PackedSfenValue> for ShogiProgressKPAbs {
    const BUCKETS: usize = 8;

    fn bucket(&self, pos: &PackedSfenValue) -> u8 {
        let p = self.progress(pos);
        let raw = (p * 8.0).floor() as i32;
        raw.clamp(0, 7) as u8
    }
}
```

LayerStack の 9-bucket 実運用には、既存の `ShogiLayerStackBucket9` enum に
新 variant を追加して組み込む。

#### 5-3-1. `ShogiLayerStackBucket9` への variant 追加

```rust
// crates/bullet_lib/src/game/outputs.rs : ShogiLayerStackBucket9
#[derive(Clone, Copy)]
pub enum ShogiLayerStackBucket9 {
    KingRank9,
    Ply9([u16; 8]),
    Progress8(ShogiProgressBucket8),
    Progress8GikouLite(ShogiProgressBucket8GikouLite),
    // ↓ 追加
    Progress8KPAbs(ShogiProgressKPAbs),
}
```

`ShogiProgressKPAbs` が `Copy` を満たしていれば、
enum 全体も `Copy` を維持できる。

```rust
// impl OutputBuckets への dispatch 追加
impl OutputBuckets<PackedSfenValue> for ShogiLayerStackBucket9 {
    const BUCKETS: usize = 9;

    fn bucket(&self, pos: &PackedSfenValue) -> u8 {
        match self {
            Self::KingRank9        => ShogiKingRankBucket::<9>.bucket(pos),
            Self::Ply9(bounds)     => { /* 既存 */ }
            Self::Progress8(b)     => b.bucket(pos),             // 既存
            Self::Progress8GikouLite(b) => b.bucket(pos),        // 既存
            // ↓ 追加
            Self::Progress8KPAbs(b) => b.bucket(pos),
        }
    }
}
```

**リファレンス元:** `crates/bullet_lib/src/game/outputs.rs` の `ShogiLayerStackBucket9` enum と `OutputBuckets<PackedSfenValue>` impl

#### 5-3-2. CLI / `BucketMode` への追加

```rust
// examples/shogi_layerstack.rs : BucketMode
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum BucketMode {
    #[default]
    Kingrank9,
    Ply9,
    Progress8,
    #[value(name = "progress8gikou")]
    Progress8Gikou,
    // ↓ 追加
    #[value(name = "progress8kpabs")]
    Progress8KPAbs,
}
```

**変更が必要なのは `bucket_impl` の構築ブロックだけではない。**
`BucketMode` に variant を追加すると、以下 3 つの match も同時に網羅する必要がある
（どれか一つでも漏れるとコンパイルエラーになるか、`--progress-coeff` の検証で弾かれる）。

##### `resolved_ply_bounds()`

```rust
// 既存: BucketMode::Progress8 | BucketMode::Progress8Gikou => { ... Ok(None) }
// ↓ 追加
BucketMode::Progress8 | BucketMode::Progress8Gikou | BucketMode::Progress8KPAbs => {
    if self.ply_bounds.is_some() {
        Err("--ply-bounds can only be used with --bucket-mode ply9".to_string())
    } else {
        Ok(None)
    }
}
```

**リファレンス元:** `examples/shogi_layerstack.rs` の `resolved_ply_bounds`

##### `bucket_mode_name()`

```rust
// 既存
BucketMode::Progress8Gikou => "progress8gikou",
// ↓ 追加
BucketMode::Progress8KPAbs => "progress8kpabs",
```

**リファレンス元:** `examples/shogi_layerstack.rs` の `bucket_mode_name`

##### `load_progress_bucket()` と `LoadedProgressBucket`

まず `LoadedProgressBucket` enum に variant を追加:

```rust
enum LoadedProgressBucket {
    V1(ShogiProgressBucket8),
    Gikou(ShogiProgressBucket8GikouLite),
    KPAbs(ShogiProgressKPAbs),   // ↓ 追加
}
```

次に `load_progress_bucket()` に arm を追加:

```rust
BucketMode::Progress8KPAbs => {
    let path = self
        .progress_coeff
        .as_ref()
        .ok_or_else(|| "--bucket-mode progress8kpabs requires --progress-coeff".to_string())?;
    ShogiProgressKPAbs::load_from_bin(path).map(|v| Some(LoadedProgressBucket::KPAbs(v)))
}
// また既存の _ => { ... } アームの検証メッセージも更新が必要:
// "can only be used with --bucket-mode progress8/progress8gikou/progress8kpabs"
```

**リファレンス元:** `examples/shogi_layerstack.rs` の `LoadedProgressBucket` enum と `load_progress_bucket` メソッド

##### `bucket_impl` 構築ブロック

```rust
BucketMode::Progress8KPAbs => match progress_bucket {
    Some(LoadedProgressBucket::KPAbs(b)) => ShogiLayerStackBucket9::Progress8KPAbs(b),
    _ => panic!("progress coeff (progress8kpabs) must exist in progress8kpabs mode"),
},
```

**リファレンス元:** `examples/shogi_layerstack.rs` の main 内 `bucket_impl` 構築 (`match args.bucket_mode`)

### 5-4. 係数ファイルの読み込み

```rust
impl ShogiProgressKPAbs {
    /// yaneurao/YaneuraOu の Progress::Save() が出力する形式と同一の
    /// progress.bin を読み込む。
    ///
    /// ファイル形式: double[81][1548] のバイナリ（little-endian）。
    ///
    /// 重みは process-global な OnceLock に保持する。
    /// 1 プロセス中でロードできる KPAbs モデルは 1 つだけ。
    ///
    /// リファレンス: yaneurao/YaneuraOu / old_engines/eval/progress/progress.cpp / Tanuki::Progress::Save / Tanuki::Progress::Load
    pub fn load_from_bin(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let expected = SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS * std::mem::size_of::<f64>();
        if bytes.len() != expected {
            return Err(format!(
                "progress.bin サイズ不一致: got {} bytes, expected {}",
                bytes.len(), expected
            ));
        }

        let weights: Vec<f32> = bytes
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()) as f32)
            .collect();

        SHOGI_PROGRESS_KP_ABS_WEIGHTS
            .set(weights.into_boxed_slice())
            .map_err(|_| "KPAbs weights are already loaded in this process".to_string())?;

        Ok(Self)
    }
}
```

> **注意1:** `yaneurao/YaneuraOu` は `double`（64-bit float）でファイルに保存する。
> bullet-shogi の推定ループでは `f32` に変換して使用する。
>
> **注意2:** `OnceLock` により、1 プロセス中でロードできる KPAbs モデルは 1 つだけ。
> 別の `progress.bin` を使う場合はプロセスを再起動する。

---

## 6. 係数ファイルの仕様

### ファイル形式

| 項目 | 値 |
|------|-----|
| ファイル名 | `progress.bin`（任意） |
| バイト数 | `81 × 1548 × 8 = 1,003,104 bytes`（約 1 MB） |
| エンコーディング | little-endian `f64` の連続 |
| レイアウト | `weights[sq_k][bp]`（row-major、外側が sq_k） |

```
offset = (sq_k * 1548 + bp) * 8  bytes
```

**リファレンス元:** `yaneurao/YaneuraOu` の `old_engines/eval/progress/progress.cpp` の `Tanuki::Progress::Load` / `Tanuki::Progress::Save`

### nodchip/nnue-pytorch での学習と保存

`nodchip/nnue-pytorch` の `train_progress.py` は同形式の `progress.bin` を出力する。

```python
# nodchip/nnue-pytorch : progress_tools.py
def export_progress_weights(weights: torch.Tensor, path: str):
    # weights.shape = [KP_ABS_NUM_WEIGHTS] (f32 → f64 で保存)
    w64 = weights.double().numpy()
    w64.reshape(81, 1548).tofile(path)
```

**リファレンス元:** `nodchip/nnue-pytorch` の `progress_tools.py`

教師ラベルの生成:

```python
# nodchip/nnue-pytorch : progress_tools.py
def build_linear_targets(length: int) -> torch.Tensor:
    # 棋譜内の position index / total_moves を 0.0..1.0 に線形マッピング
    return torch.linspace(0.0, 1.0, steps=length)
```

損失関数: `MSE`、最適化: `Adam`（β1=0.9、β2=0.999、lr=2e-4）

**リファレンス元:** `nodchip/nnue-pytorch` の `train_progress.py`、
および `yaneurao/YaneuraOu` の `old_engines/eval/progress/progress.cpp` の `Tanuki::Progress::Learn`

---

## 7. 既存実装との差分まとめ

| 観点 | 既存 (coeff_v1/v2) | 新実装 (KP絶対) |
|------|-------------------|----------------|
| 特徴量 | 手作りカウント（6〜34個） | KP絶対インデックス（動的列挙） |
| パラメータ数 | 7〜35個 | 125,388個 |
| 標準化 | z-score あり | **なし** |
| 係数ファイル | JSON (`rshogi.progress_coeff.v1/v2`) | バイナリ `progress.bin`（f64） |
| 学習スクリプト | 別途用意 | `nodchip/nnue-pytorch` の `train_progress.py` が使用可能 |
| 表現力 | 低（局面の細部が消える） | 高（玉位置と各駒配置を記憶） |

---

## 8. 実装時の注意事項

### 8-1. `BonaPiece::ZERO` のスキップ

`from_piece_square` は玉（King）に対して `BonaPiece::ZERO` を返す。
また空マスも ZERO を返す。重みテーブルの index=0 への加算を避けるため、
`bp != BonaPiece::ZERO` のチェックが必要。

**リファレンス元:** `crates/bullet_lib/src/shogi/bona_piece.rs` の `BonaPiece::from_piece_square`（玉および空マスで `BonaPiece::ZERO` を返す）

### 8-2. 後手玉の反転

後手玉位置は必ず `inverse()` してから使う。

```rust
// NG: board.white_king_sq.index()
// OK: board.white_king_sq.inverse().index()
```

YaneuraOu での対応: `Inv(pos.king_square(WHITE))`

**リファレンス元:** `yaneurao/YaneuraOu` の `old_engines/eval/progress/progress.cpp` の `Tanuki::Progress::Estimate`（`Inv(pos.king_square(WHITE))`）

### 8-3. 持ち駒は1枚ずつ列挙

YaneuraOu の `piece_list` は持ち駒も 1 枚ずつ別の BonaPiece として展開済み。
bullet-shogi では `from_hand_piece(perspective, owner, pt, count)` を 1〜n 枚分
ループで呼び出すことで同等の列挙を実現する。

### 8-4. YaneuraOu の `readme.txt` の記載

```
progress/
    進行度の学習用。進行度自体が微妙だったので結局活用せず。
    いずれ復活させるかも。
```

**リファレンス元:** `yaneurao/YaneuraOu` の `old_engines/eval/readme.txt`

YaneuraOu 本体への組み込みは見送られたが、`nodchip/nnue-pytorch` 側では
`tanuki_progress.cpp` として独立実装が存在し、学習スクリプトも整備されている。

---

## 9. 関連ファイル一覧

### bullet-shogi

| ファイル | 内容 |
|---------|------|
| `crates/bullet_lib/src/shogi/bona_piece.rs` | BonaPiece 定義・インデックス計算 |
| `crates/bullet_lib/src/shogi/packed_sfen.rs` | PackedSfenValue・ShogiBoard 定義 |
| `crates/bullet_lib/src/shogi/types.rs` | Square::inverse、BOARD_PIECE_TYPES 等 |
| `crates/bullet_lib/src/game/outputs.rs` | 既存の進行度推定実装（coeff_v1/v2） |
| `examples/shogi_layerstack.rs` | 学習ループ・BucketMode 実装例 |
| `examples/shogi_progress_kpabs_train.rs` | progress.bin 学習ツール（CPU 版） |
| `examples/shogi_progress_kpabs_train_cuda.rs` | progress.bin 学習ツール（CUDA + reader 並列版） |
| `docs/shogi/shogi_progress_kpabs_train.md` | 学習ツールの CLI 仕様・コマンド例 |

### yaneurao/YaneuraOu (<https://github.com/yaneurao/YaneuraOu>)

| ファイル | 内容 |
|---------|------|
| `old_engines/eval/progress/progress.h` | Progress クラス定義・weights_[81][1548] |
| `old_engines/eval/progress/progress.cpp` | Estimate・Learn・Load/Save 実装 |
| `source/evaluate.h` | BonaPiece enum・fe_end 定義 |

### nodchip/nnue-pytorch (<https://github.com/nodchip/nnue-pytorch>)

Tanuki チームによる将棋向け実装（[official-stockfish/nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch) 系統の派生）。

| ファイル | 内容 |
|---------|------|
| `tanuki_progress.cpp` | C++ 版進行度推定（データローダー用） |
| `tanuki_progress.h` | 同ヘッダ |
| `train_progress.py` | 学習スクリプト（PyTorch） |
| `progress_tools.py` | `build_linear_targets`・`export_progress_weights` |
| `progress_bucket_viz.py` | `progress_to_bucket` 実装 |
