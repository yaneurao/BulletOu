# KP絶対を用いた進行度推定

[English](../../en/shogi/kp-absolute-progress.md) / **日本語**

bullet-shogi が採用している進行度推定方式の 1 つである **KP絶対（KP Absolute）**
ベースのロジスティック回帰について、概念・仕様・利用上の注意点を解説する。

学習ツール（`shogi_progress_kpabs_train` / `shogi_progress_kpabs_train_cuda`）の
CLI 仕様・コマンド例は [`shogi_progress_kpabs_train.md`](shogi_progress_kpabs_train.md) を参照。

---

## 1. 背景

YaneuraOu/tanuki- が WCSC27（2017）で採用した方式：
**「KP絶対を用いたロジスティック回帰」** を bullet-shogi も同方式で採用している。

> 進行度の推定には激指・技巧等が使用した、ロジスティック回帰を使用する。
> 特徴量には KP絶対を用いる。

**リファレンス元:**
- [yaneurao/YaneuraOu](https://github.com/yaneurao/YaneuraOu) `old_engines/eval/progress/progress.h` の `Tanuki::Progress` クラス定義（`weights_[SQ_NB][Eval::fe_end]` メンバを含む）
- [yaneurao/YaneuraOu](https://github.com/yaneurao/YaneuraOu) `old_engines/eval/progress/progress.cpp` の `Tanuki::Progress::Estimate`
- [nodchip/nnue-pytorch](https://github.com/nodchip/nnue-pytorch) `tanuki_progress.cpp` の `Tanuki::Progress::Estimate`

> 以降の本文では `yaneurao/YaneuraOu`（公式 YaneuraOu）と
> `nodchip/nnue-pytorch`（Tanuki チームの将棋向け nnue-pytorch 派生）の
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

各インデックスに対応した実数重みをテーブルとして持ち、該当する重みを合算して
sigmoid を適用することで進行度（0.0〜1.0）を得る。

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

### Bucket 量子化

進行度 `p ∈ [0.0, 1.0]` を LayerStack の bucket index へ量子化する式は次の通り:

```
bucket = clamp(floor(p * 8), 0, 7)
```

挙動の要点:

- **切り捨て (floor) を使う**（切り上げではない）。`p` を等幅 1/8 ごとに区切り、
  `[0, 1/8)` → 0, `[1/8, 2/8)` → 1, ..., `[7/8, 1)` → 7 と割り当てる
- **結果値域は `{0, 1, ..., 7}` の 8 値**。`progress8kpabs` の "8" はこの 8 値分割を指す
- **`p = 1.0` の境界ケース**は `floor(p * 8) = 8` となるが、`clamp(0, 7)` が
  上限 7 に押し戻すため bucket 数は 8 に保たれる
- 下限 0 のクランプは `sigmoid` 出力が常に `[0, 1]` に収まるため通常はトリガしないが、
  数値誤差・無効局面の保険として機能する

実装: `crates/bullet_lib/src/game/outputs.rs` の `ShogiProgressKPAbs` の
`OutputBuckets::bucket` メソッド（`((p * 8.0).floor() as i32).clamp(0, 7) as u8`）。

#### LayerStack の 9 buckets と Progress8KPAbs の 8 buckets の関係

`ShogiLayerStackBucket9` enum は `BUCKETS = 9` で **LayerStack 全体の容量は
9 bucket** だが、`Progress8KPAbs` variant が使うのはその内 **0〜7 の 8 bucket
だけ**で、index 8 は未使用。`progress8kpabs` の bucket 数（8）と LayerStack の
bucket 容量（9）は別の数値である点に注意。

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

## 5. 係数ファイル (progress.bin) の仕様

### バイナリレイアウト

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

> 推定ループ内では `f32` に変換して使用する（`f64` 重みのまま積和すると速度面で不利なため）。
> ファイルそのものは `f64` で保存される。

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

## 6. 現状の進行度推定実装の特性

bullet-shogi には複数の進行度推定実装が存在し、LayerStack の bucket 選択用に
切り替えて使用できる。

| 観点 | `ShogiProgressBucket8` (coeff_v1) | `ShogiProgressBucket8GikouLite` (coeff_v2) | `ShogiProgressKPAbs` |
|---|---|---|---|
| 特徴量 | 手作りカウント（盤上駒数・持ち駒数・成り駒数・玉の段） | v1 + 玉からの Chebyshev 距離 3 段階 | KP絶対インデックス（動的列挙） |
| 特徴量数 | 6 | 34 | 動的（玉位置 + 駒数で変動） |
| パラメータ数 | 数個（特徴量 + bias） | 数十個（特徴量 + bias） | 125,388 |
| 標準化 | z-score あり | z-score あり | なし |
| 係数ファイル | JSON | JSON | バイナリ `progress.bin`（`f64`、1,003,104 bytes） |
| 学習スクリプト | 別途用意 | 別途用意 | `nodchip/nnue-pytorch` の `train_progress.py`、または bullet-shogi の `shogi_progress_kpabs_train` / `shogi_progress_kpabs_train_cuda` |
| 表現力 | 低（局面の細部が消える） | 中 | 高（玉位置と各駒配置を記憶） |
| 由来 | 技巧（Gikou）の進行度特徴量を参考にした手作り | 同上の拡張 | YaneuraOu/tanuki- 由来 |

実装は `crates/bullet_lib/src/game/outputs.rs` に集約されている。LayerStack 学習・推論で
どの実装を使うかは `examples/shogi_layerstack.rs` の `BucketMode` で選択する。

---

## 7. 実装上の注意点

### 7-1. `BonaPiece::ZERO` のスキップ

`BonaPiece::from_piece_square` は玉（King）および空マスに対して `BonaPiece::ZERO`
を返す。重みテーブルの index=0 への加算を避けるため、KP絶対の積和ループでは
`bp != BonaPiece::ZERO` のチェックが必要。

**リファレンス元:** `crates/bullet_lib/src/shogi/bona_piece.rs` の `BonaPiece::from_piece_square`（玉および空マスで `BonaPiece::ZERO` を返す）

### 7-2. 後手玉の反転

後手玉位置は必ず `inverse()` してから使う。

```rust
// NG: board.white_king_sq.index()
// OK: board.white_king_sq.inverse().index()
```

YaneuraOu での対応: `Inv(pos.king_square(WHITE))`

**リファレンス元:** `yaneurao/YaneuraOu` の `old_engines/eval/progress/progress.cpp` の `Tanuki::Progress::Estimate`（`Inv(pos.king_square(WHITE))`）

### 7-3. 持ち駒は 1 枚ずつ列挙

YaneuraOu の `piece_list` は持ち駒も 1 枚ずつ別の BonaPiece として展開済み。
bullet-shogi 側は `BonaPiece::from_hand_piece(perspective, owner, pt, count)` を
1〜n 枚分ループで呼び出すことで同等の列挙を実現する必要がある。

### 7-4. YaneuraOu 本体での扱い

YaneuraOu の `old_engines/eval/readme.txt` には次の記載がある:

```
progress/
    進行度の学習用。進行度自体が微妙だったので結局活用せず。
    いずれ復活させるかも。
```

**リファレンス元:** `yaneurao/YaneuraOu` の `old_engines/eval/readme.txt`

YaneuraOu 本体への組み込みは見送られたが、`nodchip/nnue-pytorch` 側では
`tanuki_progress.cpp` として独立実装が存在し、学習スクリプトも整備されている。

---

## 8. 関連ファイル一覧

### bullet-shogi

| ファイル | 内容 |
|---------|------|
| `crates/bullet_lib/src/shogi/bona_piece.rs` | BonaPiece 定義・インデックス計算 |
| `crates/bullet_lib/src/shogi/packed_sfen.rs` | PackedSfenValue・ShogiBoard 定義 |
| `crates/bullet_lib/src/shogi/types.rs` | Square::inverse、BOARD_PIECE_TYPES 等 |
| `crates/bullet_lib/src/game/outputs.rs` | 進行度推定の各種実装（`ShogiProgressBucket8` / `ShogiProgressBucket8GikouLite` / `ShogiProgressKPAbs`） |
| `examples/shogi_layerstack.rs` | LayerStack 学習・`BucketMode` による実装切り替え |
| `examples/shogi_progress_kpabs_train.rs` | progress.bin 学習ツール（CPU 版） |
| `examples/shogi_progress_kpabs_train_cuda.rs` | progress.bin 学習ツール（CUDA + reader 並列版） |
| `docs/shogi/shogi_progress_kpabs_train.md` | 学習ツールの CLI 仕様・コマンド例 |

### yaneurao/YaneuraOu (<https://github.com/yaneurao/YaneuraOu>)

| ファイル | 内容 |
|---------|------|
| `old_engines/eval/progress/progress.h` | `Tanuki::Progress` クラス定義・`weights_[SQ_NB][Eval::fe_end]` メンバ |
| `old_engines/eval/progress/progress.cpp` | `Estimate` / `Learn` / `Load` / `Save` 実装 |
| `source/evaluate.h` | `BonaPiece` enum・`fe_end` 定義 |

### nodchip/nnue-pytorch (<https://github.com/nodchip/nnue-pytorch>)

Tanuki チームによる将棋向け実装（[official-stockfish/nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch) 系統の派生）。

| ファイル | 内容 |
|---------|------|
| `tanuki_progress.cpp` | C++ 版進行度推定（データローダー用） |
| `tanuki_progress.h` | 同ヘッダ |
| `train_progress.py` | 学習スクリプト（PyTorch） |
| `progress_tools.py` | `build_linear_targets`・`export_progress_weights` |
| `progress_bucket_viz.py` | `progress_to_bucket` 実装 |
