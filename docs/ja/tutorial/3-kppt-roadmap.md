# 3. KPPT / KPP_KPPT ロードマップ

<a href="../../en/tutorial/3-kppt-roadmap.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

> **状況: 未実装**。このページは BulletOu における KPPT / KPP_KPPT 対応の設計予定をまとめたもの。現状のコードは NNUE のみ学習する。進捗はリポジトリの issue / PR で追う。

## なぜ KPPT / KPP_KPPT に対応するのか

やねうら王には NNUE 以前から旧来の評価関数系列がある:

- **KK** — 玉 vs 玉のみ
- **KKP** — 玉 × 玉 × 駒
- **KPP** — 玉 × 駒 × 駒 (Apery / Bonanza 流の元祖)
- **KPPT** — KPP + 手番テンソル T
- **KPP_KPPT** — KPPT の factorise 版 (KP + KPPT、パラメータ共有)

多くの強い将棋エンジンがこれらの上に構築されてきた。今でも価値があるシナリオ:

- 古典的評価関数を改良・再学習し、研究のベースラインにする
- BulletOu の GPU パイプラインで、歴史的に CPU 専用でとても遅かった学習を加速
- 同じ学習データで古典評価関数と NNUE を比較する

## NNUE との構造の違い

NNUE は「**疎特徴量変換器 + 小さい MLP**」 — それなりに標準的な NN 形状で、bullet の IR に自然に収まる。

KPPT は「**巨大な疎 embedding テーブルの和、隠れ層なし**」:

```
eval(pos) = KK[bk][wk] 
          + Σ_i KKP[bk][wk][p_i] 
          + Σ_{i<j} KPP[bk][p_i][p_j]
          + (手番項 T)
```

NN 的な「隠れ層」がない。巨大なルックアップテーブルの和。

最大のテーブル (`KPP`) はおおよそ **184 M パラメータ** (`81 × 1548 × 1548 / 2 × 2 チャンネル`) で、典型的 1〜10 M パラメータ NNUE とは桁違いのメモリ規模。

## BulletOu に追加が必要なもの

設計の議論はワークスペース側の spec [`docs/spec/bullet/shogi-port.md`](https://github.com/yaneurao/YaneuraOu) (本リポジトリ外、yaneurao の作業ディレクトリ側) を参照:

1. **タプル `SparseInputType` 群** (KK / KKP / KPP) — 各 `num_inputs()` / `max_active()` / `map_features()` を定義する。難所は KPP の BonaPiece ペア列挙
2. **複数入力対応の `ValueTrainerBuilder`** — 現状の `inputs(SingleInputType)` は単一の trait object を取る。KPPT には `inputs((Kk, Kkp, Kpp))` のような tuple input 拡張が要る。builder DSL に対する非自明な変更
3. **隠れ層なし構造** — embedding 出力を直接合算、FT のあとに MLP を入れない
4. **やねうら王形式の writer** — 現状 BulletOu の `SavedFormat` は rshogi 互換の量子化バイナリを書く。KPPT/KPP_KPPT は、やねうら王の `evaluate_kppt.cpp` が期待する `KK_synthesized.bin` / `KKP_synthesized.bin` / `KPPT_synthesized.bin` の 3 点セットを正確なレイアウトで書く必要がある
5. **学習スケジュールのチューニング** — KPPT は歴史的に ELMO 式 WDL 教師、強めの weight decay、小さい lr を使う。NNUE で動くハイパーパラメータがそのままは通用しない

## 作業規模の見積もり

ワークスペース側 spec (`docs/spec/bullet/shogi-port.md`) より:

| 構成要素 | 行数 (概算) |
|---|---|
| KK / KKP / KPP の SparseInputType 実装 | 600〜1,200 |
| ValueTrainerBuilder の tuple input 拡張 | 200〜500 (bullet コア。上流に PR するか fork でメンテするかの判断要) |
| やねうら王形式 weight writer | 300〜500 |
| factorise 用ヘルパー (KP_KPPT 用) | 200〜400 |
| スケジュール・正則化のチューニング | コードよりは実験コスト |
| **合計** | **~1,300〜2,600 行 + 実験時間** |

対する将棋 NNUE 対応 (上流 `bullet-shogi` で済んでいる分) は ~17,000 行なので、KPPT 対応は純粋な行数では小さい。ただし bullet コアの builder DSL という繊細な箇所に手を入れる必要がある。

## ユーザー目線のインターフェイス (予定)

KPPT 対応が入ったときに、エンドユーザーが触る形は NNUE の場合と類似:

```bash
cargo run --release --features cuda --example shogi_kppt_train -- \
  --data /data/shogi/train.pack \
  --output checkpoints/my-kppt-eval \
  --eval-format kppt \
  --epochs 1
```

出力ファイルはやねうら王の KPPT レイアウト (`KK_synthesized.bin`, `KKP_synthesized.bin`, `KPPT_synthesized.bin`) を checkpoint ディレクトリにそのまま書く。

`KPP_KPPT` はその上のスイッチ: `--eval-format kpp_kppt` (またはそれに類するフラグ) + 任意で factorise の制御。

## 今できること

KPPT 対応が実装されるまでは:

- **やねうら王本体の `learn` コマンド**で KPPT 系を学習する (オリジナル実装、CPU 専用で遅いが機能はする)
- **BulletOu は NNUE 用に使う** — GPU 加速が完全に効く領域
- KPPT 対応の実装に参加するなら、`crates/bullet_lib/src/shogi/bona_piece.rs` の BonaPiece ペア列挙コードが自然な出発点。数学的な背景は `docs/ja/shogi/kp-absolute-progress.md` (BonaPiece レイアウト) と `docs/spec/bullet/shogi-port.md` (全体設計) に詳述されている

---

KPPT 対応が landed したら、このページは [2. NNUE チュートリアル](2-nnue-tutorial.md) と並ぶ「3. KPPT チュートリアル」相当に書き直される予定。それまでは前向きな設計サマリとしてここに残しておく。
