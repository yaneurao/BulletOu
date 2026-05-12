<div align="center">

# BulletOu

<a href="README.md"><img alt="Read in English" src="https://img.shields.io/badge/README-English-DC2626?style=flat-square"></a>

</div>

**将棋の評価関数**を学習するための、Rust 製ドメイン特化型 ML ライブラリ。やねうら王 ([YaneuraOu](https://github.com/yaneurao/YaneuraOu)) との組み合わせで使用することを想定している。

対応する評価関数:

- **NNUE 系の評価値ネットワーク** (現状動作する。HalfKP / HalfKA / KP 絶対進行度 bucket 付き Layer Stack)
- **KPPT / KPP_KPPT** などの **やねうら王の旧評価関数** (対応予定。詳細はチュートリアルのロードマップを参照)

BulletOu は、jw1912 による [bullet](https://github.com/jw1912/bullet) (チェス向け汎用学習器) を SH11235 が将棋向けに fork した [bullet-shogi](https://github.com/SH11235/bullet-shogi) を、さらに yaneurao が fork したもの。本家 `bullet` は世界トップクラスのチェスエンジンの NNUE 学習に広く採用されている、GPU 上で最高クラスの性能を持つ Rust 製トレーナー。

### 系譜・上流

- **オリジナル**: [jw1912/bullet](https://github.com/jw1912/bullet) — 汎用 NNUE トレーナー (チェス向け)
- **上流**: [SH11235/bullet-shogi](https://github.com/SH11235/bullet-shogi) — 将棋向け fork。PackedSfenValue loader、HalfKA / HalfKP / Threat / HandThreat 特徴量、KP 絶対進行度 bucket 付き Layer Stack を実装
- **本リポジトリ**: [yaneurao/BulletOu](https://github.com/yaneurao/BulletOu) — yaneurao によるやねうら王向け改造版

### 使い方

**はじめての方** は [チュートリアル (docs/ja/tutorial/)](docs/ja/tutorial/) から読むことを推奨。インストール、ビルド、最初の学習までを順を追って解説する。

教師データのフォーマット、出力フォーマットなどの仕様レベルの詳細は [docs/ja/0-contents.md](docs/ja/0-contents.md) を参照。

通常はリポジトリを clone し、[examples](/examples) のいずれかを自分の目的に合わせて編集して使う。上流からの pull で消えにくい独自 example を作りたい場合は、[`bullet_lib` の `Cargo.toml`](crates/bullet_lib/Cargo.toml) に example を登録する。

`bullet_lib` crate を他プロジェクトから import することもできる:

```toml
bullet = { git = "https://github.com/yaneurao/BulletOu", package = "bullet_lib" }
```

詳細な API ドキュメントは Rust の docstring に書かれている。ローカルで `cargo doc` を実行するとブラウザで参照できる。

### ビルド

```bash
# NVIDIA GPU (CUDA 12.x + cuDNN 9.x)
CUDA_PATH=/usr/local/cuda cargo build --release --features cuda

# AMD GPU (ROCm)
HIP_PATH=/opt/rocm cargo build --release --features rocm
```

CPU のみでの学習はサポートされていない (mock GPU ランタイムは型チェック用のスタブのみ)。

### ドキュメント

[docs/ja/](docs/ja/) を参照。

### ライセンス

MIT (上流から継承)。元のコピーライト表記は `LICENSE` に保持されている。

### サポート・フィードバック

- 不具合や要望は <https://github.com/yaneurao/BulletOu/issues> へ
- 上流側 (チェス NNUE) に関する一般的な議論は、[Engine Programming](https://discord.com/invite/F6W6mMsTGN) Discord の `#bullet` チャンネルで
