<div align="center">

# BulletOu

</div>

[English](README.md) / **日本語**

**将棋エンジン用の NNUE 評価値ネットワーク**を学習するための、Rust 製ドメイン特化型 ML ライブラリ。やねうら王 ([YaneuraOu](https://github.com/yaneurao/YaneuraOu)) との組み合わせで使用することを想定している。

BulletOu は、jw1912 による [bullet](https://github.com/jw1912/bullet) (チェス向け汎用 NNUE 学習器) を SH11235 が将棋向けに fork した [bullet-shogi](https://github.com/SH11235/bullet-shogi) を、さらに yaneurao が fork したものである。本家 `bullet` は世界トップクラスのチェスエンジンの NNUE 学習に広く採用されている、GPU 上で最高クラスの性能を持つ Rust 製トレーナー。

### 系譜・上流

- **オリジナル**: [jw1912/bullet](https://github.com/jw1912/bullet) — 汎用 NNUE トレーナー (チェス向け)
- **上流**: [SH11235/bullet-shogi](https://github.com/SH11235/bullet-shogi) — 将棋向け fork。PackedSfenValue loader、HalfKA / HalfKP / Threat / HandThreat 特徴量、KP 絶対進行度 bucket 付き Layer Stack を実装
- **本リポジトリ**: [yaneurao/BulletOu](https://github.com/yaneurao/BulletOu) — yaneurao によるやねうら王向け改造版

上流の変更を取り込むには:

```bash
git remote add upstream https://github.com/SH11235/bullet-shogi.git
git fetch upstream
git merge upstream/shogi-support
```

### NNUE / 評価値ネットワーク学習での使い方

最初に [docs/ja/](docs/ja/) のドキュメントに目を通すことを推奨。ビルド方法、教師データの扱い、ネットワークの出力フォーマットなど主要な情報を扱う (※ 一部のドキュメントは現状英語版のみ。順次翻訳予定)。

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

- English: [docs/en/](docs/en/)
- 日本語: [docs/ja/](docs/ja/)

### ライセンス

MIT (上流から継承)。元のコピーライト表記は `LICENSE` に保持されている。

### サポート・フィードバック

- 不具合や要望は <https://github.com/yaneurao/BulletOu/issues> へ
- 上流側 (チェス NNUE) に関する一般的な議論は、[Engine Programming](https://discord.com/invite/F6W6mMsTGN) Discord の `#bullet` チャンネルで
