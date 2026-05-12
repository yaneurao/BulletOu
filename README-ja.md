<div align="center">

# BulletOu

<a href="README.md"><img alt="Read in English" src="https://img.shields.io/badge/README-English-DC2626?style=flat-square"></a>

</div>

将棋AIの評価関数パラメーターを学習するための、Rust 製ドメイン特化型 ML ライブラリ。やねうら王 ([YaneuraOu](https://github.com/yaneurao/YaneuraOu)) で用いる評価関数パラメーターの学習に使用することを想定している。

対応する評価関数:

- NNUE halfKP
- NNUE KP
- NNUE halfka1/halfka2
- SFNN + layerstack9(NNUEwoSQPT1536)
- KPPT
- KPP_KPPT

### 使い方

**はじめての方** は [チュートリアル (docs/ja/tutorial/)](docs/ja/tutorial/) から読むことを推奨。インストール、ビルド、最初の学習までを順を追って解説する。

教師データのフォーマット、出力フォーマットなどの仕様レベルの詳細は [docs/ja/0-contents.md](docs/ja/0-contents.md) を参照。

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


### 系譜・上流

- **オリジナル**: [jw1912/bullet](https://github.com/jw1912/bullet) — 汎用 NNUE トレーナー (チェス向け)
- **上流**: [SH11235/bullet-shogi](https://github.com/SH11235/bullet-shogi) — 将棋向け fork。`.pack` loader (やねうら王 `gensfen` のゲーム単位可変長ファイルを内部で `PackedSfenValue` 列に展開)、HalfKA / HalfKP / Threat / HandThreat 特徴量、KP 絶対進行度 bucket 付き Layer Stack を実装
- **本リポジトリ**: [yaneurao/BulletOu](https://github.com/yaneurao/BulletOu) — yaneurao によるやねうら王向け改造版


### ライセンス

MIT (上流から継承)。元のコピーライト表記は `LICENSE` に保持されている。

### サポート・フィードバック

- 不具合や要望は <https://github.com/yaneurao/BulletOu/issues> へ
- 上流側 (チェス NNUE) に関する一般的な議論は、[Engine Programming](https://discord.com/invite/F6W6mMsTGN) Discord の `#bullet` チャンネルで
