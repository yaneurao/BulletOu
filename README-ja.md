<p align="center">
  <img src="docs/images/bulletou-logo-mascot-s0.png" alt="BulletOu UltraFast Shogi AI ML" width="480px">
</p>

<div align="center">

  <h1>BulletOu</h1>

  将棋AIの評価関数パラメーターを学習するための、Rust 製ドメイン特化型 ML ライブラリ。やねうら王 ([YaneuraOu](https://github.com/yaneurao/YaneuraOu)) で用いる評価関数パラメーターの学習に使用することを想定している。

<a href="README.md"><img alt="Read in English" src="https://img.shields.io/badge/README-English-DC2626?style=flat-square"></a>

</div>


対応する評価関数:

- KPPT
- KPP_KKPT
- NNUE_HALFKP
- NNUE_KP
- NNUE_KA2
- NNUE_HALFKPE9
- NNUE_HALFKPVM
- SFNN_HALFKA1HM
- SFNN_HALFKA2HM
- SFNN_HALFKA2
- SFNN_KA2

対応するLayerStack:

- k3k3(king3-by-king3)


### 使い方

- [チュートリアル](docs/ja/tutorial/) : インストール、ビルド、最初の学習までを順を追って解説する。
- [ドキュメント](docs/ja/) : 教師データのフォーマット、出力フォーマットなどの仕様レベルの詳細。


### ビルド

```bash
# NVIDIA GPU (CUDA 12.x; Windows-native CUDA C++ backend)
cargo build --release --features cuda-cpp-backend --example bulletou
```

CPU のみでの学習はサポートされていない。現在の BulletOu 学習 backend は `cuda-cpp`。


### 系譜・上流

- **オリジナル**: [jw1912/bullet](https://github.com/jw1912/bullet) — 汎用 NNUE トレーナー (チェス向け)
- **上流**: [SH11235/bullet-shogi](https://github.com/SH11235/bullet-shogi) — 将棋向け fork。`.pack` loader (やねうら王 `gensfen` のゲーム単位可変長ファイルを内部で `PackedSfenValue` 列に展開)、HalfKA / HalfKP / Threat / HandThreat 特徴量、KP 絶対進行度 bucket 付き Layer Stack を実装
- **本リポジトリ**: [yaneurao/BulletOu](https://github.com/yaneurao/BulletOu) — やねうらお によるやねうら王向け改造版


### ライセンス

MIT (上流から継承)。元のコピーライト表記は `LICENSE` に保持されている。
