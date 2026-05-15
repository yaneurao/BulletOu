# BulletOu リファレンス

<a href="../en/"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

学習パイプラインの仕様レベルのドキュメント。「ある程度わかっている前提」で書かれている。

初めて BulletOu を使う場合は、まず [チュートリアル](tutorial/) を参照。

- [NNUE の基礎](reference/1-basics.md) — 入力/隠れ/出力層、perspective ネットワーク
- [BulletOu を始める](reference/2-getting-started.md) — トレーニング全体の概観 (上流由来)
- [学習データフォーマット](reference/3-data.md) — bulletformat / .pack / .hcpe / .hcpe3 / .psv (上流由来 + 将棋拡張)
- [学習済みネットワーク](reference/4-saved-networks.md) — チェックポイントのレイアウト、SavedFormat、量子化、変換チェーン

将棋固有:

- [shogi/halfkp.md](shogi/halfkp.md) — NNUE HalfKP 評価関数の学習
- [shogi/halfkpe9.md](shogi/halfkpe9.md) — NNUE HalfKPE9 評価関数の学習 (HalfKP に利き数情報を加えた版)
- [shogi/kp.md](shogi/kp.md) — NNUE K-P 評価関数の学習 (HalfKP と同じ NN、入力だけ違う版)
- [shogi/ka2.md](shogi/ka2.md) — NNUE K-A2 / SFNN K-A2 評価関数の学習 (`FeatureSet<K, A2>` v2 玉 collapse 全駒特徴)
- [shogi/kppt.md](shogi/kppt.md) — KPPT / KPP_KKPT 評価関数の学習
- [shogi/sfnn-1536.md](shogi/sfnn-1536.md) — SFNN-1536 (やねうら王 NNUEwoSQPT1536 ビルド向け、LayerStacks=9) の詳細仕様
- [shogi/kp-absolute-progress.md](shogi/kp-absolute-progress.md) — KP 絶対値を用いた進行度推定
- [shogi/shogi_progress_kpabs_train.md](shogi/shogi_progress_kpabs_train.md) — `shogi_progress_kpabs_train` ツールの CLI 仕様
