# BulletOu リファレンス

<a href="../en/"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

学習パイプラインの仕様レベルのドキュメント。「ある程度わかっている前提」で書かれている。

- [NNUE の基礎](1-basics.md) — 入力/隠れ/出力層、perspective ネットワーク
- [学習済みネットワーク](4-saved-networks.md) — チェックポイントのレイアウト、SavedFormat、量子化、変換チェーン

将棋固有:

- [shogi/halfkp.md](shogi/halfkp.md) — NNUE HalfKP 評価関数の学習
- [shogi/kppt.md](shogi/kppt.md) — KPPT / KPP_KKPT 評価関数の学習
- [shogi/kp-absolute-progress.md](shogi/kp-absolute-progress.md) — KP 絶対値を用いた進行度推定
- [shogi/shogi_progress_kpabs_train.md](shogi/shogi_progress_kpabs_train.md) — `shogi_progress_kpabs_train` ツールの CLI 仕様
