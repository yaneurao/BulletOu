# BulletOu ドキュメント

<a href="../en/"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

ドキュメントは 2 階層に分かれている:

- **チュートリアル** — 初めての人向けの段階的ガイド。BulletOu に初めて触れる場合は最初にこちらを読む
- **リファレンス** — 仕様と設計の詳細。特定の挙動を理解・改変したい場合に参照する

---

## チュートリアル (最初に読む)

1. [概要 — BulletOu が学習する対象と、対応する評価関数の種類](tutorial/0-overview.md)
2. [クイックスタート — インストール、ビルド、最初の学習を動かす](tutorial/1-quickstart.md)
3. [NNUE チュートリアル — 将棋 NNUE の学習を詳しく見ていく](tutorial/2-nnue-tutorial.md)

## リファレンス

学習パイプラインの仕様レベルのドキュメント。「ある程度わかっている前提」で書かれている。

- [NNUE の基礎](1-basics.md) — 入力/隠れ/出力層、perspective ネットワーク
- [学習済みネットワーク](4-saved-networks.md) — チェックポイントのレイアウト、SavedFormat、量子化、変換チェーン

将棋固有:

- [shogi/kppt.md](shogi/kppt.md) — KPPT / KPP_KKPT 評価関数の学習
- [shogi/kp-absolute-progress.md](shogi/kp-absolute-progress.md) — KP 絶対値を用いた進行度推定
- [shogi/shogi_progress_kpabs_train.md](shogi/shogi_progress_kpabs_train.md) — `shogi_progress_kpabs_train` ツールの CLI 仕様
