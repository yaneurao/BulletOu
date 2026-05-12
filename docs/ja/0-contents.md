# BulletOu ドキュメント

<a href="../en/0-contents.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

ドキュメントは 2 階層に分かれている:

- **チュートリアル** — 初めての人向けの段階的ガイド。BulletOu に初めて触れる場合は最初にこちらを読む
- **リファレンス** — 仕様と設計の詳細。特定の挙動を理解・改変したい場合に参照する

---

## チュートリアル (最初に読む)

全体の目次は [`tutorial/`](tutorial/) に。

1. [概要 — BulletOu が学習する対象と、対応する評価関数の種類](tutorial/0-overview.md)
2. [クイックスタート — インストール、ビルド、最初の学習を動かす](tutorial/1-quickstart.md)
3. [NNUE チュートリアル — 将棋 NNUE の学習を詳しく見ていく](tutorial/2-nnue-tutorial.md)
4. [KPPT / KPP_KPPT ロードマップ — 旧評価関数対応の現状と計画](tutorial/3-kppt-roadmap.md)

## リファレンス

学習パイプラインの仕様レベルのドキュメント。「ある程度わかっている前提」で書かれている。

NNUE 学習の中身:

1. [NNUE の基礎](1-basics.md) — 入力/隠れ/出力層、perspective ネットワーク、よくある罠
2. [はじめかた](2-getting-started.md) — rustup、examples、bullet-utils、バックエンド設定
3. [教師データ](3-data.md) — ワークフロー、同梱データローダー、ChessBoard / binpack 形式
4. [学習済みネットワーク](4-saved-networks.md) — チェックポイントのレイアウト、SavedFormat、量子化、変換チェーン

将棋固有:

- [shogi/kp-absolute-progress.md](shogi/kp-absolute-progress.md) — KP 絶対値を用いた進行度推定
- [shogi/shogi_progress_kpabs_train.md](shogi/shogi_progress_kpabs_train.md) — `shogi_progress_kpabs_train` ツールの CLI 仕様

KPPT / KPP_KPPT:

- 現状は未実装。設計案は [tutorial/3-kppt-roadmap.md](tutorial/3-kppt-roadmap.md) を参照。

## examples

実際にコピーして編集する例は [`examples/`](../../examples/) ディレクトリにある (progression シリーズ [`examples/progression/`](../../examples/progression/) と、将棋固有の `shogi_simple`, `shogi_layerstack` など)。チュートリアルの中でこれらのいくつかを順に動かしていく。
