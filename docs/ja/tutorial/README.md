# BulletOu チュートリアル

<a href="../../en/tutorial/README.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

BulletOu を初めて使う人向けの段階的ガイド。上から順に読むことを推奨。

| # | ページ | この章でやること |
|---|---|---|
| 0 | [概要](0-overview.md) | BulletOu が何を学習するか、どの評価関数に対応しているかを理解する |
| 1 | [クイックスタート](1-quickstart.md) | 必要なものを揃えて BulletOu をビルドし、最小の学習を動かして動作確認する |
| 2 | [`bulletou_lib` を自分のコードから使う](2-bullet-lib.md) | 開発者向け補足 (独自 example の登録、外部からの import など。任意) |
| 3 | [教師データを用意する](3-data.md) | architecture を選び、教師データを用意・シャッフルする |
| 4 | [学習を走らせる](4-train.md) | `bulletou` コマンドの実行 (arch / 教師の渡し方) |
| 5 | [中断・再開](5-resume.md) | `--output` と学習設定が同じなら自動 resume |
| 5.5 | [追加学習の仕方](5b-additional-training.md) | 完走後にさらに epoch を積む / 設定変更で続行 (batch_size 変更、教師差し替え、LR 微調整、fine-tune) |
| 6 | [学習をチューニング](6-tune.md) | スケジュール (`--lr` / `--superbatches`) と教師ターゲット (`--lambda`) の調整 (任意) |
| 7 | [結果を確認する](7-result.md) | 出力レイアウト / `learn.log` の読み方 |
| 8 | [エンジンに組み込む](8-engine.md) | やねうら王エンジンでの動作確認 |
| 9 | [LayerStack](9-layerstack.md) | 局面ごとに別のサブネットを使い分ける評価関数の使い方 (SFNN 系のとき適用) |

チュートリアルを読み終えたあとは、仕様レベルの詳細 (データフォーマット、出力フォーマットなど) は [リファレンス](../) を参照。
