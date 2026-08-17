# BulletOu チュートリアル

<a href="../../en/tutorial/README.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

BulletOu を初めて使う人向けの最短ガイドです。
上から順に読めば、学習を1回動かして、出力をエンジンで読むところまで進めます。

細かい調整、比較実験、出力ファイルの詳しい検証は [応用編](../advanced/) に分けています。

| # | ページ | この章でやること |
|---|---|---|
| 0 | [概要](0-overview.md) | BulletOu が何を学習するか、どの評価関数に対応しているかを理解する |
| 1 | [クイックスタート](1-quickstart.md) | 必要なものを揃えて BulletOu をビルドし、最小の学習を動かして動作確認する |
| 2 | [教師データを用意する](2-data.md) | architecture を選び、教師データを用意する |
| 3 | [学習を走らせる](3-train.md) | 最低限の `bulletou` コマンドを実行する |
| 4 | [検証を有効にする](4-validation.md) | `--test-teacher` と `--validation-rate` で accuracy / loss を見る |
| 5 | [中断・再開](5-resume.md) | 止まった学習を続きから再開する |
| 6 | [結果を確認する](6-result.md) | 出力ファイルと学習ログを見る |
| 7 | [エンジンに組み込む](7-engine.md) | やねうら王エンジンで動作確認する |

次に調整や比較実験をしたくなったら [応用編](../advanced/) へ進んでください。
仕様レベルの詳細は [リファレンス](../) を参照してください。
