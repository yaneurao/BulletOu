# 0. 概要

<a href="../../en/tutorial/0-overview.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

BulletOu は、将棋エンジン用の評価関数を学習するツールです。
大量の教師局面を読み込み、やねうら王が読める `nn.bin` などの評価関数ファイルを書き出します。

BulletOu 自体は将棋を指しません。役割は「教師データから評価関数を学習すること」です。

## 最初は HalfKP NNUE から

このチュートリアルでは、まず次の評価関数を学習します。

```text
NNUE_halfkp_256x2_32_32
```

小さめで扱いやすく、動作確認に向いています。

他にも NNUE K-P、NNUE K-A2、SFNN、KPPT 系を学習できます。詳しい一覧は [リファレンス](../) と [応用編](../advanced/) を参照してください。

## 全体の流れ

```text
教師データを用意する
        ↓
BulletOu で学習する
        ↓
checkpoints/ に nn.bin ができる
        ↓
やねうら王で nn.bin を読む
```

## このチュートリアルでやること

| # | ページ | 内容 |
| --- | --- | --- |
| 1 | [クイックスタート](1-quickstart.md) | ビルドと smoke test |
| 2 | [教師データを用意する](2-data.md) | `--arch` と `--teacher` の準備 |
| 3 | [学習を走らせる](3-train.md) | 最小コマンドで学習 |
| 4 | [検証を有効にする](4-validation.md) | accuracy / loss を測る |
| 5 | [中断・再開](5-resume.md) | 止まったときの再開 |
| 6 | [結果を確認する](6-result.md) | 出力ファイルとログ |
| 7 | [エンジンに組み込む](7-engine.md) | やねうら王で動作確認 |

次へ: [1. クイックスタート](1-quickstart.md)
