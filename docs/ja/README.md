# BulletOu ドキュメント

<a href="../en/"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

BulletOu は、やねうら王向け評価関数を学習するためのプロジェクトです。
初めて使う場合は、まず [チュートリアル](tutorial/) を読んでください。必要最小限の手順だけで、学習を1回動かすところまで案内します。

より細かい調整や検証は [応用編](advanced/) に分けています。

## 読む順番

| 目的 | ページ |
| --- | --- |
| まず学習を動かしたい | [チュートリアル](tutorial/) |
| 学習条件を調整したい、出力を詳しく検証したい | [応用編](advanced/) |
| ファイル形式や実装仕様を確認したい | [リファレンス](reference/) |
| 評価関数ごとの詳細を確認したい | [将棋向け評価関数](shogi/) |

## リファレンス

- [NNUE の基本](reference/1-basics.md)
- [BulletOu の学習パイプライン](reference/2-getting-started.md)
- [学習データフォーマット](reference/3-data.md)
- [保存されるネットワーク](reference/4-saved-networks.md)

## 将棋向け評価関数

- [NNUE HalfKP](shogi/halfkp.md)
- [NNUE HalfKPE9](shogi/halfkpe9.md)
- [NNUE K-P](shogi/kp.md)
- [NNUE K-A2 / SFNN K-A2](shogi/ka2.md)
- [KPPT / KPP_KKPT](shogi/kppt.md)
- [SFNN-1536](shogi/sfnn-1536.md)
