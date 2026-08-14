# 2. 教師データを用意する

<a href="../../en/tutorial/2-data.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

ゴール: 学習する評価関数を1つ選び、BulletOu に渡す教師データを用意します。

## 2.1 最初に選ぶ `--arch`

初回はこれで十分です。

```text
NNUE_halfkp_256x2_32_32
```

これは小さめの HalfKP NNUE です。学習が通るか確認しやすく、やねうら王にも読み込ませやすい構成です。

他の評価関数を試すのは、最初の学習が通ってからで構いません。
このチュートリアルでは、この `--arch` だけを使います。

## 2.2 教師データ

BulletOu は次の形式を読み込めます。

| 拡張子 | 説明 |
| --- | --- |
| `.psv` | やねうら王系の固定長局面データ |
| `.bin` | `.psv` と同じ形式として扱う |
| `.hcpe` | Apery / dlshogi 系の固定長局面データ |
| `.hcpe3` | dlshogi 系の棋譜ベースデータ |
| `.pack` | やねうら王の gensfen スクリプトの出力 |

チュートリアルでは、作業ディレクトリに `teachers/` フォルダを作り、その中に教師ファイルを置く想定にします。

```text
teachers/
  teacher.psv
```

フォルダを `--teacher teachers` のように指定すると、中の教師ファイルを順に読みます。

## 2.3 shuffle について

教師局面は混ざっているほうが学習が安定します。

BulletOu はデフォルトで学習時 shuffle を行うので、初回は特に指定しなくて構いません。
メモリ使用量や shuffle window を調整したくなったら [応用編](../advanced/) を見てください。

## 2.4 score scale が違う教師データを混ぜる場合

DL 系モデルで re-score した教師データ同士でも、勝率を評価値へ戻すときの係数が違うと、score の大きさがそろいません。

最初の学習では気にしなくて構いません。複数の教師データを混ぜる段階になったら、応用編の [loss の scale と `FV_SCALE`](../advanced/scale-and-fv-scale.md) を見てください。

---

次へ: [3. 学習を走らせる](3-train.md)

前へ: [1. クイックスタート](1-quickstart.md)
