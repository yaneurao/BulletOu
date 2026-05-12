# 0. 概要 — BulletOu は何を学習するか

<a href="../../en/tutorial/0-overview.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

コードを書き始める前に、3 つの質問に答えておく:

1. BulletOu は何のためのツールか?
2. どの評価関数を学習できるか?
3. 全体の流れはどうなっているか?

## BulletOu は何のためのツールか?

BulletOu は **将棋の評価関数を学習するトレーナー**。大量の学習用局面 (各局面にスコアと対局結果のラベルが付いている) を入力すると、将棋エンジン — 主に [やねうら王 (YaneuraOu)](https://github.com/yaneurao/YaneuraOu) — が読み込んで評価関数として使えるバイナリファイルを出力する。

BulletOu 自体は将棋を **指さない**。パイプラインの中の「学習部分」を担うツールであり、出力されたファイルは別途エンジンが対局時に使う。

## 対応する評価関数

BulletOu の中核は `bullet` (jw1912) と `bullet-shogi` (SH11235) から継承しており、**将棋向けの NNUE 評価値ネットワーク**と、**やねうら王の旧 KPPT 系評価関数**の両方を対象とする。

| 評価関数 | 概要 |
|---|---|
| **NNUE (HalfKP / HalfKA / Layer Stack)** | bullet-shogi から継承。KP 絶対進行度 bucket 付き Layer Stack が最強構成として典型 |
| **NNUE + Threat / HandThreat / HandCount 特徴量** | 7 種類の入力特徴量バリエーションあり |
| **KPPT** | `bullet_ou_train --eval-type {kppt-kk,kppt-kkp,kppt-kpp}` で KK / KKP / KPP の各 component を学習し、3 ファイル `.bin` を生成 (elmo(WCSC27) 互換)。詳細は [3. KPPT / KPP_KKPT 学習](3-kppt-roadmap.md) |
| **KPP_KKPT (factorise 版)** | KK / KKP は KPPT と共通、KPP のみ手番チャンネルなしで書く (`--eval-type kpp-kkpt-kpp`) |
| KK のみ / KKP のみ | 単体 `bullet_ou_train --eval-type kppt-kk` / `kppt-kkp` で生成可能 |

## 学習データはどこから来るか

BulletOu は以下のいずれかのフォーマットで学習データを読み込める:

- **`.pack`** — [YaneuraOu-ScriptCollection](https://github.com/yaneurao/YaneuraOu-ScriptCollection) の `gensfen` スクリプトで生成
- **`.hcpe`** / **`.hcpe3`** — dlshogi 系のフォーマット

学習データは BulletOu には同梱されていない。自分で生成するか、共有データセットを使う。

## 出力はどこに行くか

学習が完了したとき (および途中のチェックポイントごとに)、BulletOu は対象の評価関数タイプに応じたバイナリファイルを書き出す。NNUE 系であれば **`nn.bin`** (やねうら王エンジンが対局時に読み込む評価関数パラメーターファイル)。KPPT 系であれば `KK_synthesized.bin` / `KKP_synthesized.bin` / `KPP_synthesized.bin` の 3 ファイル。

具体的なエンジンへの組み込み方はエンジン側のドキュメントを参照。

## 全体の流れ

```
[ 学習データの生成・取得 ]
        │
        │  YaneuraOu-ScriptCollection の gensfen スクリプト → *.pack
        ▼
[ BulletOu で学習 ]              ← このチュートリアルの対象
        │
        │  cargo run --release --example ... -- --data ... --output ...
        ▼
[ 出力ファイル ]                  ← エンジンが対局時に読み込む
        nn.bin (NNUE 系)
        または KK_synthesized.bin / KKP_synthesized.bin / KPP_synthesized.bin (KPPT 系)
```

このチュートリアルの残り:

- [1. クイックスタート](1-quickstart.md) — ツールチェーンを整えて、smoke test 用の最小学習を動かす
- [2. NNUE チュートリアル](2-nnue-tutorial.md) — 実際の NNUE を学習する流れを詳しく
- [3. KPPT / KPP_KKPT 学習](3-kppt-roadmap.md) — 旧評価関数の学習方法
