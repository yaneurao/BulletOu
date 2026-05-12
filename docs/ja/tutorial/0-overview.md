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

現状 `bulletou` で学習できるのは以下の 4 種類:

| `--eval-type` | 概要 | 出力 |
|---|---|---|
| **`NNUE_HALFKP`** ★初心者はここから | 古典的 HalfKP NNUE (那須さん 2018 年 PR #75 と同等)。詳細は [NNUE HalfKP 学習](../shogi/halfkp.md) | `nn.bin` |
| `NNUE_KP` | HalfKP と同じ 4 層 ClippedReLU だが入力を K + P に分割した軽量版。詳細は [NNUE K-P 学習](../shogi/kp.md) | `nn.bin` |
| `KPPT` | 旧来の KK + KKP + KPP の 3 ファイル組 (elmo(WCSC27) 互換)。詳細は [KPPT / KPP_KKPT 学習](../shogi/kppt.md) | `KK_synthesized.bin` + `KKP_synthesized.bin` + `KPP_synthesized.bin` |
| `KPP_KKPT` | KPPT の factorised 版 (KPP のみ手番チャンネルなしでサイズ半減) | 同上 (KPP の layout だけ違う) |

将来サポート予定 (input feature の Rust 実装はあるが、`bulletou` からは未到達): HalfKA / HalfKA_hm / Threat / HandThreat / HandThreatDefensive / HandCount / SFNN + ls9 (NNUEwoSQPT1536) など。

## 学習データはどこから来るか

BulletOu は以下のいずれかのフォーマットで学習データを読み込める:

| フォーマット | gensfen スクリプト で生成可能 | dlshogi の selfplay で生成可能 | 説明 |
|---|---|---|---|
| `.pack` | ☑ | □ | やねうら王の gensfen スクリプトで生成される |
| `.psv` | ☑ | □ | やねうら王の旧来からある教師フォーマット |
| `.hcpe` | ☑ | ☑ | Apery の教師フォーマット |
| `.hcpe3` | ☑ | ☑ | hcpe を dlshogi 作者が拡張したフォーマット |

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
        │  cargo run --release --example bulletou -- --eval-type ... --teacher ... --output ...
        ▼
[ 出力ファイル ]                  ← エンジンが対局時に読み込む
        nn.bin (NNUE 系)
        または KK_synthesized.bin / KKP_synthesized.bin / KPP_synthesized.bin (KPPT 系)
```

このチュートリアルの残り:

- [1. クイックスタート](1-quickstart.md) — ツールチェーンを整えて、smoke test 用の最小学習を動かす
- [2. 学習を走らせる](2-training.md) — 実データで評価関数を学習し、エンジンに繋ぐまでを通す
- [KPPT / KPP_KKPT 学習](../shogi/kppt.md) — 旧評価関数の学習方法 (リファレンス)
