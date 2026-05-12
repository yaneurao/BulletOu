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

| 評価関数 | 概要 |
|---|---|
| **NNUE HalfKP** | 玉位置 × 駒位置の sparse 特徴量 (古典的 NNUE) |
| **NNUE HalfKA** | HalfKP + 玉も特徴量に含める |
| **NNUE Layer Stack** | 局面進行度 bucket に応じて出力サブネットを切替える構成 (KP 絶対進行度 bucket が典型) |
| **NNUE + Threat** | 盤上駒の利き threat を追加した入力特徴量 |
| **NNUE + HandThreat** | 持ち駒 drop threat を追加した入力特徴量 |
| **NNUE + HandThreatDefensive** | 防御的な持ち駒 drop threat を追加 (非対称 emission) |
| **NNUE + HandCount** | 持ち駒枚数を dense aux input として追加 |
| **KPPT** | `bulletou --eval-type kppt` で KK / KKP / KPP を連続学習し、3 ファイル組を 1 コマンドで生成 (elmo(WCSC27) 互換)。詳細は [KPPT / KPP_KKPT 学習](../shogi/kppt.md) |
| **KPP_KKPT (factorise 版)** | KK / KKP は KPPT と共通、KPP のみ手番チャンネルなしで書く (`--eval-type kpp-kkpt-kpp`) |
| KK のみ / KKP のみ | 単体 `bulletou --eval-type kppt-kk` / `kppt-kkp` で生成可能 |

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
        │  cargo run --release --example ... -- --data ... --output ...
        ▼
[ 出力ファイル ]                  ← エンジンが対局時に読み込む
        nn.bin (NNUE 系)
        または KK_synthesized.bin / KKP_synthesized.bin / KPP_synthesized.bin (KPPT 系)
```

このチュートリアルの残り:

- [1. クイックスタート](1-quickstart.md) — ツールチェーンを整えて、smoke test 用の最小学習を動かす
- [2. NNUE チュートリアル](2-nnue-tutorial.md) — 実際の NNUE を学習する流れを詳しく
- [KPPT / KPP_KKPT 学習](../shogi/kppt.md) — 旧評価関数の学習方法 (リファレンス)
