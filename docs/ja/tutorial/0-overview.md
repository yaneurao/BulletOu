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

BulletOu の中核は `bullet` (jw1912) と `bullet-shogi` (SH11235) から継承しており、**将棋向け NNUE 評価値ネットワーク** にはネイティブで対応。やねうら王の旧来の評価関数群への対応は予定中。

| 評価関数 | BulletOu での状態 | 備考 |
|---|---|---|
| **NNUE (HalfKP / HalfKA / Layer Stack)** | **現在対応** | bullet-shogi から継承。KP 絶対進行度 bucket 付き Layer Stack が最強構成として典型 |
| **NNUE + Threat / HandThreat / HandCount 特徴量** | 現在対応 | 7 種類の入力特徴量バリエーションあり |
| **KPPT** | 対応予定 | 構造 (大規模な疎 embedding テーブルの和、隠れ層なし) が NNUE と違う。builder DSL 拡張と、やねうら王形式 `.bin` の writer が必要 |
| **KPP_KPPT** | 対応予定 | KPPT の factorise 版。KPPT 対応のあとに着手 |
| その他やねうら王旧評価関数 | 状況次第 | KK のみ / KKP のみ等のミニ版は KPPT 対応後に検討 |

「対応予定」のものは [3-kppt-roadmap.md](3-kppt-roadmap.md) で進捗を追っている。

現状の主たる対象は **将棋 NNUE**。このチュートリアルもそれを中心に進める。

## 学習データはどこから来るか

BulletOu は **PackedSfenValue** (`.pack`) — やねうら王の `gensfen` コマンドが生成するフォーマット — を読み込む。各レコードは「圧縮された局面 + その局面でのエンジン評価値 + 最終的な対局結果」のセット。

上流から継承した別のフォーマット (`bulletformat` / `binpack` 等) もサポートされているが、将棋作業では `.pack` が標準。

学習データは BulletOu には **同梱されていない**。やねうら王の `gensfen` で自分で生成するか、他者が共有している `.pack` データセットを使う。

## 出力はどこに行くか

学習が完了したとき (および途中のチェックポイントごとに)、BulletOu は以下を書き出す:

- `raw.bin` — float の生の重み (学習再開に使う)
- `quantised.bin` — 量子化済み整数重み (推論に使う)
- `optimiser_state/` — optimizer の内部状態 (学習再開に使う)

将棋 NNUE の場合、エンジンが対局時に読むのは `quantised.bin`。具体的なエンジンへの組み込み方は、対象エンジンによる。やねうら王に対する組み込みについてはやねうら王側のドキュメントを参照。

## 全体の流れ

```
[ 学習データの生成・取得 ]
        │
        │  やねうら王の gensfen → *.pack
        ▼
[ BulletOu で学習 ]              ← このチュートリアルの対象
        │
        │  cargo run --release --example shogi_layerstack -- ...
        ▼
[ 出力ファイル ]
        ├── raw.bin
        ├── quantised.bin          ← 対局時にエンジンが読む
        └── optimiser_state/
```

このチュートリアルの残り:

- [1. クイックスタート](1-quickstart.md) — ツールチェーンを整えて、smoke test 用の最小学習を動かす
- [2. NNUE チュートリアル](2-nnue-tutorial.md) — 実際の NNUE を学習する流れを詳しく
- [3. KPPT / KPP_KPPT ロードマップ](3-kppt-roadmap.md) — 旧評価関数対応の現状と将来像
