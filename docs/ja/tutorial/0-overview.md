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

| 代表的な `--arch` の値 | 概要 | 出力 |
|---|---|---|
| **`NNUE_halfkp_256x2_32_32`** ★初心者はここから | 古典的な HalfKP NNUE。やねうら王がもっとも長く採用している評価関数形式。詳細は [NNUE HalfKP 学習](../shogi/halfkp.md) | `nn.bin` |
| `NNUE_kp_256x2_32_32` | HalfKP と同じ 4 層 ClippedReLU だが入力を K + P に分割した軽量版。詳細は [NNUE K-P 学習](../shogi/kp.md) | `nn.bin` |
| `NNUE_ka2_256x2_32_32` | K+A2 入力の NNUE。詳細は [NNUE K-A2 学習](../shogi/ka2.md) | `nn.bin` |
| `NNUE_halfkpe9_256x2_32_32` | HalfKP に「駒のマスに対する自軍/敵軍の利き数 (0/1/2 にクリップ、9 通り)」を多重化した拡張版 (1,128,492 次元、HalfKP × 9)。詳細は [NNUE HalfKPE9 学習](../shogi/halfkpe9.md) | `nn.bin` |
| `NNUE_halfkpvm_256x2_32_32` | HalfKP の玉位置を左右対称に折り畳んだ版 (6 筋以降を 4 筋以前にミラー、69,660 次元、HalfKP の約 1/2) | `nn.bin` |
| `SFNN_halfkahm2_1536_15_32_k3k3` | やねうら王 NNUEwoSQPT1536 ビルド用の LayerStacks 系評価関数 (HalfKA_hm2 入力)。詳細は [SFNN-1536 学習リファレンス](../shogi/sfnn-1536.md)、使い方は [§9 LayerStack](9-layerstack.md) | `nn.bin` |
| `SFNN_halfkahm1_1536_15_32_k3k3` | ↑ の HalfKA_hm1 (v1) を使ったアブレーション用 | `nn.bin` |
| `SFNN_ka2_1536_15_32_k3k3` | SFNN-1536 LayerStacks の topology はそのままで、入力を `K + A2` (1791 次元) に置き換えた版。軽量アブレーション用。 | `nn.bin` |
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
        │  ./target/release/examples/bulletou --arch ... --teacher ... --output ...
        ▼
[ 出力ファイル ]                  ← エンジンが対局時に読み込む
        nn.bin (NNUE 系)
        または KK_synthesized.bin / KKP_synthesized.bin / KPP_synthesized.bin (KPPT 系)
```

このチュートリアルの残り:

- [1. クイックスタート](1-quickstart.md) — ツールチェーンを整えて、smoke test 用の最小学習を動かす
- [2. `bulletou_lib` を自分のコードから使う](2-bullet-lib.md) — 開発者向け補足 (任意)
- [3. 教師データを用意する](3-data.md) — architecture の選択と教師データの前処理 (シャッフル)
- [4. 学習を走らせる](4-train.md) — `bulletou` コマンドの実行
- [5. 中断・再開](5-resume.md) — `--output` と学習設定が同じなら自動 resume
- [5.5 追加学習の仕方](5b-additional-training.md) — 完走後にさらに epoch を積む / batch_size や教師を変えて続行
- [6. 学習をチューニング](6-tune.md) — スケジュールと教師ターゲット (`--lambda`) の調整 (任意)
- [7. 結果を確認する](7-result.md) — 出力レイアウト、`learn.log` の読み方
- [8. エンジンに組み込む](8-engine.md) — やねうら王で動作確認
- [9. LayerStack](9-layerstack.md) — 局面ごとに別サブネットを使い分ける評価関数 (SFNN 系) の使い方
- [KPPT / KPP_KKPT 学習](../shogi/kppt.md) — 旧評価関数の学習方法 (リファレンス)
