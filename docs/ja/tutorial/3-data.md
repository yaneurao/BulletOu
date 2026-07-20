# 3. 教師データを用意する — 学習対象の選択とデータ前処理

<a href="../../en/tutorial/3-data.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

ゴール: 何を学習させるかを決め、学習に渡すための教師データを準備する。

この章は [1. クイックスタート](1-quickstart.md) を完了している前提 — ツールチェーンが動き、smoke test の学習が成功した状態。

本チュートリアルでは **NNUE HalfKP を例に** 解説するが、`--arch` を切り替えるだけで他のターゲット (NNUE K-P / NNUE HalfKPE9 / KPPT / KPP_KKPT) も同じコマンド形式で学習できる。

## 3.1 学習対象を選ぶ

`bulletou --arch <X>` で学習する評価関数を選ぶ。KPPT 系は `KPPT` / `KPP_KKPT`、NNUE / SFNN 系は `YANEURAOU_ENGINE_` prefix を取り除いた architecture 名を指定する。代表的な `<X>`:

| `--arch` の値 | 何を学習するか | 出力ファイル (per save) |
|---|---|---|
| **`NNUE_halfkp_256x2_32_32`** ★初心者はここから | 古典的な HalfKP NNUE。やねうら王がもっとも長く採用している評価関数形式。詳細は [NNUE HalfKP 学習](../shogi/halfkp.md) | `nn.bin` |
| `NNUE_kp_256x2_32_32` | HalfKP と同じ NN だが入力が K + P の独立特徴。詳細は [NNUE K-P 学習](../shogi/kp.md) | `nn.bin` |
| `NNUE_ka2_256x2_32_32` | K+A2 入力の NNUE。詳細は [NNUE K-A2 学習](../shogi/ka2.md) | `nn.bin` |
| `NNUE_halfkpe9_256x2_32_32` | HalfKP に利き数情報 (自軍/敵軍 0/1/2 の 9 通り) を多重化した拡張版。詳細は [NNUE HalfKPE9 学習](../shogi/halfkpe9.md) | `nn.bin` |
| `NNUE_halfkpvm_256x2_32_32` | HalfKP の玉位置を左右対称に折り畳んだ版 (6 筋以降を 4 筋以前にミラー)。入力次元は HalfKP の約 1/2 | `nn.bin` |
| `SFNN_halfkahm2_1536_15_32_k3k3` | やねうら王 `YANEURAOU_ENGINE_SFNN1536` ビルド用の LayerStacks 系。使い方は [§9 LayerStack](9-layerstack.md)、仕様詳細は [SFNN-1536 リファレンス](../shogi/sfnn-1536.md) | `nn.bin` |
| `SFNN_halfkahm1_1536_15_32_k3k3` | ↑ の v1 アブレーション版 | `nn.bin` |
| `SFNN_ka2_1536_15_32_k3k3` | 軽量な K+A2 入力の SFNN | `nn.bin` |
| `KPPT` | 旧来の KK + KKP + KPP 3 ファイル組 (elmo(WCSC27) 互換)。詳細は [KPPT / KPP_KKPT 学習](../shogi/kppt.md) | `KK_synthesized.bin` + `KKP_synthesized.bin` + `KPP_synthesized.bin` |
| `KPP_KKPT` | KPPT の factorised 版 (KPP のみ手番チャンネルなし、サイズ半減) | 同上 (KPP layout のみ違う) |

`NNUE_halfkp_1024x2_8_64` や `SFNN_ka2_8192_7_64_g8_k3k3` のようなサイズ違いも、対応するやねうら王 architecture があれば実験用途で受け付ける。

## 3.2 学習データを用意する

`.pack` / `.hcpe` / `.hcpe3` / `.psv` のいずれかのファイルが必要。

- **自分で生成** — [YaneuraOu-ScriptCollection](https://github.com/yaneurao/YaneuraOu-ScriptCollection) の `gensfen` スクリプトで `.pack` を出力するか、dlshogi 系のデータ生成で `.hcpe` / `.hcpe3` を作る。チュートリアル目的なら 1000 万〜1 億局面で十分。
- **共有データセットを使う** — 将棋コミュニティでは各フォーマットのデータが共有されている。

本チュートリアルでは作業ディレクトリ直下に `teachers/` を作り、その下に教師ファイルを置く構成を仮定する:

```
teachers/
    teacher.pack
```

(`.hcpe` / `.hcpe3` / `.psv` でも同様に動く。フォーマットは拡張子から自動判別される。複数ファイル混在もディレクトリ指定で OK だが、すべて同じ拡張子であること。)

### 教師ファイルは事前にシャッフルしておく

> ⚠️ **重要**: 教師ファイルは BulletOu に渡す前にシャッフルしておくこと。

BulletOu は学習時に教師局面を追加シャッフルしない。`--buffer-mb` は読み込みバッファのサイズであり、シャッフル用の指定ではない。

`gensfen` / dlshogi-style 生成器の出力は **同一対局内の局面が連続して並んでいる** のが普通なので、ファイル全体をシャッフルしないまま学習すると、近い局面ばかりが連続して mini-batch に入り、loss や plateau 判定が教師の局所的な偏りに振り回される。

対策:
- **`.hcpe` / `.psv`**: [YaneuraOu-ScriptCollection](https://github.com/yaneurao/YaneuraOu-ScriptCollection) の `teacher/shuffle_split_teacher_external.py` を使う。巨大な教師フォルダでも全体をメモリに載せず、bucket 分配してから出力ファイルへ分割できる。
- **`.hcpe3` / `.pack`**: 棋譜単位の可変長形式なので単純な固定長レコード shuffle には向かない。生成時点で局面順を混ぜる、または `.psv` / `.hcpe` のような固定長局面形式に変換してからシャッフルする。

HCPE/PSVフォルダを 1000 万局面ごとにシャッフル分割する例:

```bash
python /path/to/YaneuraOu-ScriptCollection/teacher/shuffle_split_teacher_external.py \
    src_teacher_folder \
    dst_teacher_folder \
    --positions 10000000
```

出力ファイルは `shuffled-00001.hcpe`, `shuffled-00002.hcpe`, ... のような名前になる。1000 万局面ずつ 1 万ファイルを超える規模なら `--digits 6` のように桁数を増やす。

### 小さなサブセットで動作確認したい場合

巨大なデータセット (数十 GB) でいきなり動かす前に、小さなサブセットで試したいときは、`gensfen` 等で小さめのファイルを生成するか、`--positions-per-superbatch` を指定して 1 superbatch あたりの消費量を絞る ([§6.1 学習スケジュール](6-tune.md#61-学習スケジュール) 参照)。

---

次へ: [4. 学習を走らせる](4-train.md) — 実データで評価関数を学習する

前へ: [1. クイックスタート](1-quickstart.md)
