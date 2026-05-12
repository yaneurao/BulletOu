# 3. 教師データを用意する — 学習対象の選択とデータ前処理

<a href="../../en/tutorial/3-data.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

ゴール: 何を学習させるかを決め、学習に渡すための教師データを準備する。

この章は [1. クイックスタート](1-quickstart.md) を完了している前提 — ツールチェーンが動き、smoke test の学習が成功した状態。

本チュートリアルでは **NNUE HalfKP を例に** 解説するが、`--eval-type` を切り替えるだけで他のターゲット (NNUE K-P / NNUE HalfKPE9 / KPPT / KPP_KKPT) も同じコマンド形式で学習できる。

## 3.1 学習対象を選ぶ

`bulletou --eval-type <X>` で学習する評価関数を選ぶ。現在公開されている `<X>`:

| `--eval-type` | 何を学習するか | 出力ファイル (per save) | `--arch` を使うか |
|---|---|---|---|
| **`NNUE_HALFKP`** ★初心者はここから | 古典的な HalfKP NNUE。やねうら王がもっとも長く採用している評価関数形式。詳細は [NNUE HalfKP 学習](../shogi/halfkp.md) | `nn.bin` | 使う |
| `NNUE_KP` | HalfKP と同じ NN だが入力が K + P の独立特徴。詳細は [NNUE K-P 学習](../shogi/kp.md) | `nn.bin` | 使う |
| `NNUE_HALFKPE9` | HalfKP に利き数情報 (自軍/敵軍 0/1/2 の 9 通り) を多重化した拡張版。詳細は [NNUE HalfKPE9 学習](../shogi/halfkpe9.md) | `nn.bin` | 使う |
| `NNUE_HALFKPVM` | HalfKP の玉位置を左右対称に折り畳んだ版 (6 筋以降を 4 筋以前にミラー)。入力次元は HalfKP の約 1/2 | `nn.bin` | 使う |
| `KPPT` | 旧来の KK + KKP + KPP 3 ファイル組 (elmo(WCSC27) 互換)。詳細は [KPPT / KPP_KKPT 学習](../shogi/kppt.md) | `KK_synthesized.bin` + `KKP_synthesized.bin` + `KPP_synthesized.bin` | 使わない |
| `KPP_KKPT` | KPPT の factorised 版 (KPP のみ手番チャンネルなし、サイズ半減) | 同上 (KPP layout のみ違う) | 使わない |

将来 `--eval-type` に追加予定: HalfKA / SFNN + ls9 (NNUEwoSQPT1536) など。

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

bullet のローダーは **メモリ内 shuffle buffer (デフォルト 256 MB ≒ HCPE で 670 万局面)** で局面を Fisher-Yates シャッフルしてから batch に切り出す。これは **buffer 内シャッフルのみ** で buffer をまたいだクロスシャッフルはしないため、教師ファイル先頭から順に 670 万局面ずつ読みながら局所的にシャッフルしているだけになる。

`gensfen` / dlshogi-style 生成器の出力は **同一対局内の局面が連続して並んでいる** のが普通なので、ファイル全体をシャッフルしないまま学習すると、buffer 境界 (≒ 410 batch ごと、`--batch-size 16384` の場合) で分布が突然変わって **loss が周期的に跳ねる** ことになる。

対策:
- **`.hcpe` / `.hcpe3`**: dlshogi のシャッフルスクリプトを使うのが簡単。HCPE は固定長レコード (38 byte) なのでバイト単位のランダムシャッフルで OK。dlshogi リポジトリの `utils/` 配下にツールが用意されている。
- **`.pack`**: `gensfen` の出力時点でシャッフルオプションを有効にする、もしくは出力後に PSV に変換して shuffle。
- **緊急回避** (シャッフル前のファイルしか手元にないとき): `--buffer-mb` をファイルサイズと同等以上に上げて 1 バッファに全件収める。例: 1.94 GB の `.hcpe` (≒ 5100 万局面) なら `--buffer-mb 2048` で buffer 境界が無くなる。GPU メモリではなく **CPU 側 RAM** を食うので、メモリに余裕がある環境向け。

### 小さなサブセットで動作確認したい場合

巨大なデータセット (数十 GB) でいきなり動かす前に、小さなサブセットで試したいときは、`gensfen` 等で小さめのファイルを生成するか、`--batches-per-superbatch` を指定して 1 superbatch あたりの消費量を絞る ([§6.1 学習スケジュール](6-tune.md#61-学習スケジュール) 参照)。

---

次へ: [4. 学習を走らせる](4-train.md) — 実データで評価関数を学習する

前へ: [1. クイックスタート](1-quickstart.md)
