# 3. 学習を走らせる

<a href="../../en/tutorial/3-train.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

ゴール: 用意した教師データから、やねうら王が読める評価関数ファイルを作ります。

このページは [2. 教師データを用意する](2-data.md) の続きです。

## 3.1 まずビルドする

```powershell
cargo build --release --features cuda-cpp-backend --example bulletou
```

Windows では、実行ファイルは次にできます。

```text
.\target\release\examples\bulletou.exe
```

## 3.2 最小コマンド

最初は HalfKP NNUE で動作確認するのが簡単です。

```powershell
.\target\release\examples\bulletou.exe `
  --arch NNUE_halfkp_256x2_32_32 `
  --teacher teachers `
  --tag first-halfkp
```

`--arch` は学習する評価関数の形です。`NNUE_halfkp_256x2_32_32` は小さめで試しやすい構成です。

`--teacher` には教師ファイル、または教師ファイルが入ったフォルダを指定します。

`--tag` は実験名です。省略しても動きますが、複数回試すなら付けておくほうが出力を見分けやすくなります。

## 3.3 出力先

学習結果は `checkpoints/` の下に保存されます。NNUE / SFNN なら各 checkpoint に `nn.bin` ができます。

例:

```text
checkpoints/
  NNUE_HALFKP-NNUE_halfkp_256x2_32_32-first-halfkp/
    0001/
      nn.bin
      state.bin
```

`nn.bin` が、やねうら王に読み込ませる評価関数ファイルです。

## 3.4 学習を短く試したい場合

巨大な教師データでいきなり長時間回す前に、次のように小さめにして動作確認できます。

```powershell
.\target\release\examples\bulletou.exe `
  --arch NNUE_halfkp_256x2_32_32 `
  --teacher teachers `
  --positions-per-superbatch 1000000 `
  --superbatches 1 `
  --max-epochs 1 `
  --tag smoke-halfkp
```

まずこの形で「読み込み・学習・保存」が通ることを確認してください。

## 3.5 画面に出るもの

学習中は、おおむね次のような行が出ます。

```text
[train] epoch 1  sb 1/1  this-sb=... pos  wall=...s  train=...s  pos/s=...
```

`pos/s` は学習速度の目安です。保存や検証の時間は学習速度からは除外されます。

accuracy / loss を学習中に見たい場合は、次のページで検証用局面を指定します。

## 3.6 bucket 数が多い SFNN を学習する場合

`hand1024_k3k3_progress4` のように bucket 数が多い SFNN では、ほとんど出現しない bucket の個別成分が不安定になることがあります。その場合は、教師データから bucket の出現回数を数えた `count.bin` を作り、学習時に次のように指定します。

```powershell
--sfnn-bucket-counts D:\BulletOu-snapshots\counts\count.bin `
--sfnn-residual-count-confidence 1.0
```

`--sfnn-residual-count-confidence 1.0` は、「bucket 固有成分のパラメーター数と同じぐらい出現するまでは、その bucket を強く信用しない」という意味です。

count による confidence は、指定しなければ無効です。factorizer 側にも同じ考え方を使う場合は、次のように指定します。

```powershell
--sfnn-axis-count-confidence 1.0 `
--sfnn-pair-count-confidence 1.0
```

必要なら、`--sfnn-king-axis-count-confidence`、`--sfnn-hand-axis-count-confidence`、`--sfnn-progress-axis-count-confidence`、`--sfnn-king-hand-pair-count-confidence`、`--sfnn-king-progress-pair-count-confidence`、`--sfnn-hand-progress-pair-count-confidence` のように、factorizer の種類ごとに分けて指定できます。

residual / axis / pair の confidence は、同じ `count.bin` を使います。詳しい作り方と式は [応用編: SFNN factorizer](../advanced/sfnn-factorizer.md) を参照してください。

`progress4` など `progressN` を含む architecture の `count.bin` を作る場合は、進行度bucketを決めるために同じ architecture の `nn.bin` も指定します。

巨大な教師フォルダ全体から `count.bin` を作る場合、`bucket-count` は固定長の `.psv` / `.bin` をまとめ読みしながら count します。Dドライブなどで読み込み速度が波打つ場合は、応用編の説明を見て `--buffer-mb` と `--read-buffers` を調整してください。

---

次へ: [4. 検証を有効にする](4-validation.md)

詳しい調整や比較実験: [応用編](../advanced/)

前へ: [2. 教師データを用意する](2-data.md)
