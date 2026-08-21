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

## 3.2 最小の設定ファイル

最初は HalfKP NNUE で動作確認するのが簡単です。

BulletOu フォルダに `bulletou-settings.json` を作ります。

```json
{
  "arch": "NNUE_halfkp_256x2_32_32",
  "teacher": "teachers",
  "tag": "first-halfkp"
}
```

実行はこうです。

```powershell
.\target\release\examples\bulletou.exe --settings-file .\bulletou-settings.json
```

`arch` は学習する評価関数の形です。`NNUE_halfkp_256x2_32_32` は小さめで試しやすい構成です。

`teacher` には教師ファイル、または教師ファイルが入ったフォルダを指定します。

`tag` は実験名です。省略しても動きますが、複数回試すなら付けておくほうが出力を見分けやすくなります。

JSON に書いた値は、コマンドラインで上書きできます。

```powershell
.\target\release\examples\bulletou.exe `
  --settings-file .\bulletou-settings.json `
  --tag another-test
```

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

巨大な教師データでいきなり長時間回す前に、次のように小さめの設定にして動作確認できます。

```json
{
  "arch": "NNUE_halfkp_256x2_32_32",
  "teacher": "teachers",
  "positions_per_superbatch": 1000000,
  "superbatches": 1,
  "max_epochs": 1,
  "tag": "smoke-halfkp"
}
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

`progress4` など `progressN` を含む architecture では、進行度を計算するパラメーターも学習され、`nn.bin` の Progress section に保存されます。`count.bin` を作る場合は、進行度bucketを決めるために同じ architecture の `nn.bin` も指定します。

`progressN` 付きの architecture は、何も指定しない場合、学習中に進行度パラメーターも更新します。この経路では学習時に隣接bucketを使うため、通常のbucketよりかなり遅くなります。

進行度パラメーターをこれ以上動かさない段階では、再開時に次を指定します。

```powershell
--sfnn-freeze-progress
```

これを指定すると、進行度パラメーターを固定し、`nn.bin` に書き出すときと同じ hard bucket 判定で学習します。学習と validation cache の両方が軽くなります。最初から進行度パラメーターを学習したい場合は、このオプションは付けないでください。

巨大な教師フォルダ全体から `count.bin` を作る場合、`bucket-count` は固定長の `.psv` / `.bin` をまとめ読みしながら count します。Dドライブなどで読み込み速度が波打つ場合は、応用編の説明を見て `--buffer-mb` と `--read-buffers` を調整してください。

---

次へ: [4. 検証を有効にする](4-validation.md)

詳しい調整や比較実験: [応用編](../advanced/)

前へ: [2. 教師データを用意する](2-data.md)
