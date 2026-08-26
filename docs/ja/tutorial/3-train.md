# 3. 学習を走らせる

<a href="../../en/tutorial/3-train.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

このページでは、用意した教師データから、やねうら王で読み込める `nn.bin` を作るところまでを説明します。

前のページ: [2. 教師データを用意する](2-data.md)

## 3.1 まずビルドする

CUDA C++ backend を使う場合は、次のコマンドでビルドします。

```powershell
cargo build --release --features cuda-cpp-backend --example bulletou
```

実行ファイルは次の場所にできます。

```text
.\target\release\examples\bulletou.exe
```

## 3.2 最小の設定ファイル

最初の動作確認では、小さめの HalfKP NNUE が扱いやすいです。

BulletOu フォルダに `bulletou-settings.json` を作ります。

```json
{
  "arch": "NNUE_halfkp_256x2_32_32",
  "teacher": "teachers",
  "tag": "first-halfkp"
}
```

実行します。

```powershell
.\target\release\examples\bulletou.exe --settings-file .\bulletou-settings.json
```

`arch` は学習する評価関数の形です。`teacher` には教師ファイル、または教師ファイルが入ったフォルダを指定します。`tag` は実験名です。複数の実験を比較するときに見分けやすくなります。

JSON に書いた値は、コマンドラインで上書きできます。

```powershell
.\target\release\examples\bulletou.exe `
  --settings-file .\bulletou-settings.json `
  --tag another-test
```

## 3.3 出力先

学習結果は `checkpoints/` の下に保存されます。NNUE / SFNN では、保存された checkpoint ごとに `nn.bin` と `state.bin` ができます。

```text
checkpoints/
  NNUE_HALFKP-NNUE_halfkp_256x2_32_32-first-halfkp/
    0001/
      nn.bin
      state.bin
```

`nn.bin` は、やねうら王に読み込ませる評価関数ファイルです。

`state.bin` は、BulletOu で学習を再開するためのファイルです。再開する可能性がある最新 checkpoint では削除しないでください。

SFNN の `progress4` など、`progressN` を含む architecture では、checkpoint フォルダに `progress.bin` も保存されます。これは progress bucket を決める分類器だけを抜き出したファイルです。

## 3.4 短い動作確認

巨大な教師データで長時間回す前に、小さい設定で読み込み・学習・保存が通るか確認します。

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

## 3.5 画面に出るもの

学習中は、おおむね次のような行が出ます。

```text
[train] epoch 1  sb 1/1  this-sb=... pos  wall=...s  train=...s  pos/s=...
```

`pos/s` は学習速度の目安です。保存や validation の時間は、学習速度からは除外されます。

accuracy / loss を学習中に見るには、次のページで validation 用局面を指定します。

## 3.6 bucket 数が多い SFNN

`hand1024_k3k3_progress4` のように bucket 数が多い SFNN では、ほとんど出現しない bucket の個別成分が不安定になることがあります。その場合は、教師データから bucket の出現回数を数えた `count.bin` を作り、学習時に指定します。

```powershell
--sfnn-bucket-counts D:\BulletOu-snapshots\counts\count.bin
```

`--sfnn-bucket-counts` を指定し、SFNN factorizer が有効な場合、BulletOu はデフォルトで bucket 固有 residual を count に応じて弱めます。出現回数が少ない bucket は共有成分に寄せ、十分に出現した bucket は個別成分を使いやすくします。

この gate を止めたい場合は、次のように指定します。

```powershell
--sfnn-residual-count-gate-confidence 0
```

axis / pair factorizer の行も count に応じて弱めたい場合は、必要に応じて次のように指定します。

```powershell
--sfnn-axis-count-confidence 1.0 `
--sfnn-pair-count-confidence 1.0
```

`progress4` など `progressN` を含む architecture では、progress bucket を決めるために `progress.bin` を使います。`progress.bin` は progress 付き checkpoint を保存すると同じフォルダに自動保存されます。既存の checkpoint から取り出すこともできます。

```powershell
.\target\release\examples\bulletou.exe export-progress-bin `
  --arch SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4 `
  --state-bin D:\path\to\checkpoint\state.bin `
  --output D:\path\to\checkpoint\progress.bin
```

その `progress.bin` を使って count を作ります。

```powershell
.\target\release\examples\bulletou.exe bucket-count `
  --teacher D:\sojoteam_datasets `
  --arch SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4 `
  --progress-bin D:\path\to\checkpoint\progress.bin `
  --output D:\BulletOu-snapshots\counts\count.bin
```

`count.bin` を使って追加学習する場合は、count を作ったときの progress 判定と学習中の progress 判定を揃えるため、通常は同じ `progress.bin` を指定し、progress を固定します。

```powershell
--sfnn-bucket-counts D:\BulletOu-snapshots\counts\count.bin `
--sfnn-progress-bin D:\path\to\checkpoint\progress.bin `
--sfnn-freeze-progress
```

`--sfnn-progress-bin` を指定しない場合は、resume 元の `state.bin` に入っている progress parameter を使います。新規学習では scratch 初期化されます。`count.bin` と `progress.bin` は厳密な一致チェックをしません。実験のために仮の組み合わせで使うこともできます。

`count.bin` の作り方、式、読み込み buffer の調整は [応用編: SFNN factorizer](../advanced/sfnn-factorizer.md) を参照してください。

## 3.7 population search で決まった値を使って通常学習する

population search で調整した `tuning-settings.json` の `parameters.current` だけを使い、候補探索は行わずに通常学習を続けたい場合があります。その場合は `tuning.enabled` を `false` にします。

```json
"tuning": {
  "enabled": false
}
```

この状態で `tuning_parameters.py` を起動します。

```powershell
python .\tuning_parameters.py `
  --settings-file D:\BulletOu-snapshots\settings\tuning-settings.json `
  --resume
```

このモードでは、runner は `bulletou.exe` を1回だけ起動します。candidate 生成、worker cache、snapshot 保持は行いません。`parameters.current` を `--sfnn-factorizer-alpha` と count confidence オプションに変換して渡すだけなので、メモリ面のオーバーヘッドは `bulletou.exe` を単体で動かす場合とほぼ同じです。

このモードでは、`superbatches` は `trial_sbs`、`max_epochs` は `generations` から runner が補います。`validation_rate` と `quantized_validation_rate` は `tuning-settings.json` の `tuning` に書いた値を使います。`lr` や `save_rate` などは `bulletou-settings.json` に書きます。

`recommended-parameters.json` の読み方や推薦値の計算式は、[応用編: 固定長 trial によるパラメーター調整](../advanced/parameter-tuning.md) を参照してください。

---

次へ: [4. validation を有効にする](4-validation.md)

詳しい調整や比較実験: [応用編](../advanced/)
