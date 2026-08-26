# 3. 学習を走らせる

<a href="../../en/tutorial/3-train.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

このページでは、用意した教師データから `nn.bin` を作るところまでを扱います。

前のページ: [2. 教師データを用意する](2-data.md)

## 3.1 ビルドする

CUDA C++ backend を使う場合は、次のようにビルドします。

```powershell
cargo build --release --features cuda-cpp-backend --example bulletou
```

実行ファイルは次の場所にできます。

```text
.\target\release\examples\bulletou.exe
```

## 3.2 最小の設定ファイル

BulletOu フォルダに `bulletou-settings.json` を作ります。

```json
{
  "backend": "cuda-cpp",
  "arch": "NNUE_halfkp_256x2_32_32",
  "teacher": "teachers",
  "tag": "first-halfkp"
}
```

実行します。

```powershell
.\target\release\examples\bulletou.exe --settings-file .\bulletou-settings.json
```

`arch` は学習する評価関数の形です。`teacher` には教師ファイル、または教師ファイルが入ったフォルダを指定します。`tag` は実験名です。複数の実験を比べるときに見分けやすくなります。

JSON に書いた値は、コマンドラインで上書きできます。

```powershell
.\target\release\examples\bulletou.exe `
  --settings-file .\bulletou-settings.json `
  --tag another-test
```

## 3.3 出力先

学習結果は `checkpoints/` の下に保存されます。NNUE / SFNN では checkpoint ごとに `nn.bin` と `state.bin` ができます。

```text
checkpoints/
  NNUE_HALFKP-NNUE_halfkp_256x2_32_32-first-halfkp/
    0001/
      nn.bin
      state.bin
```

`nn.bin` は、やねうら王に読み込ませる評価関数ファイルです。

`state.bin` は、BulletOu が学習を再開するためのファイルです。再開する可能性がある最新 checkpoint では削除しないでください。

## 3.4 短く動作確認する

まずは小さな設定で、読み込み・学習・保存が通ることを確認します。

```json
{
  "backend": "cuda-cpp",
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

`pos/s` は学習速度の目安です。保存や validation の時間は学習速度から除かれます。

accuracy / loss を学習中に見るには、次のページで validation 用局面を指定します。

## 3.6 bucket 数が多い SFNN

`hand1024_k3k3_progress4` のように bucket 数が多い SFNN では、ほとんど出現しない bucket の個別成分が不安定になることがあります。その場合は、教師データから bucket の出現回数を数えた `count.bin` を作り、学習時に指定します。

```powershell
--sfnn-bucket-counts D:\BulletOu-snapshots\counts\count.bin
```

`--sfnn-bucket-counts` を指定し、かつ SFNN factorizer が有効な場合、BulletOu はデフォルトで bucket 固有 residual を count に応じて弱めます。出現回数が少ない bucket は共有成分に寄せ、十分に出現した bucket は個別成分を使いやすくします。

この挙動を止めたい場合は、次のように指定します。

```powershell
--sfnn-residual-count-gate-confidence 0
```

axis / pair factorizer の行も count に応じて弱めたい場合は、必要に応じて次のように指定します。

```powershell
--sfnn-axis-count-confidence 1.0 `
--sfnn-pair-count-confidence 1.0
```

`progress4` など `progressN` を含む architecture では、progress bucket を決めるために、`bucket-count` 実行時に同じ architecture の `nn.bin` を指定します。

```powershell
.\target\release\examples\bulletou.exe bucket-count `
  --teacher D:\sojoteam_datasets `
  --arch SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4 `
  --nn-bin D:\path\to\nn.bin `
  --output D:\BulletOu-snapshots\counts\count.bin
```

`count.bin` を使って追加学習する場合は、count を作ったときの progress 判定と学習中の progress 判定を揃えるため、通常は progress を固定します。

```powershell
--sfnn-freeze-progress
```

`count.bin` の作り方、式、読み込み buffer の調整は [応用編: SFNN factorizer](../advanced/sfnn-factorizer.md) を参照してください。

## 3.7 population search で決まった値を使って通常学習する

population search で調整した `tuning-settings.json` の `parameters.current` だけを使い、population search の候補探索は行わずに通常学習を続けたい場合があります。その場合は `tuning.enabled` を `false` にします。

```json
"tuning": {
  "enabled": false
}
```

この状態で `tuning_parameters.py` を起動すると、runner は `bulletou.exe` を 1 回だけ起動します。candidate 生成、worker cache、snapshot 保持は行いません。`parameters.current` を `--sfnn-factorizer-alpha` と count confidence オプションに変換して渡すだけなので、メモリ面のオーバーヘッドは `bulletou.exe` を単体で動かす場合とほぼ同じです。

```powershell
python .\tuning_parameters.py `
  --settings-file D:\BulletOu-snapshots\settings\tuning-settings.json `
  --resume
```

このモードでは、`superbatches` は `trial_sbs`、`max_epochs` は `generations` から runner が補います。`validation_rate` と `quantized_validation_rate` は `tuning-settings.json` の `tuning` に書いた値を使います。`lr` や `save_rate` などは `bulletou-settings.json` に書きます。

`recommended-parameters.json` の読み方や推奨値の算出式は、[応用編: 固定長 trial によるパラメーター調整](../advanced/parameter-tuning.md) を参照してください。

---

次へ: [4. validation を有効にする](4-validation.md)

詳しい調整や比較実験: [応用編](../advanced/)
