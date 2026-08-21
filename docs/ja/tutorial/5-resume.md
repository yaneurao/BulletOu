# 5. 中断・再開

<a href="../../en/tutorial/5-resume.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

学習を途中で止めても、同じ設定で再実行すれば最新 checkpoint から続きます。

## 5.1 基本

同じ `tag` または同じ `output` で、同じ `bulletou-settings.json` を使ってもう一度実行します。`output_folder` で保存先ドライブを変えている場合は、その値も同じままにします。

```powershell
.\target\release\examples\bulletou.exe --settings-file .\bulletou-settings.json
```

`checkpoints/.../000N/state.bin` が見つかると、BulletOu が自動で読み込みます。

checkpointをDドライブに置きたい場合は、`output_folder` で親フォルダだけを指定できます。`tag` はそのまま使えます。

```json
{
  "arch": "SFNN_halfka2_1024_7_64_k3k3",
  "teacher": "D:/sojoteam_datasets",
  "output_folder": "D:/checkpoints",
  "tag": "sfnn-test"
}
```

この場合、保存先は次のようになります。

```text
D:\checkpoints\SFNN_HALFKA2-SFNN_halfka2_1024_7_64_k3k3-sfnn-test
```

```text
checkpoints/.../
  0001/
  0002/
  0003/   ← ここまで保存済み
  0004/   ← 再開後はここから保存
```

## 5.2 checkpointを掃除するとき

checkpointは大きくなりやすいので、古い保存点を削除してもかまいません。resumeに必要なのは、親フォルダ直下の `resume-config.txt` と、残したい最新checkpointフォルダの中にある3つのファイルです。

```text
checkpoints/.../
  resume-config.txt
  0074/
    state.bin
    learn.log
    dataloader_pos.txt
```

| ファイル / フォルダ | resumeに必要か | 説明 |
| --- | --- | --- |
| `0074/state.bin` | 必要 | 重みとoptimizer state。これがないとそのcheckpointから再開できません |
| `0074/dataloader_pos.txt` | 必要 | 教師データをどこまで読んだかの位置 |
| `0074/learn.log` | 必要 | そのcheckpointが正常に保存完了したことを判定するためのメタ情報 |
| `resume-config.txt` | 必要 | 同じ学習条件かどうかを確認するための設定記録 |
| `0074/nn.bin` | 不要 | やねうら王で使うための量子化済み評価関数。resumeには使いません |
| `summary-learn.log` | 不要 | これまでの検証結果を見るための通算ログ。resume自体には使いません |
| 古い `0001/`〜`0073/` | 不要 | `0074` から再開するなら削除できます |

たとえば `0074` から再開できればよく、古い `nn.bin` も不要なら、`0001`〜`0073` のフォルダは丸ごと削除できます。

```powershell
$exp = "C:\shogi\YaneuraOuWorks\BulletOu\checkpoints\実験フォルダ名"

Get-ChildItem $exp -Directory |
  Where-Object { $_.Name -match '^\d{4}$' -and [int]$_.Name -lt 74 } |
  Remove-Item -Recurse -Force -WhatIf
```

まず `-WhatIf` 付きで削除対象を確認してください。問題なければ、最後の `-WhatIf` を外します。

```powershell
Get-ChildItem $exp -Directory |
  Where-Object { $_.Name -match '^\d{4}$' -and [int]$_.Name -lt 74 } |
  Remove-Item -Recurse -Force
```

学習中に掃除するときは、いま書き込み中のcheckpointフォルダは削除しないでください。不安なら最新2個を残すと安全です。

## 5.3 設定を変えるとき

`lr`、`batch_size`、`superbatches` などを変えると、自動再開は止まります。

意図して同じ checkpoint から続けたい場合だけ、`--resume` を付けてください。

```powershell
.\target\release\examples\bulletou.exe `
  --settings-file .\bulletou-settings.json `
  --resume
```

新しい実験として始めたい場合は、`tag` を変えるのが安全です。

---

次へ: [6. 結果を確認する](6-result.md)

詳しい使い方: [応用編](../advanced/)

前へ: [4. 検証を有効にする](4-validation.md)
