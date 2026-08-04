# 4. 中断・再開

<a href="../../en/tutorial/4-resume.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

学習を途中で止めても、同じ設定で再実行すれば最新 checkpoint から続きます。

## 4.1 基本

同じ `--tag` または同じ `--output` で、同じコマンドをもう一度実行します。

```powershell
.\target\release\examples\bulletou.exe `
  --arch NNUE_halfkp_256x2_32_32 `
  --teacher teachers `
  --tag first-halfkp
```

`checkpoints/.../000N/state.bin` が見つかると、BulletOu が自動で読み込みます。

```text
checkpoints/.../
  0001/
  0002/
  0003/   ← ここまで保存済み
  0004/   ← 再開後はここから保存
```

## 4.2 設定を変えるとき

`--lr`、`--batch-size`、`--superbatches` などを変えると、自動再開は止まります。

意図して同じ checkpoint から続けたい場合だけ、`--resume` を付けてください。

```powershell
.\target\release\examples\bulletou.exe `
  --arch NNUE_halfkp_256x2_32_32 `
  --teacher teachers `
  --tag first-halfkp `
  --resume `
  --lr 0.0001
```

新しい実験として始めたい場合は、`--tag` を変えるのが安全です。

---

次へ: [5. 結果を確認する](5-result.md)

詳しい使い方: [応用編](../advanced/)

前へ: [3. 学習を走らせる](3-train.md)
