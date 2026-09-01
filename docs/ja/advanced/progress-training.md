# 対局棋譜から進行度分類器を作る

<a href="../../en/advanced/progress-training.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

`progress4` や `progress8` を含む SFNN は、局面を序盤・中盤・終盤などに分けるために `progress.bin` を使います。このページでは、評価関数の loss とは分離して、完結した対局棋譜から進行度だけを学習する方法を説明します。

## 教師値

1局に、評価可能な局面が `L` 個入っているとします。先頭から `i` 番目の局面に与える教師値は次の通りです。

```text
target = i / (L - 1)
```

- 先頭局面: `0`
- 最後に評価可能な局面: `1`
- 途中の局面: 対局内での位置に応じて `0..1`

`.pack` の終局マーカー後には評価対象の盤面がないため、最後の着手直前の局面を `1` とします。学習結果は q16 の `progress.bin` に保存され、実行時には `0..255` の整数へ変換されます。

## `.pack` が必要な理由

この学習には、1局の開始位置と終了位置が必要です。そのため、対局境界を保持する YaneuraOu `.pack` を指定します。

PSV、HCPE、固定長 `.bin` は局面単位のデータです。シャッフル後のファイルから「この局面が対局の何割目か」を復元できないため、進行度教師には使えません。

ファイル名に含まれる数字は対局数として解釈しません。コマンドが `.pack` を走査し、実際の対局数と局面数を表示します。

## 実行例

```powershell
.\target\release\examples\bulletou.exe progress-train `
  --teacher C:\path\to\games.pack `
  --output C:\path\to\progress.bin `
  --epochs 5 `
  --batch-size 4096 `
  --lr 0.0002
```

設定の意味は次の通りです。

| オプション | 内容 |
|---|---|
| `--teacher` | 完結した対局を保持する `.pack` 1ファイル |
| `--output` | 出力する q16 `progress.bin` |
| `--epochs` | `.pack` 全体を学習に使う回数 |
| `--batch-size` | 1回の Adam update にまとめる局面数 |
| `--lr` | Adam の learning rate。内部倍率はかかりません |
| `--validation-game-stride` | N局ごとに1局を検証用へ回す。初期値は20、0なら検証なし |
| `--max-games` | 読む対局数の上限。0ならファイル全体 |
| `--overwrite` | 同名の出力ファイルを置き換える |

各 epoch の表示には MSE、MAE、先頭局面と最終局面の平均予測値、`progress4` / `progress8` の bucket 分布が含まれます。検証 MSE が最も低かった epoch のパラメーターを出力します。

## `progress4` と `progress8` で共用できる理由

`progress.bin` が出力する値は、bucket 数と無関係な `0..255` です。architecture ごとの bucket 番号は次の式で決まります。

```text
bucket = min(progress_0_255 * bucket_count / 256,
             bucket_count - 1)
```

そのため、1つの `progress.bin` を `progress4`、`progress8`、`progress16` などで共用できます。一方、`count.bin` は bucket 数を含む architecture ごとの出現回数なので、architecture を変えた場合は作り直します。

## count と SFNN 学習に使う

作成した分類器で `count.bin` を作ります。

```powershell
.\target\release\examples\bulletou.exe bucket-count `
  --teacher D:\teacher `
  --arch SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4 `
  --progress-bin C:\path\to\progress.bin `
  --output D:\counts\count.bin
```

SFNN の学習でも同じ分類器を読み込み、固定します。

```powershell
--sfnn-progress-bin C:\path\to\progress.bin `
--sfnn-freeze-progress
```

これにより、count を集計したときと SFNN を学習するときで progress bucket の判定が一致します。

