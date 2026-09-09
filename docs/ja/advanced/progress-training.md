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

`.pack` の終局マーカー後には評価対象の盤面がないため、最後の着手直前の局面を `1` とします。学習結果は bias なし・f64 形式の `progress.bin` に保存され、実行時には `0..255` の整数へ変換されます。

## ファイル形式と推論式

`progress.bin` は tatara / [YaneuraOu PR #326](https://github.com/yaneurao/YaneuraOu/pull/326) と同じ、**ヘッダーなしの `f64 little-endian[81][1548]`** です。125,388 個の重み、合計 **1,003,104 bytes** で、bias・hash・bucket 数は格納しません。index は `king_square * 1548 + BonaPiece`、両玉の視点から非玉駒の特徴を足します。

専用の `progress-train` が学習する式は次の通りです。独立した bias はありません。

```text
z = Σ w[active_KP_feature]
prediction = sigmoid(z)
loss = (prediction - i / (L - 1))²
```

通常の NNUE 学習・count・検証では、ファイルの各重みを `round(w * 65536)` で整数化します（i32 範囲外は clamp）。その和を、sigmoid に対応する固定閾値で `0..255` に変換します。閾値はソース側にあり、ファイルには入りません。

`nn.bin` にはこの分類器を**埋め込みません**。構造は `NNUE header → FeatureTransformer → 各 stack の network` で、progress 専用 hash も追加しません。checkpoint 保存時には同じフォルダへ `progress.bin` も出力します。これは実際に使用した整数重みを `q16 / 65536` の f64 として保存するため、読み直しても bucket 判定が変わりません。

探索で使うときは、外部分類器形式に対応したやねうら王と、arch の一致する `nn.bin` / `progress.bin` をセットで使ってください。旧「nn.bin に分類器を埋め込む」実装には対応しません。BulletOu の `quantized-test` なども、progress 付き arch なら `nn.bin` と同じディレクトリの `progress.bin` を読みます。

### 旧形式から現在の学習を継続する場合

通常の読み込みは新形式だけに対応します。既存の進行度を維持して移行するための一度限りの変換は、次のスクリプトで行えます。同じ出力先なら元ファイルを `.q16-with-bias.bak` に退避します。

```powershell
.\convert_progress_format.ps1 `
  -InputPath C:\path\to\progress.bin `
  -OutputPath C:\path\to\progress.bin
```

単に bias を捨てると bucket が変わります。この変換では、平手で常に飛車が2枚（龍・持ち飛車を含む）あり、両視点で合計4項になることを利用して、対応する重みへ `bias_q16 / 4` を加えます。bias が4の倍数なら整数和が厳密に一致します。**飛車を落とした駒落ち局面には、この保存則は適用できません。** 学習し直す場合の `progress-train` には、この移行処理も固定の加算値もありません。

現在の旧 `state.bin` からの継続に限り、先頭 bias を含む進行度レコードを同じ方法で移行します。評価 NN の重み・その optimizer state・教師読み込み位置は変更しません。変換済み `--sfnn-progress-bin` のパスと通常の `--resume` を使えます。新しく保存した state の進行度レコードには bias を含めません。`export-progress-bin` の入力は `--state-bin` です（`nn.bin` からの抽出は廃止）。

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
| `--output` | 出力するヘッダーなし f64 `progress.bin` |
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
--sfnn-progress-bin C:\path\to\progress.bin
```

通常の SFNN 学習では、この分類器は常に固定です。評価値の loss では更新しないので、count を集計したときと SFNN を学習するときの progress bucket 判定が一致し、validation の cache も再利用できます。分類器を学習し直したいときは、`progress-train` を別途実行してください。

HalfKA2 / KA2 の学習では、CPU のデータ準備スレッドが盤面の decode と同時に進行度 bucket を計算します。GPU に渡す前に bucket が確定するため、進行度用の特徴番号を局面ごとに保存したり、学習側で再計算したりする必要はありません。通常学習と worker の両方で自動的に使われ、追加オプションや追加の VRAM は不要です。
