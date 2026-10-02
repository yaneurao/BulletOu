# L2/L3の中心化

## L1の実効重み制限（QAT併用の比較用）

`--sfnn-l1-effective-weight-clip`（JSON: `"sfnn_l1_effective_weight_clip": true`）は、L1のfold済み重みを量子化可能範囲 `[-2, 127/64]` に制限します。**デフォルトOFF**。中心化が無効にする従来のoptimizer clipとは別機能です。

各optimizer更新・Lookahead更新・中心化の逆変換を終えた後に、`effective = residual + alpha*shared` を制限し、`residual = clamp(effective) - alpha*shared` とします。sharedは保持し、fastとslowそれぞれのsharedを使って両方の個別重みを補正します。範囲内の値は変更しません。浮動小数点の加減算に伴う微差はあり得ます。

- 対象はL1重みのみ。FT/L2/L3、bias、momentum・分散推定、STEは変更しません。biasによる相殺も行いません。
- cuda-cpp、dense L1、factorizer none/shared、update-scope=allに対応。axis/pair・count gateは非対応。QATや中心化は必須ではありません。
- BPU>1ではoptimizer更新時だけ適用します。L1 freeze（LR倍率0）中は適用しません。
- resume直後の読み込みだけでは補正せず、最初のL1 optimizer更新後から適用します。既存ファイルを直接書き換えません。
- checkpoint/nn.bin形式は変更なし。追加の大型VRAMバッファやCPUへの重み転送はありません。
- epoch指定（例：`{"epoch1": false, "epoch2": true}`）にも対応。起動時表示とresume signatureへ記録します。

共通設定でQAT・中心化を揃えて、別のgridフォルダで比較できます：

```powershell
python .\grid_search.py `
  --settings-file <共通設定.json> `
  --output-folder <新しい比較フォルダ> `
  --grid sfnn_l1_effective_weight_clip false true
```

acc/qaccの差を減らすだけでは成功とは言えません。量子化側の精度・飽和率・棋力で評価してください。optimizer履歴は保持されるため、checkpoint途中からONにする実験と、最初からONの学習は異なります。

## L1中心化のA/Bテスト

`--sfnn-l1-center`（JSON: `"sfnn_l1_center": true`、デフォルトOFF）で、L1にも同じoptimizer座標の中心化を適用できます。L2/L3中心化とは独立しており、併用も可能です。

```powershell
python .\grid_search.py `
  --settings-file <共通設定.json> `
  --output-folder <新しい比較フォルダ> `
  --grid sfnn_l2_l3_center true `
  --grid sfnn_l1_center false true
```

L1に入力されるFT結合特徴の平均をGPU上で求め、bucket個別重みとshared重みに適用します。bucket別平均ではなく、更新対象の全バッチにわたる共通平均です。BPU>1、L1 QATに対応します。dense L1・factorizer none/sharedに限定し、その他の併用条件は下記と同じです。FT自身の中心化・出力の正規化・飽和率への罰則ではありません。飽和率低下は保証されないため、A/Bで確認してください。

L1の追加GPU作業領域はFT幅1024で約132 KiBです。重み全体のCPU転送はしません。forwardとnn.binは従来のfold済み形式です。epoch指定によるON/OFFとresume時の切替も可能ですが、optimizer状態はリセットしません。

GPU平均計算は列方向だけでなく行方向も並列化します。L1のbucket個別・sharedの座標変換は1カーネルにまとめ、更新前後それぞれの起動回数を削減しています。平均を取る局面の間引きはしません。加算順序の変更による浮動小数点の微差はありますが、中心化の式・学習率・optimizerの定義は変えていません。

中心化処理では、BPU=1の不要な平均値の加算・ゼロ初期化・コピーを省き、BPU>1でも平均の確定を1カーネルにまとめています。速度改善量は環境と構成によるため、実測で比較してください。A/Bでは両条件とも `optimizer_weight_clip: 0` にして、clipの有無まで変わらないようにしてください。

`--sfnn-l2-l3-center`（JSON: `"sfnn_l2_l3_center": true`）は、L2/L3のoptimizer更新を入力平均で中心化した座標で行う比較実験用オプションです。デフォルトは無効です。FT/L1を中心化する機能ではありません。

## 指定方法

既存の学習JSONに以下を指定します。他の学習条件はそのままです。

```json
{
  "sfnn_l2_l3_center": true,
  "batches_per_update": 1,
  "sfnn_factorizer": "shared",
  "optimizer_weight_clip": 0,
  "optimizer_weight_decay": 0,
  "sfnn_norm_loss_strength": 0,
  "sfnn_saturation_penalty": 0,
  "sfnn_factorizer_residual_decay": 0
}
```

CLIでは `--sfnn-l2-l3-center --batches-per-update 1 --optimizer-weight-clip 0` を指定します。他の対応条件も満たす必要があります。既存の設定ファイルを自動で変更することはありません。

比較用の共通JSONを上記の対応条件にした上で、次のgrid指定が使えます。

```powershell
python .\grid_search.py `
  --settings-file <共通設定.json> `
  --output-folder <比較結果フォルダ> `
  --grid sfnn_l2_l3_center false true
```

`--grid sfnn_l2_l3_center true` ならONだけを試します。ハイフン表記と `sfnn_l2_l3_center` の両方を指定できます。`grid_summary.csv`にもON/OFFが記録されます。epoch指定の `{"epoch1": false, "epoch2": true}` も使用できます。

## Glorot初期化との組み合わせ比較

初期化は層ごとに独立して指定できます。すべてcuda-cpp SFNNの新規学習用で、checkpointから読み込んだ重みは変更しません。JSONとgridの項目名はアンダースコアで統一します。

| JSON / grid項目 | デフォルト | trueの動作 |
|---|---|---|
| `sfnn_init_ft_glorot` | false | FT本体のGlorot uniform |
| `sfnn_init_l1_glorot` | false | L1個別重みのGlorot uniform |
| `sfnn_init_l2_glorot` | false | L2のGlorot uniform |
| `sfnn_init_l3_glorot` | false | L3のGlorot uniform |
| `sfnn_init_ft_shared_zero` | true | FT shared重みをゼロ初期化 |
| `sfnn_init_l1_shared_zero` | false | L1 shared重みをゼロ初期化 |

`sfnn_init_l2_l3_glorot` は廃止しました。L2/L3をそれぞれ指定してください。

```json
{
  "sfnn_init_ft_glorot": true,
  "sfnn_init_l1_glorot": true,
  "sfnn_init_l2_glorot": true,
  "sfnn_init_l3_glorot": true,
  "sfnn_init_ft_shared_zero": true,
  "sfnn_init_l1_shared_zero": true
}
```

Glorot半幅は `sqrt(6 / (fan_in + fan_out))` に `nnue_pytorch_init_scale` を掛けます。FT本体は特徴数とFT幅、L1はdense入力/出力数（skip出力を含む）、L2は `2*H1 + H2`、L3は `H2 + 1` を分母に使います。bucket数は掛けません。L2/L3は既存の層別初期化scaleも掛けます。biasは従来の設定のままです。

FT sharedをfalseにすると、同じFT初期化方式をshared行数のfan-inで適用し、別seedの乱数で初期化します。FT本体の乱数は変わりません。FT factorizerが無効ならshared指定は作用しません。L1 sharedのfalseは従来の一様乱数±0.01です。FT sharedは従来からゼロだったため、既定値trueを維持しています。

FT Glorotがfalseの場合、FT本体は従来の±`init_scale / sqrt(特徴数)`です。HalfKA2ではこの幅がFT量子化の半刻みより小さく、QATで全ゼロになることがあります。非BNのFT QATを有効にするときは、fold・量子化後のFT重みとbiasが全部ゼロなら警告して停止します。初期化やQAT設定を自動変更しません。この検査はQAT開始・切替・復元時に行い、毎batchのCPU読み戻しは行いません。

4通りの比較例：

```powershell
python .\grid_search.py `
  --settings-file <新規学習用の共通設定.json> `
  --output-folder <新しい比較結果フォルダ> `
  --grid sfnn_init_ft_glorot false true `
  --grid sfnn_init_ft_shared_zero false true
```

6項目すべて `--grid 項目名 false true` で比較できます（同時なら64条件）。初期化の比較ではcheckpoint入力を外してください。途中epochでの初期化変更は非対応です。これだけでConductor全体の条件が同一になるわけではありません。BulletOuはbucketごとに独立した乱数を使います。

## 中心化の計算

FTの継続飽和には別途[FT線形penalty](ft-saturation-penalty.md)をA/B実験できます。これは中心化と併用可能な、デフォルトOFFのactivation penaltyです。

L2とL3それぞれの入力平均ベクトル `c` をGPU上で計算します。bucket別平均ではなく、optimizer更新に使う全batch・全局面の平均です。`batches_per_update=4` なら4batch分の平均と累積勾配を使い、最後のbatchだけでは計算しません。更新直前に `beta = b + W*c`、`gW_center = gW - gb*c` とし、この座標で既存のoptimizer更新を行います。更新後に `b = beta - W*c` に戻します。Lookaheadのslow weight/biasも同様に変換します。

forward・validation・nn.binは従来どおり `W*x+b` です。BatchNormや入力の分散正規化ではありません。中心化した勾配に対してmomentum等を更新するため、通常のRangerと同じ学習アルゴリズムではありません。報告されるlossに追加の罰則は加えません。

平均計算用GPU領域は有効時に確保して再利用します。1024/7/64構成では約10KiBで、重み全体をCPUへ転送する処理はありません。

## 現在の対応範囲と再開

- cuda-cppのSFNN、`batches_per_update>=1`、`sfnn_update_scope=all`。bpuは比較対象と同じ値を使用できます。
- L1 factorizerはnone/shared。FT factorizerとL1 QATは使用可能。
- weight clipは中心化中は無効。省略または正の値なら警告を出して無効化し、学習を続行します。明示的に0なら警告は出しません。JSON自体は書き換えません。weight decay・Norm loss・saturation penalty・factorizer residual decayは0。
- bucket counts/count gates、axis/pair factorizerとの併用は現在非対応。
- L2入力幅・出力幅は各256以下、`bucket数 × L2幅` は65536以下。
- clip以外の非対応の組み合わせは引き続きエラーにします。中心化OFFのepochでは設定されたclip動作に戻ります。中心化だけを比較したい場合は、共通設定で `optimizer_weight_clip=0` としてください。

checkpointとnn.binの形式は変更しません。checkpoint保存時には通常のweight/bias表現に戻っています。ON/OFFを変えて再開できますが、optimizerのmomentum等は引き継ぎ、自動リセットしません。これは学習条件変更なので、新規学習でのA/B比較と同じではありません。再開時もJSONまたはCLIでON/OFFを明示してください。

旧調査用の `BULLETOU_EXPERIMENT_*` 環境変数を設定したシェルでは使用せず、通常の環境からこのオプションを指定してください。検証では飽和率・検証精度の改善を確認しましたが、長期学習の棋力改善を保証するものではありません。
