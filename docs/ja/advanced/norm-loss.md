# テンソル単位のNorm loss

`--sfnn-norm-loss-strength` はSFNNのパラメーターをL2ノルム1へ近づけます。デフォルト `0` は無効。JSONでは `"sfnn_norm_loss_strength": 0.0001`。係数は有限の非負数、目標は1固定です。強制的な正規化ではありません。

参照元はConductorではなく、[bullet-shogi Ranger21（commit 44941de4）](https://github.com/nodchip/bullet-shogi/blob/44941de4/crates/acyclib/src/trainer/optimiser/ranger21.rs) と、その `examples/shogi_layerstack.rs` の適用対象です。同例の係数は `1e-4`。**Norm lossだけの移植で、BulletOuのoptimizerをRanger21に置き換える機能ではありません。**

## 計算・対象

各テンソル全体で計算します。bucket／unit単位ではなく、sharedと個別重みをfoldしてから計算するものでもありません。

```text
n = sqrt(sum(w*w))
w *= 1 - learning_rate_norm * 2 * strength * (1 - 1/(n + 1e-7))
```

概念的には `strength*(||W||₂-1)²`。学習勾配へ加算せず、momentum/velocityと独立して重みを補正します。ゼロテンソルはゼロのままです。

| 対象 | タイミング |
|---|---|
| FT重み（FT factorizer含む） | 対象外 |
| FT bias | RAdam・clip後、Lookahead前 |
| L1個別／shared重み・bias | 同上 |
| L2重み・bias | 同上 |
| L3重み | 同上 |
| L3 bias | RAdam前 |

BulletOu独自のL1 axis/pairも格納テンソルごとにL1の規則で適用します。旧L2/L3 factorizerが更新対象として存在する場合も同じ層の規則です。

元実装のLR補正も使用します。更新前は層別倍率を反映したLR。更新後は以下です。βは現在のoptimizer設定、tはoptimizer更新回数です。

```text
learning_rate_norm = lr * sqrt(1-beta2^t)
                    / ((1-beta1^t) * sqrt((1+beta2)^2+beta2^2))
```

optimizer更新ごとに適用するため、bpu=4なら4batchごとです。freeze／更新対象外テンソルには掛けません。dirty更新時もNorm loss自体は未使用bucketを含む全テンソルへ掛かります。optimizer stateのdirty更新規則は変えません。

検証loss/qlossには正則化項を加算しません。GPUで計算し、CPUへの重みreadbackは不要。追加作業VRAMはcontextあたり約1KiBです。有効時は全テンソルの読み書きが増えるため速度への影響はあります。無効時は専用kernel・作業バッファを使いません。GPUの加算順の違いから参照実装とのbit一致は保証しません。

## Grid search

```powershell
cargo build --release --features cuda-cpp-backend --example bulletou
python .\scripts\grid_search.py `
  --settings-file .\bulletou-settings.json `
  --output-folder D:\BulletOu-snapshots\20260915\grid-norm-loss `
  --grid sfnn_norm_loss_strength 0 0.000001 0.00001 0.0001
```

4条件を比較し、係数は `grid_summary.csv` にも記録します。候補値は最適値の保証ではありません。共通JSONのtarget offset・LR・epoch数・初期state・教師位置は変更しません。`--dry-run` で計画だけ確認できます。既存target-offset gridとは軸が異なるため、新しい出力フォルダを使ってください。

追加学習の比較は共通JSONの `initial_state` と `initial_dataloader_pos` に同じ開始点を指定します。既存trialの係数変更をresumeで上書きすることはできません。新条件として比較してください。保存形式は変更せず、オプションのない旧checkpointは無効として従来どおり再開できます。
