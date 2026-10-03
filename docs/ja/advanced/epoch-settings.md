# epochごとの設定変更

## L1 factorizerのaxis/shared切替

```json
"sfnn_l1_factorizer": {
  "epoch1": "axis",
  "epoch5": "shared",
  "epoch8": "axis"
}
```

epoch開始時に切り替えます。`axis → shared`では、axisの重み・biasを、その時点のalpha/count係数を使ってbucket個別重みへfoldします。Lookaheadのslow重みも同じ変換を行います。base/sharedのmomentum・velocityと更新stepは保持し、axisのmomentum・velocityは破棄します。浮動小数点の丸めを除き直後のforwardは維持しますが、切替後の更新則まで等価にはなりません。

`shared → axis`ではbase/sharedを保持し、axisの重み・bias・optimizer状態をゼロで追加します。過去に削除したaxisの復活や、個別重みからの再推定はしません。

非BN・dense L1のcuda-cpp SFNN通常学習に対応します。k3k3、progress8、複合archとも、そのarchに存在する軸だけを使います。progress8のみでは軸とbucketが一対一なので、bucket間共有の効果はありません。BN/compact L1およびworkerのepochスケジュールは未対応です。count gateが0で非ゼロ係数のaxisをfoldできない場合は、出力を黙って変えず停止します。

checkpoint再開でも保存済みaxisの有無に応じて必要な変換だけを行います。将来のaxis指定で初期epochにaxisを先行確保しません。`[L1 FACTORIZER]`に切替方向・optimizerの扱いを表示します。

grid searchでは共通JSONに指定してください。`--grid sfnn_l1_factorizer shared`等の明示指定はJSONのスケジュール全体を上書きするので、途中切替するときはそのgrid軸を外します。

SFNNの通常学習・`grid_search.py` の共通設定JSONでは、途中epochから数値・真偽値を切り替えられます。重み・optimizer stateは維持し、プロセス再起動は行いません。

```json
{
  "lr": {"epoch1": 0.000400, "epoch11": 0.000200},
  "lr_min": {"epoch1": 0.000050, "epoch11": 0.000030},
  "batches_per_update": {"epoch1": 1, "epoch11": 4},
  "sfnn_qat_l1": {"epoch1": true, "epoch11": false}
}
```

既存の学習設定に加える項目の例です。epoch 1～10は前半、11以降は後半の値です。`true`→`false`、`false`→`true`のどちらも可能です。

- 通常は`epoch1`必須です。4つのrevive設定のみ省略でき、最初の指定まではfalseです。キーは `epoch1`, `epoch2`, …（先頭ゼロ不可）。順序は問いません。
- 未指定epochは直前の値を引き継ぎます。従来の数値・真偽値だけの指定は全epoch共通です。
- `lr` / `lr_min` は各epoch内のLRスケジュールの開始値／下限値。stepの自動gammaも再計算します。
- `--resume` は再開先epochの値を使います。別runを `--initial-state` から開始する場合は、新runのepoch 1からです。
- 明示的なCLI値・grid軸の値は、その項目のepoch別設定全体を上書きします。
- 設定は起動時に読み込み、実行中のJSON編集は反映しません。
- epoch開始時に `[epoch settings]` で有効値を表示します。`grid_summary.csv` の条件列も各epochの値です。checkpointの `bulletou-settings.json` は元のスケジュールを保存します。

## 対応項目

### FT factorizerを途中epochからOFFにする

非BNの `SFNN_halfka2` では、次の指定でepoch 1～2をON、epoch 3以降をOFFにできます。正式名は`sfnn_ft_factorizer`、旧名`ft_factorizer`はaliasです。

```json
"sfnn_ft_factorizer": {"epoch1": true, "epoch3": false}
```

epoch 3の最初のbatchより前に、共有FT重みを個別FT重みへfoldします。Lookahead slow重みも別途foldし、共有行を削除します。個別FTのmomentum/velocityと更新回数は維持し、共有FTのmomentum/velocityは破棄します。実効重みは保持しますが、浮動小数点の加算順による微差はあり、その後のoptimizer更新はON時と等価ではありません。unitのリセットは行いません。

`--resume` でも同じJSONを使えます。ONのcheckpointからOFFのepochへ再開する場合は復元時にfoldし、すでにOFFのcheckpointなら再foldしません。OFF→ONは未対応です。BN、他のarch、worker、plateauでの途中切り替えも未対応です。切り替え時には色付きの `[FT FACTORIZER]` 行を表示します。

`grid_search.py` の共通JSONでも使えます。ただし `--grid ft_factorizer true` 等を明示するとJSONのepoch指定全体を上書きするため、そのgrid軸は外してください。実行中のJSON編集は反映しないので、設定変更後はcheckpointから `--resume` してください。

| 項目 | 意味 |
|---|---|
| `lr`, `lr_min` | 学習率の開始値・下限値 |
| `batches_per_update` | 1更新あたりの累積batch数 |
| `ft_factorizer` / `sfnn_ft_factorizer` | 非BN SFNN HalfKA2でON→OFFへ自動fold |
| `sfnn_qat_l1` | L1 QATの有効・無効 |
| `sfnn_bn_qat` | BN QATの有効・無効（BN層は最初から有効化。例：`{"epoch1":false,"epoch6":true}`） |
| `sfnn_freeze_l1` | L1の固定・解除 |
| `sfnn_l1_lr_mult` | L1の学習率倍率 |
| `sfnn_norm_loss_strength` | ノルム正則化係数 |
| `sfnn_saturation_penalty`, `sfnn_saturation_threshold` | 飽和penaltyの係数・閾値 |
| `optimizer_weight_clip` | 重みclip幅（0は無効） |
| `optimizer_weight_decay` | weight decay係数 |
| `bce_error_weight_k` | BCEの誤差重み付け係数（BCE有効時のみ） |

未対応項目のオブジェクト指定はエラーです。arch、batch size、教師データ、FT以外のfactorizer構造、loss種類などは途中切替できません。worker/tuning、direct-step smoke、plateauにも未対応です。

## bpu変更時の丸め

保存時に未反映の累積勾配を残さないため、batches/sbは**そのepochで有効なbpuだけ**で割り切れる数に切り下げます。将来のepochの設定は、現在のbatch数・LR周期・勾配更新境界に影響しません。

40M局面/sb・batch size 65,536・bpuをepoch1=1、epoch11=4とした場合、epoch1〜10は610 batches/sb（39,976,960局面）、epoch11以降は608 batches/sb（39,845,888局面）です。max_epochs=5なら、epoch11の指定による丸めは一切ありません。

教師shuffleの窓幅は起動時に有効な設定で決まり、同じ実行中には変更しません。将来のbpuを参照して窓幅を変えることもありません。設定全体の構文検査と実行予定の作成は起動時に行いますが、将来の設定値を現在の学習計算に適用しません。

[Grid search](grid-search.md) / [English](../../en/advanced/epoch-settings.md)
## epoch開始時のunit再初期化

`sfnn_l1_revive`, `sfnn_l1_revive_zero`, `sfnn_l2_revive`, `sfnn_l2_revive_zero`
は、そのepochでtrueなら最初のbatchの学習前に判定・処理します。単一のtrueは毎epoch適用されます。

```json
{
  "sfnn_l1_revive": {"epoch3": true, "epoch4": false},
  "sfnn_l1_revive_zero": {"epoch3": true, "epoch4": false},
  "sfnn_l2_revive": {"epoch3": true, "epoch4": false},
  "sfnn_l2_revive_zero": {"epoch3": true, "epoch4": false}
}
```

この例はepoch3だけに適用します。この4項目に限りepoch1を省略でき、最初の指定まではfalseです。epoch4:falseを省略するとepoch3以降の毎epochで適用します。

L1→L2の順に、その時点の重みを使って教師16batchで再判定します。全unitの無条件リセットではありません。既存の判定条件（bucket内1024局面以上・サンプル全件で定数）と初期化方式は変更していません。

epoch途中からresumeした場合は重複処理せず、次のepoch開始を待ちます。前epoch末尾からresumeする場合は実施します。保存前に中断して古いcheckpointへ戻った場合は再判定します。FT factorizer切り替えやBN QAT設定変更が同時にある場合は、それらの適用後に処理します。warmup epoch0はepoch1設定を使用します。

L1はBNなし、L2はBNなしまたは校正済み・統計固定BN QATに対応します。workerでの対応範囲は従来どおり未対応です。校正・処理履歴は連番CSVとstdoutの[REVIVE] epoch=N START/ENDに残します。
