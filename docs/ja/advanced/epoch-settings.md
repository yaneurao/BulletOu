# epochごとの設定変更

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

- `epoch1` は必須。キーは `epoch1`, `epoch2`, …（先頭ゼロ不可）。順序は問いません。
- 未指定epochは直前の値を引き継ぎます。従来の数値・真偽値だけの指定は全epoch共通です。
- `lr` / `lr_min` は各epoch内のLRスケジュールの開始値／下限値。stepの自動gammaも再計算します。
- `--resume` は再開先epochの値を使います。別runを `--initial-state` から開始する場合は、新runのepoch 1からです。
- 明示的なCLI値・grid軸の値は、その項目のepoch別設定全体を上書きします。
- 設定は起動時に読み込み、実行中のJSON編集は反映しません。
- epoch開始時に `[epoch settings]` で有効値を表示します。`grid_summary.csv` の条件列も各epochの値です。checkpointの `bulletou-settings.json` は元のスケジュールを保存します。

## 対応項目

| 項目 | 意味 |
|---|---|
| `lr`, `lr_min` | 学習率の開始値・下限値 |
| `batches_per_update` | 1更新あたりの累積batch数 |
| `sfnn_qat_l1` | L1 QATの有効・無効 |
| `sfnn_bn_qat` | BN QATの有効・無効（BN層は最初から有効化。例：`{"epoch1":false,"epoch6":true}`） |
| `sfnn_freeze_l1` | L1の固定・解除 |
| `sfnn_l1_lr_mult` | L1の学習率倍率 |
| `sfnn_norm_loss_strength` | ノルム正則化係数 |
| `sfnn_saturation_penalty`, `sfnn_saturation_threshold` | 飽和penaltyの係数・閾値 |
| `optimizer_weight_clip` | 重みclip幅（0は無効） |
| `optimizer_weight_decay` | weight decay係数 |
| `bce_error_weight_k` | BCEの誤差重み付け係数（BCE有効時のみ） |

未対応項目のオブジェクト指定はエラーです。arch、batch size、教師データ、factorizer構造、loss種類などは途中切替できません。worker/tuning、direct-step smoke、plateauにも未対応です。

## bpu変更時の丸め

保存時に未反映の累積勾配を残さないため、batches/sbは**そのepochで有効なbpuだけ**で割り切れる数に切り下げます。将来のepochの設定は、現在のbatch数・LR周期・勾配更新境界に影響しません。

40M局面/sb・batch size 65,536・bpuをepoch1=1、epoch11=4とした場合、epoch1〜10は610 batches/sb（39,976,960局面）、epoch11以降は608 batches/sb（39,845,888局面）です。max_epochs=5なら、epoch11の指定による丸めは一切ありません。

教師shuffleの窓幅は起動時に有効な設定で決まり、同じ実行中には変更しません。将来のbpuを参照して窓幅を変えることもありません。設定全体の構文検査と実行予定の作成は起動時に行いますが、将来の設定値を現在の学習計算に適用しません。

[Grid search](grid-search.md) / [English](../../en/advanced/epoch-settings.md)
