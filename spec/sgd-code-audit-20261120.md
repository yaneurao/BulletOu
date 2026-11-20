# SGD/QATおよびsnapshotの追加調査（2026-11-20）

本来の学習は起動しない。既存のcheckpointと検証データは読み取りのみ。

## 勾配蓄積中のsnapshot

`SfnnTrainStepRunner::snapshot_device` は未更新の累積勾配と累積batch数を保存しない。しかし従来は累積途中にも成功し、そのsnapshotをrestoreすると勾配が失われる。SGD、BCE、2 batchで1回更新する小さいGPU再現テストでは修正前に失敗。更新境界のみsnapshotを許可するよう修正後に成功。現在のユーザー設定は`batches_per_update=1`であり、これが現在のloss悪化に影響したという証拠はない。

## snapshotの頻度係数

従来のdevice snapshotにはresidual count gates、residual count decay、factorizer axis confidencesが含まれなかった。ゲート有効でsnapshotし、無効化して復元するGPUテストは修正前に失敗した。3種類の係数と各有効フラグをsnapshot/restoreに追加し、修正後に全てを確認した。現在のユーザー設定ではこれらの機能は使われていない。通常のstate.binの保存経路とは区別する。

SGD momentum・L1/L2/L3 QAT・FP16 FT momentum・L1実効clipを有効にした小規模GPU試験で、更新境界におけるdevice snapshotの復元後の重み、momentum、SGD丸め補償は、snapshotを挟まない連続実行と許容誤差内で一致した。これはstate.bin全体や実ネットワークを用いた長期再開の証明ではない。

## CPU整数推論のFT accumulator

やねうら王のFTは読み込み時にi16重み/biasを倍化し、i16 SIMD加算で累積する。一方BulletOuの `quantized_sfnn_ft_forward_sample` は従来i32累積値をそのままclampしており、i16の範囲外でエンジンと異なる。累積後にi16 wrapしてからclampするよう修正。範囲を超える合成値で回帰テストが修正前失敗・修正後成功。学習QAT/GPU proxyではなくCPU整数診断の修正。

実epoch 3 sb 324 checkpointのランダム512検証局面（2,359,296視点・unit）では、FT累積の最大絶対値460、i16範囲超過0。修正前後に921,060局面で整数推論した表示値はepoch 1 sb 324とepoch 3 sb 324の両方で同じだった（それぞれengine loss 0.67845827、0.67212898）。表示統計の一致は全局面の生出力一致を証明しない。修正前後ログは `C:\shogi\codex-sgd-study-20261118\pair-clip-20261120\exact-all-*` と `exact-i16-all-*`。

## 2026-11-20 追加: 実checkpointでの経路・設定切替

`0003-0324sb`、pair/L1実効clip、SGD momentum 0.95、lr 0.0005、元データ位置。独立実験フォルダでバッチ65536を用いた。元の学習やcheckpointは変更していない。

高速L1→FT逆伝播と`BULLETOU_FUSED_L1_FT_REFERENCE=1`の参照経路を各128 stepで比較。acc/lossは59.79315%/0.72204882と59.79326%/0.72205049、qacc/qlossは59.59072%/0.72645777と59.59118%/0.72644830、FP16勾配clipは両方32個。最終指標の比較であり、勾配の全要素一致は保証しない。関連する合成GPUテスト`fused_l1_ft_half_gradient_matches_separate_kernels`はFT幅2304で128-bit相当のhalf gradientの参照比較に通過している。

別の小規模実験ではall_optim=true→false→trueを各epoch8 batchで切り替え、pair実効clipを有効にしたまま3回のvalidationが完了した。qacc/qlossは59.57995%/0.72681290、59.56631%/0.72708732、59.53639%/0.72727007。短時間の動作確認であり長期品質は判定できない。3回目validation後に実験プロセスのみ停止し、最終checkpointは作成していない。ログ `C:\shogi\codex-sgd-study-20261118\pair-clip-20261120\switch-clip.log`。

追加で `BULLETOU_FUSED_QAT_FT_PACK_REFERENCE=1`、`BULLETOU_FUSED_L1_PACK_REFERENCE=1`、`BULLETOU_L1_TILED_FOLD_REFERENCE=1` を指定し、FT入力packの融合、L1畳み込みとpackの融合、およびL1 tiled foldを参照処理に戻して同じcheckpointから128 stepを実行。acc/loss 59.79372%/0.72204882、qacc/qloss 59.59107%/0.72644860、FT gradient clip 32。上記高速経路の最終指標に近い。ログの`vec4`は実験ランチャーのファイル名識別子であり、現行ソースに対応するSGD vec4実装はない。全勾配一致・長期品質を証明するものではない。

L1勾配集計について `l1_gather2_full_width_with_pair_matches_cpu` を追加。FT幅2304・出力16・8 bucket・king/progress/pair係数を持つ合成入力で、残差/shared/axisの重み・bias勾配および2回の累積をCPU参照と照合して通過。実checkpoint上の全勾配を直接照合したものではない。
