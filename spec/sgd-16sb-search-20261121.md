# SGDの16 sb条件比較（2026-11-21）

保存済み `0003-0324sb` のstateとdataloader位置から、試行ごとに独立して9,760更新（batch 65,536、16 sb相当）を実行する。本学習を再開せず、対局は行わない。出力先・設定・ログは `C:\shogi\codex-qacc-20261121`。検証は921,060局面、正解率の対象は決着局面872,416件。

基準はSGD momentum=0.95、lr=lr_min=0.0005、FT FP32、FT/L1/L2/L3 QATオン、L1 pair factorizer・FT factorizerおよびL1実効clip有効。入力batchの先頭・610更新ごとの全体指紋と、各batchの標本の累積指紋を保存する。全batch全要素の同一性を保証するものではない。実行間に揺れがあるため、各結果は独立した16 sb学習終了時の1回の検証であり、sb平均や単独試行による確定的な優劣ではない。

**現在の本学習設定への適用上の注意:** 上記のL1倍率比較はFT FP32・FT QATオンであり、2026-11-21に実行中だった本学習のepoch 3（all_optimオンによるFT FP16・FT QATオフ）とは異なる。後者でのL1倍率0.5対1.0の16 sb比較は実施しておらず、本学習でも同様の優位があるとは結論できない。

| 条件 | GPU proxy qacc | CPU整数qacc | GPU proxy qloss | CPU整数 train loss | 備考 |
|---|---:|---:|---:|---:|---|
| 基準 lr=0.0005 | 59.73927% | 59.69709% | 0.72454196 | 0.72493601 | `full_ft_qat-9760-16sb-exact-baseline.log` |
| lr=0.001 | 59.61124% | 59.60161% | 0.72488350 | 0.72524935 | `full_ft_qat-9760-16sb-lr001.log` |
| lr=0.00025 | 59.67990% | 59.67875% | 0.72434318 | 0.72468191 | `full_ft_qat-9760-16sb-lr00025.log` |
| L2重み100倍、bias実効1倍 | 59.66271% | 59.64517% | 0.72406977 | 0.72437620 | `full_ft_qat_l2w100-9760-16sb-exact-l2x100.log` |
| FT QATオフ、FT FP32 | 59.53960% | 59.54866% | 0.72791356 | 0.72827429 | `full-9760-16sb-ftqat-off.log` |
| lambda=0.5 | 58.30281% | 58.17500% | 0.67794180 | 0.67840278 | FT unit上限最大99.9993%；教師混合比が異なるためlossは他条件と直接比較不可。 `full_ft_qat_lambda05-9760-16sb-lambda05.log` |
| 基準再実行 lr=0.0005 | 59.63520% | 59.62843% | 0.72403961 | 0.72439456 | `full_ft_qat-9760-16sb-exact-baseline-repeat.log` |
| L1学習率倍率0.5 | 59.70557% | 59.70890% | 0.72466338 | 0.72501195 | `full_ft_qat-9760-16sb-l1lr05.log` |
| L1学習率倍率0.5 再実行 | 59.72850% | 59.70259% | 0.72348642 | 0.72382098 | `full_ft_qat-9760-16sb-l1lr05-repeat.log` |

## 実checkpoint・実バッチの勾配検証

`0003-0324sb/state.bin` と同じdataloader位置、batch 65,536、FT FP32、FT/L1/L2/L3 QAT有効、SGD設定で1バッチのforward/backwardのみを実施し、optimizer更新は実施しない。batchの勾配をCPUで独立に集計し、GPU結果と比較した。L1/L2の局面別勾配が非ゼロだったのはそれぞれ1,032,154/1,048,576、2,045,216/4,194,304。L2の0は主にclampの非活性部を含むが、その割合だけで原因を断定しない。

BCE出力勾配の最大絶対誤差4.24e-12、L2活性値が0と1の間にある1,943,636件についてL3→L2勾配は最大絶対誤差1.23e-12。先頭512局面のL2→L1入力勾配は最大絶対誤差1.56e-12。L1 bucket bias全要素の最大絶対誤差2.32e-10（CPU値の絶対最大7.40e-4）、shared bias 5.44e-9（同1.53e-3）、axis bias 9.88e-10（同4.20e-3）。L2 bucket bias全要素は1.19e-10（同2.16e-4）。最多bucket 1233（773局面）のL2重み全1,920要素は9.14e-12（同2.05e-5）、同bucketのL1残差重みから4入力特徴×16出力を選んだ64要素は8.66e-14（同1.04e-6）、L1 shared重みの同64要素は3.20e-11（同1.72e-5）。パラメータ集計はGPUが計算した各局面の中間勾配とforward値をCPUで集計したもので、すべてのforward/backwardをCPUで独立検証した意味ではない。L1重み全要素と他batchを網羅した検証でもない。ログは `full_ft_qat-1-real-gradient-ftqat.log`。

L1活性化については、丸め前のbias端数を省いた近似で、clamp境界から離れた938,287件を照合した。最大絶対誤差1.31e-9、期待値の絶対最大1.63e-5。これはbias端数を無視した近似であり、境界近傍を含む厳密な照合ではない。FT QATオフ・FT FP32、および `all_optim=true`・FT QATオフでも同じ実batchを照合し、L3→L2の最大絶対誤差はそれぞれ1.14e-12・1.32e-12、L2→L1入力は1.40e-12・1.57e-12。L1/L2のbiasと選択した重みの集計も一致した。ログは `full_ft_qat-1-real-gradient-ftqat-activation.log`、`full-1-real-gradient-ftqat-off.log`、`half-1-real-gradient-alloptim.log`。

基準条件の整数qaccは初回59.69709%、再実行59.62843%と0.06866ポイント異なり、今回試したlr 0.00025やL2倍率100との1試行差と同程度。上位を安定した優位とは判定できない。lambda=0.5はユーザーの禁止指示前に実行した履歴であり、今後は `spec/experiment-guidelines.md` のとおり混合比を変更しない。

L1学習率倍率0.5の2回は整数qacc 59.70890%・59.70259%（平均59.70575%）。基準2回は59.69709%・59.62843%（平均59.66276%）。それぞれ2試行のみの同一データ・checkpointであり、平均差0.04298ポイントが一般的な優位性を示すとは限らない。L1倍率0.5のFT unit上限最大は67.7582%・64.9803%、L1 unit上限最大7.3111%・7.3924%、L2 unit上限最大はいずれも0.5686%。入力の全体指紋を各sb末尾と初回batchで、標本の累積指紋を全batchで比較した結果、この4実行は一致した。前記のとおり未標本の入力完全一致は証明していない。

診断の再現は実験用設定で `cuda_cpp_train_steps=1`、`cuda_cpp_skip_final_output=true`、L1/L2/L3 QAT有効、環境変数 `BULLETOU_SGD_REAL_GRADIENT_DIAGNOSTIC=1` を指定する。1バッチの勾配を計算して比較し、optimizer更新は意図的に省略する。CPU比較は所定の許容誤差を超えるとエラーとなる。release版の診断をFT FP32 QATオンと `all_optim=true` で再実行し、いずれも `CPU comparisons = ok` を確認。ログは `full_ft_qat-1-real-gradient-checked.log` と `half-1-real-gradient-checked-alloptim.log`。SGD関連の明示GPUテスト15件も成功した。

## 2026-11-21 稼働中設定の静的確認

読み取り時の本学習はepoch 3、`all_optim`はepoch 1〜5で有効。設定ファイル上のFT個別フラグがfalseでも `all_optim` のscheduleとのORで4つすべてtrueとなる。FT QATはfalse、L1/L2/L3 QATはtrue、L1倍率は0.5。epoch 6からFT半精度4項目はfalseになる設定で、FT QATは自動的にオンにはならない。読み取り確認時点のsummary最新行はepoch 3 sb 467、lr_start/end=0.0005、lambda=1.0。動作中のプロセスは変更していない。

step LRはepochごとに再始動する。設定が変わらない場合、epoch 4冒頭の基本lrは0.0044であり、この時点でL1倍率0.5によりL1のlrは0.0022となる。epoch 3 sb 467時点のL1 lr 0.00025の8.8倍。LR上昇自体は実装上の仕様であり、不具合や学習への影響はこの静的確認だけでは判断できない。

現在のbucketは `(king_pair, progress)` の441×8通り。pair factorizerのking-progress成分も同じ組を添字に取るため、bucket残差との対応は一対一。係数1・残差ゲートなしのとき、lossから来る同じ座標のbucket残差とking-progress pairの勾配は、浮動小数点の集計順序を除き同じになる。両者のmomentum・重み制約・projectionの履歴は異なりうるため、実際の更新量が常に2倍になるとは限らない。shared、両玉axis、progress axisもfolded L1を同時に変える。L1倍率0.5はこれら全てのL1座標に適用される。設計上の重複であり、それだけを実装バグや今回の停滞原因と断定できない。

## epoch境界のSGD momentum

ユーザーは、step LRがepoch冒頭で戻る動作自体は意図しているが、前epochからのmomentum全体に上昇後のlrが掛かる点を問題と判断した。その後の指示で修正を実装した。方式、テストと未検証項目は `sgd-static-audit-20261121.md` の修正作業を参照。稼働中の学習は再起動しておらず、旧プロセスには未反映。
