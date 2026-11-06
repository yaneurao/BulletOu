# FTのFP16高速化（実験用）

`"sfnn_ft_fp16": true` は、学習時のFT forwardで使う重みをFP16に丸めます。FP32 master、勾配、optimizer状態、検証および `nn.bin` の保存形式は従来どおりです。`false` がデフォルトで、epoch別指定も可能です。例えば `{"epoch1": false, "epoch10": true}` とします。

`"sfnn_ft_fp16_grad": true` を併用すると、FT逆伝播の一時的な勾配もスケーリングしてFP16に丸めます。既定値は `false` で、FT重みのFP16設定が必要です。累積する勾配とoptimizer状態はFP32のままです。スケーリング後にFP16の範囲を超えた勾配は最大値に制限されます。batch 65,536・FT幅2,304では追加で約576 MiBのGPUメモリを使います。学習結果や収束が変わる可能性があるため、スクリーニング向けの実験機能です。

範囲外の勾配があった場合は、superbatchの終了時に個数を警告します。

さらに `"sfnn_ft_fp16_act": true` を指定すると、FTのCReLU後の活性値をFP16で保持し、その丸め後の値をpairwise計算と逆伝播に使います。既定値は `false` で、上記2項目を有効にする必要があります。1に近い活性値が1に丸められると、そのユニットのCReLU逆伝播は飽和として扱われます。batch 65,536・FT幅2,304では、FP16勾配用とは別に約576 MiBのGPUメモリを使います。

`"sfnn_ft_fp16_moments": true` を併用すると、FT重みのoptimizerのmomentumとvarianceをスケーリングしてFP16で保持します。既定値は `false` で、`sfnn_ft_fp16` が必要です。tataraと同じくmomentumに2²⁸、varianceに2⁴⁰を掛けて保存します。丸め・小さい値のゼロ化・範囲を超えた値のクリップにより、学習結果はFP32と異なります。範囲超過はsuperbatchごとに警告します。FP32 masterとslow weightsは維持し、checkpoint保存時にはmomentumとvarianceをFP32に戻すため、設定を変更して再開できます。FT幅2,304・入力73,305では追加で約644 MiBのGPUメモリを使います。

`--all-optim` または設定ファイルの `"all_optim": true` は、上記4つのFP16設定をまとめて有効にする実験用の省略指定です。`"all_optim": {"epoch1": true, "epoch5": false}` のように epoch 別にも指定できます。各設定は個別にも指定でき、`all_optim` が true の間は4項目を有効にし、false の間は各項目の設定に従います。CLI の `--all-optim` は常に4項目を有効にします。tataraの `--all-optim` と異なりTF32は有効にしません。

現在はFT 1536幅または2304幅の HalfKA_hm2、batch 1024以上、FT BNなしの直接学習に限ります。FT factorizerはON/OFFどちらにも対応し、ONの場合は個別重みと共有成分を合成してからFP16に変換します。L1/L2/L3のQATとは併用できますが、FT自体のQATとの併用はできません。FP16勾配にはFT saturation penaltyを無効にする必要もあります。plateau scheduleとworker modeには対応しません。これらの設定にはTF32は含まれません。
