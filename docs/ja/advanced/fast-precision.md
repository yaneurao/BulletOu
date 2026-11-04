# FT重みのFP16 forward（実験用）

`"sfnn_ft_fp16": true` は、学習時のFT forwardで使う重みをFP16に丸めます。FP32 master、勾配、optimizer状態、検証および `nn.bin` の保存形式は従来どおりです。`false` がデフォルトで、epoch別指定も可能です。例えば `{"epoch1": false, "epoch10": true}` とします。

`"sfnn_ft_fp16_grad": true` を併用すると、FT逆伝播の一時的な勾配もスケーリングしてFP16に丸めます。既定値は `false` で、FT重みのFP16設定が必要です。累積する勾配とoptimizer状態はFP32のままです。スケーリング後にFP16の範囲を超えた勾配は最大値に制限されます。batch 65,536・FT幅2,304では追加で約576 MiBのGPUメモリを使います。学習結果や収束が変わる可能性があるため、スクリーニング向けの実験機能です。

範囲外の勾配があった場合は、superbatchの終了時に個数を警告します。

現在はFT 2304幅の HalfKA_hm2、batch 1024以上、FT factorizerなし、FT BNなし、QATなしの直接学習に限ります。FP16勾配にはFT saturation penaltyを無効にする必要もあります。plateau schedule とworker modeには対応しません。tataraの `--all-optim` とは異なり、optimizer状態のFP16化やTF32は含みません。
