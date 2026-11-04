# FT重みのFP16 forward（実験用）

`"sfnn_ft_fp16": true` は、学習時のFT forwardで使う重みをFP16に丸めます。FP32 master、勾配、optimizer状態、検証および `nn.bin` の保存形式は従来どおりです。`false` がデフォルトで、epoch別指定も可能です。例えば `{"epoch1": false, "epoch10": true}` とします。

現在はFT 2304幅の HalfKA_hm2、batch 1024以上、FT factorizerなし、FT BNなし、QATなしの直接学習に限ります。plateau schedule とworker modeには対応しません。精度や収束が変わる可能性があるため、スクリーニング等で比較してから利用してください。これはtataraの `--all-optim` のFT重み部分だけであり、FT勾配やoptimizer状態のFP16化、TF32は含みません。
