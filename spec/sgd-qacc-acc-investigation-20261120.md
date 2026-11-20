# 2026-11-20: SGD学習中のacc/qacc乖離の調査

学習プロセスを停止せず、実行設定、ログ、保存済みcheckpoint、ソースを調査した。GPU診断・対局・学習条件変更は行っていない。

## 観測

2026-11-20に読み取ったsummaryでは、epoch 1 sb 324のacc/qacc/loss/qlossは57.6657% / 58.1602% / 0.812674 / 0.72274268。epoch 3 sb 324は55.3142% / 59.6038% / 1.663153 / 0.72667408。epoch 3 sb 374は55.2665% / 59.6619% / 1.686225 / 0.72535384。

通常のacc/lossは量子化されないFP32 masterのforward、qacc/qlossは量子化proxyのforward。訓練はL1/L2/L3のQATが有効で、これらとは異なる整数演算を含む。qvalidのGPU proxyは厳密なエンジン整数推論そのものではない。

## L1 skipの実効重み

`0001-0324sb`と`0003-0324sb`のstate.binをread-onlyで解析。L1実効重みをresidual + shared + 2 king axes + king-progress pair + progress axis（係数はすべて1）として復元。bucketは0から47刻みで76個、FT幅2304、L1出力16個。これは全bucketの頻度加重統計ではない。

| checkpoint | skipの範囲外要素 | skip全要素に対する割合 | skipの実効重み最大絶対値 |
|---|---:|---:|---:|
| epoch 1 sb 324 | 309 / 175104 | 0.17647% | 4.1727 |
| epoch 3 sb 324 | 1046 / 175104 | 0.59736% | 6.0067 |

他の15本にはこのサンプル上、範囲外はなかった。量子化i8の実効範囲は[-2,127/64]。範囲外と判定した要素だけでなく通常の丸め誤差も含むskip重みの量子化誤差RMSは0.03947→0.09350。個別のshared/axis/residualに重み制限を掛けても、その和が範囲内になる保証はない。範囲外の実効重みのため、FP32推論とQAT/量子化推論でskipの寄与が大きく異なりうる。

## 実検証局面での反実仮想（CPU）

固定検証データからseed 42でランダムに2048局面を抽出して8件ごとに1件、256局面を使用。引き分け20局面、score cap該当0局面、acc対象236局面。既存Rust実装でHalfKA_hm2特徴とk21k21_progress8 bucketを抽出し、checkpointからNumPyでFP32相当の推論を計算。2回目はL1のskip行のみ[-2,127/64]に制限し、その他の重み・特徴・演算は同じにした。量子化丸めや整数演算を含まないため、2回目はqvalidの再現ではない。少数標本であり全検証集合の値ではない。

| checkpoint | この標本の通常acc | skipのみclipしたacc | 通常loss | skipのみclipしたloss | 出力差RMS | 符号が変わった局面 |
|---|---:|---:|---:|---:|---:|---:|
| epoch 1 sb 324 | 57.20% | 60.17% | 0.7576 | 0.6944 | 0.3831 | 34/256 |
| epoch 3 sb 324 | 55.08% | 61.02% | 1.6274 | 0.7010 | 2.0689 | 88/256 |

両checkpointとも256局面すべてで範囲外のskip重みが正のFT入力と乗算されていた。FP32重みの加算順等はGPUと完全一致しないが、skip制限以外は同じCPU計算で比較している。skip実効重み範囲外が通常acc/lossの悪化に大きく寄与する強い証拠である。QAT中の勾配やqloss上昇の根本原因まで証明したものではない。

分析用コード・抽出した特徴は `C:\shogi\codex-sgd-study-20261118\speed-324\extract-validation.rs`、`skip-counterfactual.py`、`validation-features.txt`。モデルや学習設定を変更しない。

## 更新・損失

L1 QATは畳み込み後に量子化し、weight clippingを含むforwardに対してidentity STEを用いる。factorizerの各構成要素にoptimizerのclipを行っても、上記のように実効値は越えられる。`sfnn_l1_effective_weight_clip`はnone/sharedのみ対応し、今回のpair factorizerには適用できない。これは今回の乖離に関する具体的な設計上の問題だが、適切な修正方法と学習品質の改善は別途検証が必要。

BCEのforwardとgradientはそれぞれ `max(z,0)-z*t+log(1+exp(-abs(z)))`、`(sigmoid(z)-t)*600/340/batch` に対応し、検証側も同じBCEとtargetを使用している。通常loss増大を単なる計算表示バグとは判断できず、上記反実仮想で重みの値による影響を確認した。qacc上昇とqloss上昇が並行することだけでは、両指標の実装正否を断定できない。

stdoutで読み取れた範囲のFT FP16一時勾配clip警告はepoch 1で52回・合計59個、epoch 2で1174回・合計26354個、epoch 3（調査時点）で377回・合計37408個。発生は事実だが割合、影響度や上記L1 skip乖離の原因との因果関係は未測定。

step LRは仕様としてepoch冒頭でwarm restartする。epoch 2/3ともsb 1のlr_startは0.0044、sb 16は0.0005。ログ上だけでなく更新処理もepoch内のstepからLRを求める。この設定を提案した際、epochごとに高い学習率に戻る点を十分に伝えていなかった。これによる影響は未測定。

## 同一局面での訓練forwardとqvalid proxyのCPU近似

追加で同じ2048局面から4件ごとに選んだ512局面（acc対象486）について、上記のFP32/skip-only clipに加え、(a) qvalidの量子化重み+浮動小数点activationの近似と、(b) all_optimのFP16 FTとL1/L2/L3整数QAT forwardの近似を計算。CPU側の独立実装であり、GPU演算順、丸め境界、実際の全検証集合とは完全一致を保証しない。学習ステップは実行していない。

| checkpoint | FP32 acc/loss | skipのみclip acc/loss | qvalid proxy近似 acc/loss | 訓練QAT近似 acc/loss |
|---|---|---|---|---|
| epoch 1 sb 324 | 58.64% / 0.7750 | 60.29% / 0.7001 | 60.91% / 0.6983 | 60.08% / 0.7004 |
| epoch 3 sb 324 | 54.94% / 1.6807 | 61.11% / 0.7025 | 60.91% / 0.7116 | 59.88% / 0.7022 |

512件すべてで正のFT入力と範囲外skip重みの積があった。epoch 3ではskipのみclipにより183/512件で出力符号が変わった。proxyとQAT近似の平均絶対出力差は0.05795→0.11245。小さな標本だが、qloss proxyと学習時QATの差がありうるため、ログのqloss上昇をそのまま学習forwardのloss上昇と同一視できない。一方、この標本のQAT近似loss自体も0.7004→0.7022と上がっており、「学習時のlossは下がっている」とも言えない。コードは同じscratchフォルダの `training-proxy-comparison.py`。

## 2026-11-20: pair実効重みclipの修正とcheckpoint検証

`0003-0324sb/state.bin`と同じdataloader位置から、all_optim相当・SGD momentum 0.95・lr 0.0005で独立した1220ステップの比較。既存の学習フォルダとcheckpointは書き換えていない。

| 条件 | acc | loss | qacc | qloss | FT gradient clip |
|---|---:|---:|---:|---:|---:|
| 実効clipなし | 55.30332% | 1.66539073 | 59.62992% | 0.72643894 | 239 |
| 実効clipあり | 59.77859% | 0.72233951 | 59.63027% | 0.72639596 | 240 |

pairを含む全factorizerの和が範囲内に入るよう、bucket残差を補正する。shared/axis重みとmomentumは保持。SGDにおいて実際に補正した残差のFP32丸め補償だけをゼロにする。個別残差clipを併用すると補正を次stepに破壊するため、実効clip有効時はL1残差の個別clipを外す。clipなし条件からの長期学習品質改善やqloss下降は未確認。GPUテストでpairの異なる係数・fast/slow・範囲内no-op・補償クリアを確認した。ログは `C:\shogi\codex-sgd-study-20261118\pair-clip-20261120`。

## 2026-11-20: nn.bin整数推論と評価尺度

本学習を停止したまま、保存済みnn.binについて921,060検証局面を同じ条件でCPU整数推論した。BCE、target offset 0/scaling 380、prediction offset 0/scaling 340、nnue2score 600、FV_SCALE 24。対局はしていない。

| checkpoint | engine尺度の正解率 | engine尺度のBCE | train尺度のBCE |
|---|---:|---:|---:|
| epoch 1 sb 324 | 58.1667% | 0.67845827 | 0.72326577 |
| epoch 3 sb 324 | 59.6151% | 0.67212898 | 0.72704655 |

ここでengine尺度は整数出力をFV_SCALE=24で除算後のValueを340で割る。train尺度は整数出力を8128で割った値を600/340倍する。両者の比は丸めを無視すると約1.772倍。従って、学習尺度loss上昇とエンジン尺度loss低下は同時に起こる。これは同一検証集合上の観測であり、訓練目的に対してどちらが適切かや棋力を証明しない。full logsは `C:\shogi\codex-sgd-study-20261118\pair-clip-20261120\exact-all-0001-0324sb.log` と `exact-all-0003-0324sb.log`。

また `quantized-test --mode gpu` の出力はtrain尺度proxyであるのに従来`loss_engine_scale`と表示され、engine-score-offsetや整数rounding設定を黙って無視していた。GPU結果ではengine尺度を利用不可と表示し、無視される非既定設定は拒否するよう修正。実際の学習qvalid自体はtrain尺度proxyなので、この表示修正は既存の学習値を変更しない。

## FT一時勾配のFP16範囲超過: 短期比較

2026-11-20、`0003-0324sb`から同じ位置・SGD設定・実効L1 clipで独立した1220 stepを比較。FT重みforwardとmomentumのFP16は両方有効。all_optim相当（FT一時勾配とactivationがFP16）はacc/qacc/loss/qloss 59.77859% / 59.63027% / 0.72233951 / 0.72639596、FT一時勾配とactivationを両方FP32にした条件は59.77939% / 59.62946% / 0.72235012 / 0.72632730。前者には範囲超過240個の警告、後者にはなし。学習所要は各87.483秒、94.707秒。各1回・約2 sbの比較であり、2条件変更が混ざるためclip単独の影響や長期的な安全性は不明。
