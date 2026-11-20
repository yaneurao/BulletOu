# SGD/QATの追加調査（2026-11-21）

本学習は実行せず、保存checkpointを読み取り専用で使用。独立実験の設定・ログ・補助コードは `C:\shogi\codex-qacc-20261121`。対局は行っていない。

## GPU proxyとCPU整数推論

同じ `nn.bin` と検証局面921,060件（決着872,416件）を使用。CPU整数推論はFV_SCALE=24、floor演算、BCE、target 0/380。CPU正解率はengine尺度で最終整数除算後の符号、GPUは学習時のproxy。CPU lossのpredictionはengine尺度でValue/340、train尺度で(raw/8128)*600/340とし、区別する。

| checkpoint | CPU整数正解率 | GPU proxy qacc | CPU engine loss | CPU train loss | GPU proxy qloss |
|---|---:|---:|---:|---:|---:|
| epoch 2 sb 1296 | 59.3708% | 59.4249% | 0.67317772 | 0.72723216 | 0.72691172 |
| epoch 3 sb 324 | 59.6151% | 59.6038% | 0.67212898 | 0.72704655 | 0.72667408 |

この324 sb間の正解率改善はCPU整数で0.2443ポイント、GPU proxyで0.1789ポイント。両方で改善したが差は同一ではない。保存済み3-324より後の未保存モデルについては、この測定から結論できない。CPUのログは `exact-full-0002-1296sb.log` と `exact-full-0003-0324sb.log`。

同じcheckpointの学習開始1更新後を保存せずにproxyと整数で評価すると、決着872,416局面のうち18,118局面（2.0767%）で符号が異なった。proxyのみ正解8,975、整数のみ正解9,143。両者の正解率差が約0.019ポイントでも、多くの符号差が相殺されている。この測定は `full_ft_qat-1-paired-smoke.log`。

同じcheckpointとseed 42の2,048検証局面で、CPU近似を使って演算を段階的に置き換えた。FT積の整数化で31/2,048局面、続くL1整数活性化で1/2,048、続くL2整数活性化で5/2,048の符号が変わった。段階ごとの件数は重複しうる。全置換・最終整数除算までとproxyの符号差は決着1,941局面中31局面。始点proxyの近似正解率58.37197%・train BCE 0.73318938は実GPU proxyの58.37197%・0.73318940と表示桁で一致し、終点整数の近似engine正解率58.01133%・train BCE 0.73401207も実CPU整数の58.01133%・0.73401207と一致した。ただし中間段階を含む個々の出力・全検証局面での一致は証明しない。FT積の丸めがこの標本の差に大きく寄与するという観測であり、proxyの計算バグと断定するものではない。コードとログは `trace_proxy_gaps.py` と `proxy-sample-2048-*.log`。

## 3,528 bucketでの勾配照合

`sgd_dense_qat_3528_bucket_gradients_match_scalar_ste` を追加。3,528 bucket、HalfKA_hm2のFT factorizer行、king/progress pairを有効にし、FT QATのオン・オフ両方でL1/L2/L3整数QATを実行。8局面・小さいdense層と、1,024局面・L1出力16・L2出力64の2形状を使い、複数bucket・bucket重複、L2の上限/下限にあるunitを含む合成入力について、GPU backwardのFT weight/bias、L1 residual/shared/axisのweight/bias、L2/L3 weight/biasを独立したCPU連続STE計算と照合し通過。後者はL1のbucket gather経路を通る。FT幅はいずれも4であり、実ネットワーク幅2304、batch 65536、実checkpoint全勾配の検証ではない。

## bucketの頻度とSGD

batch平均勾配をbucket別重みに適用する場合、bucket bの更新は、そのbatchでbが出現した局面の勾配和を**全batch数**で割る。そのため同じ条件付き平均勾配でも、平均更新量は出現割合 `p_b` に比例する。L1 sharedなど複数bucketで使う係数の勾配は、対応するbucketを合計する。SGDではこの差をoptimizer自体は自動的に打ち消さない。これは実装の正しさを前提にした式の性質で、今回の停滞原因と断定するものではない。

参考として、以前seed 42で抽出した検証2,048局面では702 bucketが出現し、最多bucketは128局面、出現1回のbucketは348個だった。これは検証標本の頻度であり、訓練全体の頻度ではない。全bucketの量子化変化率のみでは頻出bucketの変化が見えにくいため、FT QATを揃えた比較とbucket別traceを追加する。

## FT QATを揃えた16 sbと再実行差

epoch 3 sb 324の同じcheckpoint・dataloader位置、FT FP32・FT QAT有効、lambda=1、lr=0.0005、momentum=0.95、9,760更新。通常L2倍率では量子化値が開始・終了で異なるL2重み132/6,773,760、L3重み25/225,792。前回のL2重み倍率100（bias倍率は実効1）の同FT QAT条件ではそれぞれ13,573個、20個。二時点間に変化して戻った値は数えず、bucket頻度も加味していない。

通常倍率の16 sb実験は以前qacc 59.55806%、qloss 0.72486168、今回の再実行はqacc 59.81000%、qloss 0.72426689だった。設定JSONは出力先以外同一だが、後者は層別traceとverboseを有効にし、CPU整数検証が一部並行していた。入力batchの一致はまだ検証していない。0.25194ポイントの差の原因は未特定であり、前回のL2倍率100のqacc優位0.11198ポイントを確かな改善とは判断できない。

## 1更新と4 sbの再現性

同じcheckpoint・dataloader位置・FT QAT有効・同じ学習設定から、通常のpipelined uploadと、upload後にCUDA stream全体を同期する参照経路をそれぞれ2回実行した。1更新では4回の入力batch全体の指紋とqacc/qloss表示値は一致したが、FTやL1 shared/axis重みのFP32指紋は異なった。64要素ごとに抽出した比較では、通常upload同士の最大絶対差はL1 axisの7.28e-12、同期upload同士はFTの9.09e-13。抽出部分の値であり全要素の最大差ではない。

4 sbでは、通常uploadのqaccが59.66110%・59.63623%、同期uploadは59.51759%・59.62557%。同じ順でqlossは0.72506481・0.72465497・0.72527975・0.72475541。4実行とも1、610、1220、1830、2440更新目のbatch全体の指紋と、全batchから各128局面程度を抽出した累積指紋が一致した。これは未抽出データの全要素一致を証明しない。4 sb後の64要素ごとの重み標本では、通常upload同士のL1 shared最大差は1.09e-5、同期upload同士のL1 axis最大差は1.26e-5。同期しても結果差があるため、非同期uploadだけでは説明できない。

FT逆伝播のbias集計、bucket別位置リストの作成、L1 shared/axis勾配集計にはGPU atomic加算があり、同一入力でも加算順が変わりうる。これは微小な差の候補だが、上記qacc差への因果関係や他のraceがないことを証明したものではない。比較には同じbinを用い、長時間の正解率・損失は各1回の検証値である。

## bucket別の4 sb観測

FT QAT有効・通常倍率・4 sbの一実行で、bucket別に開始時と終了時の量子化値を比較した。検証2,048局面のbucket出現回数で重み付けすると、量子化値が変わった割合はL2重み0.00470%、L2 bias 8.459%、L3重み0.00916%、L3 bias 56.348%。各bucketの値を検証標本の出現数だけ重み付けしたものであり、訓練中の実際の頻度重みではない。

標本で最多のbucket 2816（128/2048局面）では、L2重み0/1,920、L3重み0/64、L2 bias 10/64、L3 bias 1/1が開始・終了時で異なった。biasの量子化刻みは1/8128、重みは1/64で、前者は127倍細かい。勾配量や分布も異なるため、この刻みの差だけで上記の割合を定量的に説明するものではない。CSVと集計コードは `full_ft_qat-2440-repro-pipeline-a-buckets.csv` と `compare_bucket_traces.py`。

## 層別倍率の4 sb比較と実整数推論

同一checkpoint、FT FP32・FT QAT有効で4 sb（2,440更新）学習し、終了時の同一重みをGPU proxyとCPU整数推論の両方で検証した。いずれも1試行であり、上記の再実行差に注意する。出力biasも含むL3倍率と、L2重みだけを100倍にしてbiasの実効倍率を1に保つ条件を比較した。

| 条件 | GPU proxy qacc | CPU整数 qacc | GPU proxy qloss | CPU整数 train尺度 loss |
|---|---:|---:|---:|---:|
| 通常倍率 | 59.67600% | 59.64987% | 0.72359437 | 未測定 |
| L2重み100倍 | 59.67451% | 59.63027% | 0.72377533 | 0.72424126 |
| L3重み・biasとも10倍 | 59.60058% | 59.56539% | 0.72481138 | 0.72521883 |

L2重みの量子化値が開始時と終了時で異なるものは、通常倍率の46個に対し100倍で4,242個。検証2,048局面のbucket頻度を用いた重み付き割合はそれぞれ0.00470%、0.36746%だった。通常倍率の値は別試行のtraceであり、頻度も訓練全体ではなく検証標本に基づく。量子化値がより動くことは確認できるが、4 sbの正解率改善は確認できなかった。L3 10倍でも通常条件を上回らず、どちらの倍率もこの結果だけでは推奨できない。

872,416決着局面中、proxyと整数演算の符号が異なったのはL2 100倍で16,638局面、L3 10倍で16,759局面。前者はproxyのみ正解8,512、整数のみ正解8,126、後者はそれぞれ8,533、8,226。単一の集計精度が近くても局面単位の一致を意味しない。ログは `full_ft_qat_l2w100-2440-paired-l2x100.log`、`full_ft_qat_l3x10-2440-paired-l3x10.log`、通常条件は `full_ft_qat-2440-dual-baseline.log`。

追加のL3 bias単独更新では、他層を固定し、4 sb後のGPU proxy qacc 59.59932%、qloss 0.72662157、CPU整数qacc 59.61640%、train尺度loss 0.72699255。L3 biasは開始・終了時に量子化値1,129/3,528が変化し、他のトレース層は変化なし。単独更新設定ではL1実効重みclipを無効にする必要があるため、非量子化acc/lossは通常学習と直接比較しない。固定したL1を量子化評価することで整数側は引き続き範囲に収めている。この試行のみで通常学習の有効性や停滞原因は判断できない。ログは `full_ft_qat_l3biasonly-2440-paired-bias-only.log`。

## 追加コードの検証

`cargo check --release --features cuda-cpp-backend --example bulletou`、`cargo test --release -p bulletou-cuda-cpp --lib`（82 passed、109 ignored）、および `cargo test --release -p bulletou-cuda-cpp --lib sgd_ -- --ignored --test-threads=1`（15 passed）に成功。最後の15件には3,528 bucketの追加勾配テスト、SGD momentum、FT半精度momentum、factorizer切替、QATのcheckpoint再開が含まれる。診断機能は環境変数を指定した場合だけ有効になる。

FT半精度の別条件では、`all_optim=false` のままFTのweight/gradient/activation/momentsを個別にFP16にし、FT QATをオフにして同じcheckpointから4 sbを実行した。GPU proxy qacc 59.57834%、qloss 0.72689873、CPU整数qacc 59.58041%、train尺度loss 0.72724164。FT勾配のFP16範囲超過によるclip警告は491件で、momentum超過の警告はなかった。FP32条件とはFT QATの有無も異なるため、差を半精度だけの効果として解釈できない。ログは `half-2440-paired-half.log`。
