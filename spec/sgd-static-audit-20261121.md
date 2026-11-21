# 実行中学習の静的調査（2026-11-21）

本学習を停止せず、ソースと実行中設定を読み取って調査した。GPU実験や対局は実施していない。確認時は `SFNN_halfkahm2_2304_15_64_k21k21_progress8`、epoch 3、`all_optim=true`、`sfnn_qat_ft=false`、L1/L2/L3 QATオン、L1 pair factorizer、SGD momentum 0.95。

## FT半精度activationによるclamp勾配の欠落

`sfnn_hm2_one_view_half2_kernel` はFT各unitのFP32累積値にCReLUを適用した後、`sfnn_ft_fp16_act=true` ではFP16へ丸めた値を後段とbackward用cacheに保存する。`sfnn_l1_to_ft_half_gradient` はそのFP16 cacheを読み、`crelu_pre_gradient_from_value` で0より大きく1より小さい値だけを通す。

FP16へ丸める直前の値が1未満かつ十分に1に近い場合、丸め後は1になる。例えば `0.9999` はFP16で1となるので、丸めをidentity STEとした正しいCReLU判定は通過するはずの勾配が0になる。0近傍にも同様の丸めがある。これはL1/L2そのものではなくFT重み・biasへのbackwardで起きる。いまのarch・batch・all_optim設定は該当する半精度activation経路の条件を満たす。実際の頻度と学習影響は未測定であり、qacc停滞の原因と断定しない。修正は稼働中の学習を止めた後に、丸め前のclamp判定情報を保持する設計と再現テスト・速度影響を確認する。

既存の `fused_l1_ft_half_gradient_matches_separate_kernels` は0.9999を含むが、比較対象も先にFP16に丸めたactivationを使うため、fusedと参照の一致は確認できても丸め前のclamp判定との整合性は検証できない。

## 半精度勾配の範囲超過時のFT重みとbias

FTの半精度勾配経路は、局面・view・unitごとの勾配を `16384 * batch_size` 倍してFP16に変換し、65504を超えるとclipしてからFT重みへ集計する。一方FT biasは同じ勾配のFP32値をclip前に加算する。したがって範囲超過が生じた局面では、FT重みとbiasに渡る勾配は同じ値ではない。batch 65,536では、変換前の絶対値約 `6.10e-5` を超えた値が該当する。丸めによる差は範囲内でも発生する。clipは警告カウンタを持つ設計であり、ただちに実装バグと断定しない。いまのプロセスで何件発生しているかは、今回読み取ったsummary CSVには載っておらず未確認。

## epoch境界のSGD momentum

step LRのepoch再始動で過去のmomentum全体に新しいlrが掛かる点は、ユーザーの指示により修正した。方式と検証範囲は末尾の修正作業を参照。稼働中の旧プロセスには未反映。

## qlossの評価mode間の尺度

以前の実測で確認済みの点をコード上でも確認した。通常のGPU proxy quantized validationは `run_one_test_pass` に学習尺度の出力を渡す。一方 `quantized_validation_exact=true` のCPU整数検証は `report.engine_scale.test_loss` を同じquantized_value_loss欄へ返す。設定中のmodeはGPU proxyであり現在のsummary内で突然切り替わる問題ではないが、exact modeに変更した前後のqlossを同一尺度として比較することはできない。CPU整数推論の学習尺度lossも計算されるが、通常のexact validationからsummaryに記録されない。qaccもGPU proxyとCPU整数で演算が異なるため、局面別に一致するとは限らない。過去に同じcheckpointでGPU proxy qloss 0.72667408、CPU整数engine尺度loss 0.67212898、同じCPU整数出力のtrain尺度loss 0.72704655を観測している。

## clip後のmomentum残存

SGDの重み更新でclipが発生した場合もmomentumは保持され、丸め補償のみ消去される。L1実効重みclipもbucket残差の補正後にmomentumを消さない。そのため境界を越える方向のmomentumが残り、その後に勾配方向が反転しても、momentumの向きが変わるまではclipされ続ける可能性がある。これは制約付きmomentum SGDとして意図的にも選択できる動作であり、この事実だけでバグとは判定しない。現在の学習でL1やL2の該当clipがどの程度発生しているかは未測定。

## FT積の格子化と整数推論の境界差

integer QATはFT pairwise出力をまず浮動小数点で計算し、`dense_qat_lattice_integer` で `floor(127*x)` として0〜127に変換する。これは最後のFP32掛け算の丸め誤差をFMAで補正するが、それ以前のFT累積、activationのFP16化、積そのものの丸めは整数演算に戻さない。CPUのbinary32演算で手計算した例では、量子化FT累積が両方64（非飽和）なら整数推論の積は `floor(64*64/128)=32`。一方 `float32(64/127)` を両入力とし、実装どおりbinary32で積と127/128を掛けてから127倍すると約31.9999998808で、floorは31になる。FP16にactivationを丸めた場合の同じ数値例は約31.99609387で、やはり31。これは小さいCPU算術例であり、CUDAコンパイラ・GPUの実際の演算順で同じ結果になるか、実checkpointの何件で起きるかは未検証。現在はFT QATオフのため、整数重みへの量子化前後の差も別途ある。前回観測済みのFT proxyと整数推論の差と関係しうるが、この例だけでその原因や学習停滞の原因とは断定できない。

## qstatsの観測対象

`[qstats]` と `[qstats-unit]` は量子化proxy用の重みを作り、そのforward workspaceから算出する。現在の訓練forwardはFT QATオフかつall_optimによるFT半精度だが、qstatsのproxyは量子化された推論用モデルの近似であり、訓練中の半精度FT activationそのものの集計ではない。両者の飽和率や、半精度丸め直前の境界近傍の件数が一致することは保証されない。qstatsだけから前記FT STE境界問題の発生件数はわからない。

## 学習率変更時のmomentum再スケールはepoch境界以外でも起きる

現在の更新式は `m_t=0.95*m_(t-1)+g_t`、`Δw_t=-lr_t*m_t` なので、epoch再始動に限らず、通常のstep LR減衰でも過去に蓄積されたmomentum全体にその時点の新lrが掛かる。現設定では各4,000万局面ごとにlrが0.8倍となる期間も該当する。L1倍率の途中変更にも同じ性質がある。これは標準的なgradient-buffer型momentumの定義に沿う挙動で、減衰時も不具合と決めるものではない。ユーザーが指摘したepoch境界の修正では、境界だけのmomentum調整と更新量を記録する方式とで、通常減衰・再開時・層別倍率変更時の意味が変わるため、方式ごとに検証する。

## 学習ログの範囲確認

読み取り時点の `summary-learn.csv` でepoch 3 sb 325〜629の305行を確認し、test accuracy/loss、quantized accuracy/loss、および記録された各全体上限飽和率に欠損・非有限値はなかった。これは読み取り時に完了していた行だけの確認で、各unitの飽和率・FT境界勾配や半精度clipの発生有無を証明しない。

## 2026-11-21 修正作業

FT半精度activationは、順伝播の積にはFP16に丸めた値を使い、CReLUの逆伝播判定にはFP16丸め前のactivationを使う。fusedと非fusedの両経路を対象とし、0.9999と1.0の両視点の境界テストを追加した。

L1/L2/L3とFTのQATがすべてオンのとき、FTの量子化済み重み・biasからi16折り返しを含む整数累積とpairwise積を計算し、整数推論と同じ格子値を生成する。FT QATオフ時は従来どおり。境界、負値、折り返し、packed出力、小さいdense QAT全体のテストを実施した。追加のFT計算を行うため、実モデルでの速度影響は学習停止後に測定が必要。

epoch 2以降のSGD最初の更新前に、過去のmomentumを `前epoch最後のlr / 新epoch最初のlr` で変換する。これにより前epochからの更新寄与を保ちつつ、新しい勾配には新lrを適用する。通常のepoch内step LR減衰は変更しない。epoch境界checkpointからの再開時は、保存されたbulletou-settings.jsonから前epochのlrを復元し、読めない場合は警告して現在設定のscheduleから推定する。層別倍率がepoch境界で同時に変更される場合、その倍率変更自体は補正しない。これは新たな学習器が適用された後の境界から有効で、稼働中の旧プロセスには反映されない。

修正中は本学習を継続し、別ビルド先で小規模GPUテストを実行。実checkpointでの長時間学習、実モデルでの速度、epoch境界をまたぐ実checkpointの再開試験は未実施。

検証追記: FTの0近傍（正の値がFP16で0になる場合）と1近傍を両視点・3経路で検証し成功。QAT関連GPUテスト18件、SGD関連GPUテスト16件、以前の設定をcheckpointから読むepoch LR計算テストが成功した。別ビルド先は `C:\shogi\codex-bulletou-build`。epoch boundary時の層別倍率変更自体を補正しないほか、checkpointに前epoch設定がなく現在設定も変わっている場合、旧lrの推定は不確実で警告する。
追加検証: 整数FTは複数特徴・異なる先後視点のpacked出力をCPU整数計算と照合し成功。小さいQATモデルでepoch境界相当のlr変更を挟み、保存・復元した場合と継続した場合のmomentumと重みが一致した。いずれも小規模であり、実checkpointの再開や本番幅での速度を保証しない。
epoch境界で再開する際、前回checkpointがepoch途中で保存されていた場合も、そのcheckpointのsb位置までの最後のlrを保存設定から算出する。現在設定を変更して再開した場合の旧lrと新lrを混同しないためである。設定や進捗が読めない場合は警告し、現在scheduleから推定する。
別出力先のCUDA版releaseビルドと起動確認は成功。参考としてCUDA featureを付けないexampleのビルドも試したが、今回変更していない既存箇所の条件付きコンパイルで失敗した。対象のCUDA版は成功している。

## 追加静的レビュー: FT QATの整数forwardとbackward

`ce1e364` はFT QATオン時の `combined` を整数累積・i16折り返し後の結果で上書きするが、FT backwardが参照する `stm_l0` / `nstm_l0` および半精度activation cacheは、上書き前の浮動小数点FT forwardで計算した値のままである。したがって整数のi16折り返しによりCReLUの領域が変わる場合、整数forwardとbackwardのclamp判定が一致しない。

具体例: 量子化されたFT weightが32767、biasが65の1入力なら、整数forwardの2倍した整数累積は `2*(32767+65)=65664`、i16折り返し後は128なので、CReLUの内側。対してfloat側は `(32767+65)/127 > 1` として1にclampされ、FT backwardの該当勾配は0になる。相手側のpair activationと上流勾配が非ゼロなら勾配差が生じる。整数FT forwardの既存テストは折り返しを含むが、forwardのみでありこのbackwardは検証していない。

またFT QATと半精度FT重みの両方がオンでは、整数forwardが量子化済みFP32重みから整数を復元するのに対し、backward用のactivationはその重みをさらにFP16化したforwardの累積から計算される。例えば量子化整数値1,17,-18の3特徴は整数和0だが、それぞれ `q/127` をFP16化して足すと正の約6.10e-5になりうる。これもclamp境界の判定を変えうる数値例であり、実GPUでの発生頻度・影響は未測定。現在稼働中の設定はFT QATオフなので、これらの整数FT QAT固有の経路は現在使っていない。

0ちょうどでの微分の取り方に依存しないCPU算術例として、量子化整数値2033, 2, -2036の和は-1で整数側はclampで0になるが、各 `q/127` をFP16化しFP32で足すと約+0.00012207になる。このとき他方のpairと上流勾配が非ゼロなら、浮動小数点側の判定は勾配を通す。大きな重みを用いた構成例であり、実モデルでこの条件が現れると主張するものではない。

## 追加静的レビュー: epoch境界再開の旧lr復元

`ce1e364` はepoch境界checkpointからの再開時、checkpointの `bulletou-settings.json` を読み直して旧lrを算出する。しかしこのファイルは保存時に `--settings-file` の元ファイルをそのままコピーしたもので、同時に指定した `--lr`、`--lr-min`、batch関連等のCLI上書きを含まない。CLI指定は実際にはJSONより後に展開されて有効になるため、直前の学習がこれらの値をCLIで上書きしていた場合、旧lrを誤って復元し、momentumの倍率を誤る。元JSONを読めるので、このケースではfallback警告も出ない。これは先に追加した修正の条件付き不具合。実効設定を保存した `resume-config.txt` 等を参照するか、実際の最後のoptimizer lrをcheckpointに記録する修正と、CLI上書き・schedule変更・途中checkpoint再開のテストが必要。

今回確認した稼働中プロセスのコマンドには `--lr` や `--lr-min` のCLI上書きはなく、現時点でこの条件に当たる証拠はない。継続実行でのepoch境界はcheckpointから旧設定を読み直す分岐自体を使わない。ソースコードは変更していない。
