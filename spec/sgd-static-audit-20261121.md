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

## 追加静的レビュー: 検証ログの非原子的な更新

SFNNのcheckpointは重み・optimizer状態・未量子化の検証結果を書き、ディレクトリを確定してsummaryに追記した後、量子化検証を行って `learn.log` と `summary-learn.csv` を順番に更新する。従って、保存完了後から量子化検証の記録完了前に中断すると、保存は成功していてもその地点のqacc/qlossは欠ける。後者だけ更新前に中断した場合はcheckpointとsummaryの値も食い違う。検証の途中中断を学習失敗と誤認しないよう注意する必要がある。

さらに、これらの更新は `std::fs::write` で既存ファイル全体を直接上書きする。書き換えの途中でプロセスが終了したり書き込みに失敗したりすると、summary全体やcheckpointの `learn.log` が空または途中までになる可能性がある。checkpointの完了判定は `state.bin`、`learn.log`、`dataloader_pos.txt` の存在のみを見るため、壊れた `learn.log` が存在する場合も完了と判定しうる。これはログと再開処理の耐障害性の問題で、重みや勾配の計算ミスを意味しない。修正には一時ファイルへの書き出しと置換、必要な整合性検査、欠損した量子化指標の扱いの検討が必要。

## 追加静的レビュー: epoch境界でのFT半精度momentum再量子化

`all_optim=true` の半精度FT momentumは、epoch境界のlr再始動で一度FP32に戻して旧lr/新lrを掛け、次のFT更新で再びFP16に変換される。`2^18` 倍してFP16に保存するため、変換後の値が小さい場合は0に丸められ、前回の更新寄与を厳密には保存できない。例えば保存値がFP16の最小正subnormal `2^-24` で再スケール係数が `5/44` の場合、再保存値は0となる。CPUの半精度変換で数値例を確認した。通常の半精度momentumにも丸めはあるが、この再スケールでも追加の丸めが生じる。実際に該当するmomentumの個数や学習への影響は測定していない。また、係数が大きい場合は表現範囲外になりclipしうる。clipにはカウンタがあるが、ゼロへの丸めには専用の警告はない。

稼働中のプロセスはPID 20796として継続していることを読み取りで確認した。本調査では学習の停止・再開、GPU実験、ソース変更は行っていない。

## 追加修正

整数FT QATのforwardが用いたi16 wrap・clamp後のactivationをbackwardにも渡し、FP16 activation cacheも同じ値に更新する。checkpointに実際の最後のSGD更新lrをf32で保存し、epoch境界からの再開時はそれを優先する。旧checkpointに値がない場合は従来のsettings推定に戻し、CLI上書きが欠落しうることを警告する。FT半精度momentumはepoch再スケール直後の最初のFT更新をFP32で行い、次の更新からFP16へ戻す。以降は通常のFP16丸めが生じる。

量子化検証のログ更新は一時ファイルからの置換とし、checkpoint完了判定でlearn.logのヘッダーと行の基本構造を確認する。checkpointの確定と量子化検証自体は引き続き別処理なので、その間の中断によるqacc/qlossの欠損や2ファイル間の一時的不一致は未解消。自動での再検証も行わない。

CUDA版ビルドとcheckpoint関係のCPUテスト24件、epoch境界lrのCPUテストを確認。FT QATおよびmomentum再スケールのGPUテストは追加・コンパイルしたが、本学習が稼働中のため未実行。稼働中の旧プロセスに変更は反映されない。

## 追加調査: all_optimと処理速度

稼働中PID 20796の起動ログとepoch 3の設定展開は、FT weight/activation/gradient/momentsの4項目すべてFP16有効を示す。17:40頃、RTX 4090の使用率は100%、VRAM 22,798/24,564 MiB、SM 2730 MHz、63℃、ソフトウェアの電力・温度制限は非発動。これは一時点の観測である。実行中バイナリはcommit `ce1e364` で、直前の未ビルドの修正とは別。

同一stdout.logを起動単位で分割して集計すると、epoch 3 sb 375〜424（各50 sb）のtrain時間中央値は、以前のL1実効clipオフのrunでは36.35秒、clipオンの別runでは40.95秒だった。ともにall_optim有効。別runの比較なのでclipだけが差の原因と断定できない。現在のrunのsb 700〜747（48 sb）では42.95秒、約93万局面/秒（中央値）。validation時間を除いたtrain時間である。clipオンでも以前のrunのsb 700〜747は42.05秒だった。roll-backで同じsbが複数回含まれるためrunを混ぜて集計しないこと。

現在のL1実効clipは各optimizer更新でbucket全体の `3528×2304×16 = 130,056,192` 要素を走査し、sharedとaxis/pairを畳み込んでclipする。SGDでも使わないLookahead slow側を同時に読み計算する。GPU負荷増加の具体的な候補だが、未計測であり速度低下の内訳は不明。FT bias勾配の各局面からのatomic加算も候補。稼働中の学習へのGPUプロファイラ接続や実験は行っていない。

直前のmomentum修正の追加不整合: epoch境界で `all_optim=false→true` になる場合、momentum再スケール時には旧設定のFP16 momentsが無効なので `ft_half_moments_defer_once` がfalseになり、その後にFP16を有効化すると最初の更新前に再量子化される。先の修正が意図した最初の更新でのFP32保持は、この切り替え条件では成立しない。現在の設定はepoch 1〜5 true、epoch 6でfalseへの切り替えであり、このfalse→true条件には該当しない。未修正。

追記: これは設定切り替え時に限らない。新プロセスの `SfnnTrainStepRunner::new` はFP16 moments無効で初期化し、epoch境界先頭で `rescale_sgd_momentum` を呼んだ後に `set_ft_fp16_moments` する。よって同じ `all_optim=true` のままepoch境界checkpointから再開する場合も `defer_once=false` のままで、最初のFT更新前に半精度へ再量子化される。現在の設定でepoch 3末checkpointから同じ設定でepoch 4を再開する場合はこの条件に該当する。再スケール係数は保存時と再開設定に変更がなければ0.0005/0.0044。これは今回未commitの修正案自体の欠陥であり、実行中の旧プロセスのコードとは区別する。最初のFP32更新の保証とsnapshotの復元を併せて停止後に再検証する必要がある。

既存の `0003-0972sb/state.bin` のFT momentumから、固定seedで32箇所・各16,384個の連続要素を読み、係数5/44とFP16保存scale 2^18をCPUで適用した。計524,288個のうち非ゼロ477,642個、うち少なくとも492個はFP16最小subnormalの半分未満となり0に丸められる。範囲上限を超えるものは標本中0個だった。連続領域を選ぶ標本であり全tensorの割合を厳密に表さない。またこれはepoch末ではなくsb 972の保存値を用いた再量子化前後の算術比較で、次stepの勾配を含む更新差や学習品質への影響を測定したものではない。GPUは使用していない。

## 追加静的検査: summary巻き戻しとL1 clip

量子化検証結果の記録は一時ファイルからの置換に変更したが、再開時にcheckpointより後の行を削除する `truncate_summary_log_after_checkpoint` は、依然としてsummary全体に `std::fs::write` している。再開時のこの書き込み中に停止や書き込み失敗が起こると、summaryを途中までしか保存できない可能性がある。またsummaryの読み込み失敗はすべて「削除行0」として扱われ、ファイル不存在と読み込みエラーを区別しない。grid_search経由では巻き戻し前のbackupを取る経路があるが、学習器を直接起動する場合まで同じ保証はない。これはログと再開時の耐障害性の問題であり、勾配の誤りを示すものではない。未修正。

pairを含むL1実効clipはFP32で `w += limit - folded` と補正するので、補正後に再びfactorizerを加えた値が丸めで境界を僅かに超えることがある。CPUのFP32計算による構成例では、上限1.984375に対して補正後1.984375238となった。GPU上の実checkpointでの発生件数は未測定であり、この程度の超過は直ちにint8量子化結果の違いや学習停滞を意味しない。境界を厳密に保証するという意味での数値上の問題として記録する。

稼働中runのstdoutを今回読み取った時点では、FP16 FT勾配の範囲超過警告109回、合計18,607件、1警告あたり中央値169件だった。FT optimizer momentsの範囲超過警告は同じ範囲で0回。これはclipの発生数であって影響の大きさを示さない。稼働中の旧プロセスには今回追加したソース修正は反映されていない。

## 追加静的検査: PSV再開位置

固定長PSV（`.bin`を含む）の自動再開では、checkpointに `dataloader_pos.txt` が存在しても、`learn.log` の生涯累計局面数から `累計局面数×レコード長 mod 現在の教師の総バイト数` を計算し、それを優先する。過去に教師を変更したり `--initial-dataloader-pos` で非ゼロ位置から開始したりすると、生涯累計は現在の教師での読込位置とは限らない。例えば10レコードの教師を位置5から2レコード読み、別の期間も含む累計が12の場合、保存位置7に対し累計からの推定は2になる。実装は不一致の警告を出すが、位置2を採用する。既存のCPUテスト `cuda_cpp_psv_resume_prefers_learn_log_positions` は保存位置より累計由来の位置を選ぶ現挙動を確認しており、今回実行して通過した。このテストは開始オフセットや教師変更後の継続を正しく扱うことは検証していない。これは学習器の再開位置処理の問題で、教師データ自体を疑うものではない。

現在の教師ファイルは `.bin` でPSVとして扱われる。既存のepoch 3 sb 648 checkpointでは、記録された総局面数129,525,350,400、40バイト/局面、現在のファイル総サイズ1,983,794,202,520バイトから算出した位置は1,213,425,610,960バイトで、保存された `dataloader_pos.txt` と一致し、実際の直近再開ログとも一致した。したがって、このcheckpointに関して位置の食い違いは確認されていない。次のcheckpointが保存される前なので、その位置の一致はまだ確認できない。修正には旧形式の不正位置からの救済要件と、教師変更・初期位置指定を区別できる記録形式を併せて検討する必要がある。未修正。

追記: その後 `0003-0972sb` が保存された。記録された読込位置1,731,527,012,560バイトは、総局面数142,477,885,440から同じ計算で得た位置と一致した。`learn.log` のqacc/qlossも記録済み。`state.bin` について主要12グループのweights/momentum/velocity/slowレコードの存在とサイズをseekで確認し、欠落・サイズ不一致はなかった。重みの値、保存直後の実再開動作、checksumを検査したものではない。

## 追加静的検査: factorizer optimizer state欠落時の扱い

SFNNのL1 sharedやaxis/pair等の重みが存在する一方、そのグループのmomentum、velocity、slowの**3レコードすべて**がcheckpointに存在しない場合、`load_or_zero_cuda_cpp_ranger_group_state_for` は警告せずにmomentumとvelocityを0、slowを現在の重みに初期化して続行する。3つのうち一部だけがある場合はエラーになる。旧形式・移行対応としての挙動だが、現行形式の壊れたcheckpointからもそのfactorizerのSGD momentumと丸め補償だけを失った状態で再開し得る。optimizer全体のstep数は継続される。ファイル形式のバージョンや明示的な移行条件で区別し、少なくとも警告・現行形式では拒否する対策が必要。これは欠落がある場合の条件付き問題である。

参考として、現在の直近checkpoint `0003-0648sb/state.bin` をseekでレコードの名前・長さ・ファイル境界のみ検査した。FT、L1 residual/shared/axis、L2、L3の各weights/momentum/velocity/slowはすべて存在し、それぞれの要素数も一致した。これはデータ値やchecksumの検証ではなく、破損が全くないと保証するものではない。新たなcheckpointが作成されるまで現在の学習には触れず、GPU実験もしていない。

直前に追加した `ft_half_moments_defer_once` は、`snapshot_device` で保存されず、`copy_state_from_device` や `upload_state_from_host` でも復元・初期化されない。例えばepoch境界でmomentum再スケール直後、最初のFT更新前にsnapshotを作って別runnerへ復元すると、復元先のフラグ次第で最初の更新をFP32で行うという保証が変わる。逆に古いフラグが残って不要なFP32更新を1回行う可能性もある。これは前回の修正に伴うsnapshot/restore API上の条件付き不整合であり、現行の直接学習ループでその境界にsnapshotが作られる証拠はない。通常のディスクcheckpointはこのepoch境界での再スケール前に保存される。未修正。

## 追加静的検査: dense QATの整数演算とbias

現在有効なL1/L2/L3 dense QATについて、L1の整数積・bias、L1平方／線形activation、L2の整数積とclamp、L3の整数積とL1 skip加算をCPU整数forwardと照合した。通常の整数範囲では対応している。FTは現在QATオフのため、既に記録したとおりQAT学習時のFTと整数推論のFTは同一ではない。この確認は全入力に対する形式的証明やGPU実行試験ではない。

整数オーバーフローの可能性を考えるため、epoch 3 sb 972 checkpointからbias値だけを読み取った。L1 residual/shared/axis、L2、L3のbiasは全て有限で、それぞれの最大絶対値は約0.0451、0.0489、0.4571、0.1252、0.0467。L1の実効biasはこれらの和であり、個々の値だけで実効値の分布を保証しないが、これらの値から直ちにint32限界付近とはいえない。L1/L2/L3重み全体やあらゆる局面の中間和を測定したものではない。今回は新たな確定的な計算バグは見つかっていない。

## 追加静的検査: lossの分母と非有限値の警告

学習CUDAの通常BCEは各局面lossにentry weightを掛け、総和を全batch数で割る。score filterがある局面のentry weightを0にした場合も分母は減らない。一方、検証lossはフィルター後の件数で割る。例として2局面がともにlogit 0で1局面がfilterされれば、学習側のmeanは `ln(2)/2 ≈ 0.34657`、同じ出力・対象に対する検証lossは `ln(2) ≈ 0.69315` になる。勾配も全batch数で正規化される。このスケーリングを設計として選ぶこと自体と、学習lossと検証lossを同じ平均として説明することは区別する必要がある。勾配の正規化を変える場合は実効学習率が変化するため勝手に修正しない。

現在のログに出ている `test_value_loss` / `quantized_value_loss` は両方とも検証値で、上記の学習側の分母は直接使わない。起動ログの検証集合は `mate_filtered=0`、`loss_n=921060`。学習バッチでの除外数はこの調査では測定していない。したがって、今回確認した違いを現在のloss/qlossの乖離や上昇の原因とは主張できない。

FP16 FTの異常値警告には別の欠落がある。`sfnn_scaled_half_gradient` は `value>65504 || value< -65504` のみを調べ、NaNはいずれの比較もfalseになるため、NaNを半精度へ変換してもこの超過カウンタに加えない。一方 `ft_half_moment_scaled` はNaNもカウンタに加えるが、NaN自体はそのまま返すので、その後の「range exceeded and were clipped」という警告はNaNについては正確ではない。NaN発生の証拠はなく、これは発生時の診断と警告表示に関する条件付き不具合。有限の範囲超過とは区別して計数・表示する必要がある。未修正。

## 追加静的検査: k21k21_progress8のbucket選択

学習器の `k21k21` の玉位置の先後反転・21分類・先手番側×21＋非手番側と、workspace内YaneuraOuの `king21_by_king21_bucket` をソース上で比較した。合法な0〜80の升番号は双方とも `file*9+rank`、反転は `80-square` で、分類式とbucket合成順序は一致する。progress8の7境界についても、BulletOuの256値thresholdから32値ごとの境界を選ぶ式とYaneuraOuの8分割thresholdをCPU算術で比較し、q16閾値はすべて一致した。盤面中の特徴抽出・評価器バイナリ全体の同一性を証明する比較ではない。

さらに学習設定の `20280908-progress.bin` とcheckpoint `0003-0972sb/progress.bin` を読み、125,388個すべてのf64値が一致することを確認した。双方のq16変換値も一致した。検査は保存済みファイルの読み取りとCPU算術のみで、GPUや対局を使っていない。

現在の設定ではFT/L1/L2 reviveフラグはfalseだが、各閾値を指定しているため、epoch開始時にそれぞれ測定のみが実行される。stdoutのepoch 1〜3ではいずれも `measurement only`、`revived=0` であり、これらによるunit再初期化は行われていない。これはコード上の仕様で、閾値指定が測定を有効化する点を明記する。

## 追加静的検査: CPU整数検証の最終評価値clip

workspace内のYaneuraOu `evaluate_nnue.cpp` はNNUE整数出力を `FV_SCALE` で除算した後、`[-VALUE_MAX_EVAL, VALUE_MAX_EVAL]` に制限する。一方BulletOuの `quantized_test_report_from_outputs` がengine-scale検証用の出力を作る際は、この範囲制限を行わない。したがって除算後の絶対値が上限を超える場合、CPU整数検証のengine-scale lossはエンジン実際の評価値から算出したlossと一致しない。例えばFV_SCALE=24、raw=1,000,000なら除算後41666となり、通常設定 `MAX_PLY_NUM=246` のYaneuraOuでは最大31753に制限される。MAX_PLY_NUMはengineビルド条件に依存する。正の上限でのclipは符号を変えないため、score offsetなしの符号正解率にはこの違いは影響しない。`quantized_calibration_engine_report_from_outputs` も同じ範囲制限をしていない。これはCPU整数検証・calibrationの条件付き不整合であり、現在の `quantized_validation_exact=false` のGPU proxy qlossには直接該当しない。sb 972のcheckpointでこの上限を超える出力があるかは未測定。未修正。

同じ最終除算の非既定設定 `quant_final_div_round=nearest` では、丸め用の `output + FV_SCALE/2` または `output - FV_SCALE/2` をi32のまま実行する。outputがi32端付近なら加減算がoverflowし、debugではpanic、通常のoverflow check無効releaseでは符号が反転し得る。例として `output=2147483647, FV_SCALE=24` の正しい丸め除算は89478485だが、i32で加算がwrapすると-89478484になる。CPUの整数演算例で確認。これも極端値・非既定roundingの条件付き不具合であり、現在のGPU検証の原因ではない。未修正。

同様にCPU整数検証の `quant_crelu_round=nearest` は正のi32に32を足してからshift・clampするため、`2147483647` のような入力では足し算がoverflowする。releaseでwrapした場合、期待される飽和値127ではなく0になる。これはL1/L2の内部整数和が上限付近に達した時だけの問題で、既定のfloorではこの加算を使わず、現在のGPU proxy検証でも使わない。CPUの整数算術で確認。未修正。

## 追加静的検査: progressの存在しない駒スロット

workspace内YaneuraOuでは駒落ち等の存在しない駒をeval listの `BONA_PIECE_ZERO` に置き、progressの `SumQ16` は非玉駒の全スロットについて両視点の `weights_q16_[king][piece]` を加算する。一方、BulletOuの `shogi_sfnn_progress_sum_q16_from_board_explicit` は実在する駒だけを数え、BonaPiece ZEROは加えない。progress.binの `king*1548+0` が非ゼロの玉位置について駒不足の局面を処理すると、そのスロットの数だけ和が異なり、境界付近ではprogress bucketも異なり得る。progress.bin読込はこれらの0番成分が0であることを強制していない。これは学習器とエンジン間の条件付き不整合で、存在しない駒の扱いに関するものである。

現在の `20280908-progress.bin` および `0003-0972sb/progress.bin` については、81個の `king*1548+0` が全て0であることをCPUで確認した。したがって現在使用中のprogress重みに、この差は生じない。任意のprogress.binに対しては、0番を0にする制約を両側で検査するか、エンジンと一致する形で欠けた駒を扱う必要がある。未修正。

## 追加静的検査: FT QATのゼロ境界STEの設計不一致

未commitの整数FT backward修正案は整数FTのclamp後の値をbackwardの判定に使うが、量子化前のmaster FT biasの端数を保持しない。例として、あるunitのmaster biasが正の `0.25/127` で他の重みが0なら、整数化後biasは0、修正案のCReLU判定は0として勾配を止める。片方のpair activationと上流勾配が非ゼロなら、量子化をidentity STEとしてmaster biasの端数をclamp判定に復元する方式では、このunitの勾配は非ゼロとなる。一方L2 QATには整数値0や8128のときmaster biasの端数を使う処理がある。FTとL2でSTEの境界の扱いが一致しておらず、前回のFT QAT修正は整数wrapとの整合だけで完了とは言えない。どちらのSTEを仕様とするか、量子化された複数FT重みの端数・wrap・half精度を含めて設計し、独立した勾配テストで決める必要がある。現在実行中のFT QATはオフで、この条件は使われていない。GPUテスト未実施。

## 追加静的検査: BPU>1のlr_end記録

production scheduleはcheckpoint/validation区間の `lr_end` を最後のmini-batchのstep位置で計算するが、実optimizer更新は最後の累積グループの**先頭**mini-batchのstep位置のlrを使用する。BPU>1で最後の累積グループの途中にstep LRの減衰境界があると、`learn.log` と `summary-learn.csv` の `lr_end` が最後の更新に使われた値と異なる。CPU算術例としてbatch size 1、4 batch/sb、BPU 2、3局面ごとに0.8倍、初期lr 0.0044の場合、最後の更新はstep index 2の0.0044を使うが、ログはindex 3の0.00352を記録する。学習更新そのものがこのログ値を使うという意味ではない。現在のBPUは1なのでこの条件には該当しない。未commitの修正案が別に保存する `sgd-last-lr.bin` は、通常のdirect production保存では実際のoptimizer paramsから得るため、この `lr_end` 表示とは別である。未修正。

## 追加調査: L1実効重み境界の標本

現在の `all_optim=true`、L1 pair factorizer、実効重みclip有効の経路を再確認した。L1 residual/shared/axisのSGD更新後に、folded実効重みを範囲 `[-2,127/64]` に投影し、補正されたresidualの丸め補償は0にするが、momentumは残す。これは前述のとおり制約付きmomentumの設計選択であり、それだけでバグとは判定しない。

既存 `0003-0972sb/state.bin` を読み取り、固定seed 20261121で3,528 stackから32 stackを無作為抽出し、それぞれ36,864個、計1,179,648個のL1実効重みをCPUで再構成した。shared、2つのking軸、king-progress pair、progress軸を保存済みmasterに加えた。すべて係数1で、実行時のFP32加算順を使用。上限から1e-5以内は122個、下限から1e-5以内は189個。境界から1e-6を超えて範囲外にあるものは0個だった。

境界近傍のうち、residual単体のmomentumが上限側では負（SGD更新が上向き）であるもの52個、下限側では正（更新が下向き）であるもの95個。shared/axisのmomentum、次batchの勾配、clip後の移動は計算していないため、実効重み全体が外向きに動く、clipが継続する、または学習を阻害すると結論できない。標本は局面頻度で重み付けしておらず、全stackの正確な比率ではない。GPUや本学習のプロセスには触れていない。

現在のBCE設定について、教師目標は `effective_win_rate_model` を通じてWRMで作り、CUDA BCE lossとその勾配はいずれも `wrm_nnue2score / wrm_in_scaling` を使用することをソース上で確認した。この確認はデータごとの全勾配やGPU計算結果の検証ではない。

同じcheckpointのL1 revival履歴は `l1_last_reset_epoch=0`, `l1_suppress_until=0`、L1/L2 revival flagsはいずれも0だった。現在の設定には `sfnn_l1_revive_lr_suppress_sb=128` があるが、このcheckpointの履歴と設定上のrevive=falseのもとでは、その履歴によるL1学習率の0.1倍抑制は有効にならない。現在の起動後に実際のrevivalが発生していないことは別途stdoutで確認済み。初期抑制設定も0である。

FT半精度scratchをepoch 6のtrue→false切替で解放する順序を確認した。各学習batchの終了時にCUDA contextのFT半精度3フラグはfalseに戻され、次のepoch境界でscratchを解放するため、静的には解放側の「フラグを先に無効化」条件を満たす。momentumの半精度キャッシュは別管理で、設定変更後 `set_ft_fp16_moments(false)` でFP32に展開される。実際のepoch 6切替のGPU実行試験ではない。

## 追加静的検査: 現在のL1/L2 QAT逆伝播とテスト範囲

現在の幅2304、L1出力16、3,528 bucket、pair factorizer、L1/L2/L3 QATでは、L1 backwardは `sfnn_l1_bucket_gather_gradient2` の経路を選ぶ。各bucket内の `dy*x` と `dy` をresidualに加算し、sharedにはshared係数、各axis/pairには対応係数を掛けて同じ合計を加算する。QATで畳み込み済みのL1重みは、FTへ戻す入力勾配の計算にも使われる。コードを追った範囲で、shared/axisの二重加算や係数脱落は見つかっていない。

全3,528 stackについてaxis ID式をCPUで列挙した。各stackは重複しない4 IDを持ち、範囲は全て0〜3,577。先手側玉軸各21 IDと後手側玉軸各21 IDはいずれも各168 stack、玉pair×progressの各3,528 IDは各1 stack、progress軸の各8 IDは各441 stackに対応した。これはIDの範囲と組合せの検査であり、実局面bucketの出現分布やGPUでの勾配値の検証ではない。

既存の `l1_gather2_king_progress_axes_match_cpu` は3,528 bucketでwidth 64、`l1_gather2_full_width_with_pair_matches_cpu` はwidth 2304で8 bucketをそれぞれCPU参照と比較する。`sgd_dense_qat_3528_bucket_gradients_match_scalar_ste` は3,528 bucketでQATの各層勾配を比較するが、FT幅はtinyモデルの幅であり、FT半精度フラグも有効化していない。したがって、これらを合わせても現在の幅2304・3,528 bucket・batch 65,536・all_optim=trueの実checkpointにおける一括の順伝播/逆伝播のGPU参照検証ではない。今回GPUテストは実行していない。

FT factorizerのvirtual row勾配は、各mini-batchで実特徴の累積勾配を集計し直して代入される。BPU>1でも実特徴側はoptimizer更新まで累積されるため、前mini-batchのvirtual勾配を上書きすること自体は直ちに累積漏れを意味しない。現在のBPUは1。この確認は式と実装の静的照合である。

## 追加調査: FT folded重みのFP16丸め標本

`0003-0972sb/state.bin` のFT base 73,305 featureから、固定seed 20261121で128 featureを無作為抽出した。各2,304 unitについてvirtual rowを係数1で加算した294,912要素を、CPUでFP16に丸めてFP32に戻した。folded値は全要素有限・非ゼロで、FP16丸めにより0になったものは標本中0。folded絶対値の中央値は約0.019609、最大は約0.279354、丸め絶対誤差の平均は約0.00000363、最大は約0.00012183だった。

これはcheckpoint上の重み標本だけの比較で、稼働中の局面ごとの特徴出現頻度、FT累積値の誤差、CReLU境界やqaccへの影響を測定したものではない。gradientやmomentumのFP16 clipについても別問題である。全featureへの一般化はしない。

## 追加調査: state.binからnn.binへのL1 export標本照合

既存の `0003-0972sb` を対象に、`state.bin` と同じcheckpointの `nn.bin` を読み取り専用で照合した。nn.binのヘッダーと2つのLEB128チャンクの長さから後段networkの開始位置を求め、全3,528 stackの固定strideと実ファイルサイズが一致することを確認した。

固定seed 20261121で32 stackを選び、`state.bin` のL1 residualにshared、2玉軸、玉pair×progress、progress軸を順にFP32で加算し、重みは64倍して最近接・同距離なら絶対値が大きい側に丸め、i8にclipした。biasは同様に8,128倍しi32にclipした。`nn.bin` の該当する重み1,179,648個とbias 512個を比較し、不一致はどちらも0件だった。

これは保存時のCPU側のfold・量子化と出力ファイルの標本照合であり、live GPU QAT/proxyの値との直接比較ではない。全stackとFT/L2/L3の照合も実施していない。稼働中の学習やGPUへは触れていない。

同じcheckpointの追加照合として、FT bias全2,304個、FTの先頭64 feature × 2,304 unitの147,456重み、上記と同じ32 stackのL2重み61,440個・bias 2,048個、L3重み2,048個・bias 32個も `state.bin` の量子化値と `nn.bin` を比較し、不一致は0件だった。L2のパディング4,096個はすべて0。FT重みは圧縮データの先頭部分だけであり無作為標本ではない。GPU proxyや推論出力の比較ではない。

## 追加調査: FT半精度momentumのゼロ勾配時の減衰停止

現在の `all_optim=true` かつSGDでは、FT weight momentumを `2^18` 倍したFP16で保持する。更新kernelは `m = beta * decoded_m + gradient` をFP32で計算し、その `m` で重みを更新した後、momentumを再びFP16に丸める。`beta=0.95`、gradient=0の場合、FP16の非ゼロsubnormalのうち十分小さな値では、0.95倍が丸められて元の値になる。例えば格納値 `2^-24` は、0.95倍してFP16に戻すと再び `2^-24`。よって同じ条件が続く限りmomentumがゼロへ減衰せず、重み更新にも非常に小さい寄与が残る。各stepで消滅しないようにする丸め補償は重みにはあるが、momentumのFP16再格納にはない。これはFP16 moment量子化の数値上の挙動であり、最適化の近似として許容するかは設計判断を要する。

`0003-0972sb/state.bin` のFT momentumから固定seed 20261121で32箇所×16,384要素を抽出し、保存scale 2^18、beta 0.95、ゼロ勾配をCPUのFP16変換で適用した。計524,288要素中、非ゼロ477,642要素のうち8,023要素は1回の減衰後もFP16格納値が変化しなかった。該当する非スケールmomentumの絶対値は最大約2.274e-12。現在のlr 0.0005なら、その最大値でも無勾配1回のweight更新寄与は約1.08e-15。連続領域を選んだ標本であり、現在の全tensorの割合や、対象が実際に無勾配であり続ける期間、学習品質への影響は不明。CUDA上の再現試験は未実施。`all_optim=false` のFP32 momentum経路では同じFP16 subnormalによる停止は発生しない。

## 追加静的検査: epoch長を変更した再開とL1 revival後の学習率抑制

L1 revivalで実際にunitを再初期化したとき、抑制終了位置 `l1_suppress_until` は、その時点の設定から計算した累計superbatch座標（過去epoch長の合計＋抑制長）でcheckpointに保存される。一方再開後、現在位置を求める `l1_suppression_epoch_start` は、保存済みの過去epoch長ではなく、**現在の設定**の過去epoch長を合計する。再開時に過去epochの `superbatches` を変更できるので、この2つの座標系がずれ、残る抑制期間が不正になり得る。抑制中はL1学習率倍率が0.1倍となるため、単なるログ表示の問題ではない。

例: epoch 1が10 sb、epoch 2開始時にrevival、抑制16 sbとすると、保存する終了位置は10+16=26。epoch 2のsb 4のcheckpointから続行する場合、本来sb 5開始の位置は10+4=14なので残り12 sb抑制する。再開設定でepoch 1の長さを20に変更すると、コードは同じsb 5を20+4=24と計算するため、保存値26を超えるsb 7から抑制を解除し、残り2 sbしか抑制しない。逆方向の変更では抑制が長引きうる。これはコード上の算術例であり、実checkpointでこの設定変更が行われたことを示すものではない。

現在のcheckpointは `l1_suppress_until=0`、L1 revivalはfalseであり、この問題が現在の学習へ影響した証拠はない。修正時は現在設定から過去の累計を再計算せず、checkpoint由来の残り抑制step数か実績に基づく座標を保存し、epoch長変更・checkpoint途中再開・複数revivalの重複をテストする必要がある。今回は修正していない。

## 追加静的検査: direct出力の既存weights.bin置換

`write_cuda_cpp_state_records_atomic` は一時ファイルへ全レコードを書いた後、既存の出力ファイルがあれば `remove_file` し、それから `rename` する。削除後rename前に中断すると以前のファイルがなくなり、renameがエラーになった場合は一時ファイルも削除するため、旧・新の双方を失う。関数名にatomicとあるが既存ファイルの置換はatomicではない。SFNNの `cuda-cpp-direct/weights.bin` など、同じdirect出力を上書きするケースで該当する。

現在のproduction学習の番号付きcheckpointは新しい一時ディレクトリにstate.binを作成してからディレクトリを確定し、既存の完成済み同名checkpointを上書きしないため、この既存ファイル削除分岐を現時点の通常checkpointにそのまま当てはめることはできない。別に記録済みのsummary更新・中断時の問題とも区別する。修正は実際の利用経路に応じて既存ファイルを安全に置換し、rename失敗時に旧ファイルを維持する必要がある。今回はソースを変更していない。

## 追加静的検査: warmupを途中で切り上げた再開とmomentum

SGD momentumのepoch境界再スケールは `progress.epoch > 1` の場合に限定されている。通常のepoch 0 warmupは最後のoptimizer更新で開始lrまで到達し、epoch 1も同じlrから始まるので、設定を変えず完走する場合はこの条件でも問題にならない。

一方、warmupの途中checkpointから再開する際、設定変更によって `production_resume_start` がepoch 1を選ぶ場合、旧lrと新lrが違ってもepoch 1では再スケールが実行されない。例: 元のwarmupが8 sbで最初の2 sbを完了、BPU=1、lr=0.0044なら、最後のwarmup更新のlrは0.0011。再開時にwarmup長を2 sbに短縮すると、保存長8・新しい長さ2からepoch 1へ進み、新lrは0.0044になるが、保存済みmomentumに0.25を掛けず使う。以前決めた「epoch境界で過去momentumの更新寄与を保持する」修正方針に照らすと、この条件が漏れている。実際の重み変化は新しい勾配とmomentumに依存する。これは構成例で、実checkpointでの再現試験は未実施。

現在の学習は `warmup_sb=0` なので該当しない。修正時には旧checkpointの実際の最終lrを使う方針と併せて、通常のwarmup完了と途中切り上げ・再開設定変更を区別せず正しく扱う必要がある。今回は修正していない。

## 追加静的検査: 保存時のquantized_validation_rate無視

SFNN production scheduleは `quantized_validation_rate=0` をepoch末のみ、負値を無効として `chunk.run_quantized_validation` を設定する。一方、通常のcheckpoint保存分岐はそのフラグを確認せず `maybe_run_live_sfnn_quantized_validation` を呼び、この関数や `quantized_test_args_from_training_args` もrateを調べない。SFNNでtest teacherがあれば保存時に量子化検証が実行される。

例: 1 epoch 3 sb、save_rate=2、quantized_validation_rate=0の場合、schedule上はsb 2のcheckpointで量子化検証false、sb 3でtrueだが、実行時はsb 2でも量子化検証を行う。rate=-1でも各checkpointの検証が行われ、無効化できない。余計なGPUまたはCPU整数検証時間が生じ、出力CSVにも本来抑止した量子化指標が記録される。通常の非保存validation分岐は `chunk.run_quantized_validation` を確認する。

既存の `cuda_cpp_backend_zero_quantized_validation_rate_runs_epoch_end_only` と `cuda_cpp_backend_negative_quantized_validation_rate_disables_quantized_validation` はscheduleのフラグだけを検証し、checkpoint保存経路がそのフラグに従って実行を抑止することは検証していない。実行中の旧commit `ce1e364` にも保存分岐の無条件呼び出しが存在する。現在の設定はrate=1なので各save時の量子化検証は予定どおりであり、今回の問題による追加検証はない。GPU実行試験は行っておらず、今回は修正していない。

## 稼働中に実施した修正

学習中の旧実行ファイルとcheckpointには触れず、ソースを修正した。

- checkpoint保存時の量子化検証をscheduleの `run_quantized_validation` に従わせ、rate=0/-1を尊重する。実行しない場合に検証closureが呼ばれないCPUテストを追加。
- epoch境界でFT半精度momentsを後から有効化する場合も、再スケール後の最初のFT更新をFP32に保つ。snapshotに保留フラグを含め、host stateの復元では古い保留フラグを消す。CUDAテストを追加しコンパイルしたが、稼働中のため未実行。
- warmupの途中checkpointをepoch 1へ進めて再開する場合も保存時lrを用いるmomentum再スケール対象とする。通常のwarmup完了および途中切り上げの学習率算術CPUテストを追加。
- directの既存weights.binを削除してからrenameする方式を、旧ファイルを先に削除しない同一volume置換に変更。連続上書きCPUテストを追加。
- summary巻き戻し時、ファイル不存在以外の読込エラーを隠さず返し、再書き込みを一時ファイルからの置換に変更。既存テストに成功。
- CPU整数検証のnearest丸めでi32端付近の加算がoverflowする問題をi64演算で修正し、両端のCPUテストを追加。

これらの修正は稼働中の実行プロセスには反映されない。epoch長変更時のL1 revival抑制期間、FT QAT境界STE、NaN診断、CPU整数検証のエンジン評価値clip、FP16 momentumの量子化に関する問題は未修正。L1抑制は過去の実行履歴を復元できる保存形式と旧checkpointの扱いを設計する必要がある。
