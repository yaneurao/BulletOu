# SGD + Momentum 文献調査

調査日: 2026-11-18。目的はBulletOuのSGD + Momentumによる学習を理解し、lossとqlossが下がらない原因の調査に役立てること。ここに挙げる理論は現象を調べる手掛かりであり、BulletOuの不具合の実証ではない。

## 先に結論

- Momentumの係数と学習率は独立に解釈できない。論文と実装で更新式が異なるため、数値をそのまま持ってきて比較できない。
- factorizerの共有パラメーターを通じた更新は、個別重みの更新と異なる結合を作る。共通の学習率でも層やbucketの実効的な変化は同じにならない。
- 量子化重みの2時点の差だけでは、その間の変化の回数は分からない。量子化境界を繰り返し往復している可能性も、実際にほとんど動いていない可能性もある。
- 文献だけで現在のloss上昇の原因や適切なlr・momentumを確定することはできない。量子化を外した条件でもloss上昇は観測済みであり、QATだけを原因と決めつけることもできない。

## 更新式の読み替え

BulletOuの通常のSGD + Momentumを、勾配を `g`、学習率を `η`、momentum係数を `β` として概念的に書くと次の通り。

```text
m_t = β m_(t-1) + g_t
w_(t+1) = w_t - η_t m_t
```

文献には `m'_t = β m'_(t-1) + (1-β) g_t` という正規化された定義もある。βが一定で初期状態も対応するとき、`m'=(1-β)m` なので、同じ重み更新を得るには `η_BulletOu = (1-β) η_正規化式` とする。βが時間で変わる場合にはこの単純な換算はそのまま使えない。[Liu et al.](https://proceedings.neurips.cc/paper/2020/hash/d3f5d4de09ea19461dab00590df91e4f-Abstract.html) は両方の表記に触れている。

勾配が一定で十分時間がたった仮定では `m ≈ g/(1-β)`。したがって `η/(1-β)` はその条件での実効的な係数になる。たとえばβ=0.9では10倍だが、毎ステップの勾配が10倍になるという意味ではなく、学習率の推奨値でもない。過去の勾配の寄与は k step後に `β^k` 倍となる。β=0.9なら半減まで約6.6 step。

また、`m=βm+g; w-=η_t m` と `v=βv-η_t g; w+=v` は学習率一定なら対応できるが、学習率を途中変更すると一般に等価ではない。論文や他実装と比較するときはmomentum bufferの定義と学習率を掛ける位置も確認する。

## 安定性と疎な更新についての整理

単一の決定論的な二次関数 `L(x)=λx²/2`、一定のη、`0≤β<1` という非常に限定された場合、更新は `x_(t+1)=(1+β-ηλ)x_t-βx_(t-1)` となり、原点への漸近安定条件は `0<ηλ<2(1+β)` である。実際のニューラルネットワーク、確率勾配、STEや量子化を含むBulletOuにこの条件をそのまま適用して安全な学習率を決めることはできない。

ある入力特徴の1局面あたりの出現確率をq、バッチサイズをBとし、局面間独立と仮定すれば、そのバッチに1回以上出る確率は `1-(1-q)^B`。独立性は実データでは保証されない。また、平均勾配は出現時の寄与に出現頻度が掛かったものになり得る。頻度が異なるパラメーターでは、同じ学習率・同じmomentumでも時間的な更新のされ方が異なる。勾配が0のstepでもmomentumが残っていれば重みは動く。

factorizerの効果を示す簡略例として、実効重みを `z_i=w_i+αs`、`g_i=∂L/∂z_i` とする。全パラメーターに同じηの普通のSGDを適用し、ほかの項を無視すれば、1 stepの実効重みの変化は `Δz_i=-η(g_i+α²Σ_j g_j)`。共有成分を通じて他の重みの勾配が作用する。実際のBulletOuは特徴の対応、層、momentum、量子化などを含むので、この式の数値をそのまま実測値に使うことはできない。forward値が同じになるようfactorizer係数を切り替えても、以降の更新の幾何は変わり得る。

## 読んだ文献と適用上の注意

1. **Sutskever et al., 2013, [On the importance of initialization and momentum in deep learning](https://proceedings.mlr.press/v28/sutskever13.html)**。深いネットワークとRNNで、初期化と徐々に増加させるmomentumを組み合わせて学習を改善した。初期化とmomentumを一体で検討する根拠になるが、実験対象はBulletOuと異なり、そのまま係数を推奨できない。

2. **Liu, Gao, Yin, 2020, [An Improved Analysis of Stochastic Gradient Descent with Momentum](https://proceedings.neurips.cc/paper/2020/hash/d3f5d4de09ea19461dab00590df91e4f-Abstract.html)**。滑らかな目的関数について、SGDと同等の収束速度や段階的なパラメーター変更を解析。更新式の正規化の違いにも注意が必要。滑らかさの仮定があるため量子化済みforwardの挙動を直接保証しない。

3. **Kidambi et al., 2018, [On the insufficiency of existing momentum schemes for Stochastic Optimization](https://arxiv.org/abs/1803.05591)**。確率最適化の単純な問題でもmomentumによる加速を保証できない例を示した。momentumを増やせば必ず速くなるという考えへの反例であり、BulletOuでmomentumが使えないことを示すものではない。

4. **Wang et al., 2024, [The Marginal Value of Momentum for Small Learning Rate SGD](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3a6935d11910d6f9142b0a1e36fc6753-Abstract-Conference.html)**。小さな学習率の領域で、実効学習率を対応させるとSGDとmomentumの差が小さくなり得ることを解析。比較時に `lr` だけを揃えて結論を出すべきではない。ただし、常に両者が等価という意味ではない。

5. **Smith, Elsen, De, 2020, [On the Generalization Benefit of Noise in Stochastic Gradient Descent](https://proceedings.mlr.press/v119/smith20a.html)**。学習率・バッチサイズと、ノイズや曲率が学習に及ぼす影響を分析。バッチサイズに比例して学習率を上げる方法は全領域で成り立つわけではない。対象のモデルと実験条件が違うため、BulletOuのバッチサイズから最適lrは求められない。

6. **Goyal et al., 2017, [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)**。画像分類で大バッチ、線形の学習率スケーリング、warmupを組み合わせた実験。大バッチの実用例だが、異なるネットワークと学習条件へ比例則を無条件に持ち込める根拠ではない。

7. **Lee et al., 2022, [Trajectory of Mini-Batch Momentum: Batch Size Saturation and Convergence in High Dimensions](https://papers.neurips.cc/paper_files/paper/2022/hash/efcb76ac1df9231a24893a957fcb9001-Abstract-Conference.html)**。高次元の最小二乗モデルでバッチサイズとmomentumの学習軌道を解析。バッチが大きいほど常に効果が増すわけではなく、曲率分布なども重要。解析モデルとBulletOuは異なる。

8. **Everett and Paquette, 2026, [Dynamics of Stochastic Momentum with Sparse Updates in High Dimensions](https://arxiv.org/abs/2605.28961)**。疎な入力の最小二乗と希少クラスのモデルで、momentum保持時間と学習時間の関係を解析。出現頻度が異なる特徴に共通のmomentumを適用したときの違いを考える手掛かりになる。調査時点でプレプリントであり、factorizer付き将棋評価関数に対する実証ではない。

9. **Andreyev et al., 2026, [Momentum Further Constrains Sharpness at the Edge of Stochastic Stability](https://proceedings.mlr.press/v306/andreyev26a.html)**。確率的な設定では平均曲率だけでなくバッチごとの曲率の違いも安定性に関係することを調べる。そこに現れる閾値は仮定や更新式に依存するため、その数値をBulletOuの安全条件として扱わない。

10. **Lee et al., 2025, [Scheduling Weight Transitions for Quantization-Aware Training](https://openaccess.thecvf.com/content/ICCV2025/html/Lee_Scheduling_Weight_Transitions_for_Quantization-Aware_Training_ICCV_2025_paper.html)**。QATで潜在重みが量子化境界付近を往復し、forwardに使う離散重みが振動する問題を調べる。学習率を下げても振動が必ず消えるとは限らない。主な実験条件とBulletOuのint8・bucket構造は異なり、振動が現在の原因だと証明するものではない。

11. **Ichikawa, Kashiwamura, Sakata, 2026, [High-Dimensional Learning Dynamics of Quantized Models with Straight-Through Estimator](https://proceedings.mlr.press/v306/ichikawa26a.html)**。限定した量子化モデルのSTE学習を解析し、停滞の後に誤差が下がる例や量子化範囲の影響を示した。解析モデルはBulletOuそのものでもmomentum付き学習の一般理論でもない。現在のloss上昇を「待てば改善する停滞」と判断する根拠にはならない。

12. **Huh et al., 2023, [Straightening Out the Straight-Through Estimator](https://proceedings.mlr.press/v202/huh23a.html)**。ベクトル量子化でSTEの勾配と実際の最適化のずれを研究。STEの代用勾配を無条件に真の量子化目的関数の勾配とみなせないという概念上の参考。ただし、対象はBulletOuの重み量子化とは異なり、BulletOuに特定の勾配バグがある証拠ではない。

13. **Micikevicius et al., 2017, [Mixed Precision Training](https://arxiv.org/abs/1710.03740)**。混合精度学習でのFP32 master weightやloss scalingを説明。FP16の表現範囲・丸めと、QATの整数化境界は別の問題である。FP32 master weightも、FP32自体の丸めにより小さい更新が失われることまでは防がない。

14. **Duchi et al., 2011, [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](https://jmlr.org/papers/v12/duchi11a.html)**。特徴ごとに更新を調整する方法と疎な特徴を論じる。頻度差と更新スケールを理解する参考だが、ここでの目的はSGD + Momentumの修正・理解であり、別optimizerへの切り替えを提案するものではない。

15. **Nagel et al., 2022, [Overcoming Oscillations in Quantization-Aware Training](https://proceedings.mlr.press/v162/nagel22a.html)**。量子化重みの境界付近の往復を分析し、整数値の変化方向が前回の変化方向と逆になった回数から振動頻度を測る。SGD momentum=0.9を使った実験も含む。主な実験は学習済み画像モデルの3〜4 bit量子化であり、BulletOuの8 bit SFNNで同じ問題が発生することや、論文の重み固定が有効なことは示していない。

16. **Mai and Johansson, 2021, [Stability and Convergence of Stochastic Gradient Clipping](https://proceedings.mlr.press/v139/mai21a.html)** および **Zhang et al., 2020, [Improved Analysis of Clipping Algorithms for Non-convex Optimization](https://proceedings.neurips.cc/paper/2020/hash/b282d1735283e8eea45bce393cefe265-Abstract.html)**。一定の条件下で勾配クリップとmomentumの安定性を解析する。ただし、BulletOuのFP16 momentumの要素ごとの表現範囲によるクリップは、これらの手法の勾配ノルムクリップとは異なる。論文を根拠に現在のクリップが有益・無害と判断してはならない。

## 追加調査: factorizerとFT FP16 momentum

共通係数αの単純なfactorizer `z_i=w_i+αs` をN個のbucketについて考える。全体をベクトルにすると、同じlrのSGDによる実効重みの変化は `Δz=-η(I+α²11ᵀ)g`。全bucketで共通の方向の倍率は `1+Nα²`、共通方向と直交する方向の倍率は1である。これは同一ηで全パラメーターを更新し、αが一定である場合の正確な代数である。損失が全局面平均なら各成分の勾配にも頻度が反映されるため、倍率をそのまま実際の更新量と混同しない。複数軸、特徴対応、別lrやQATがある実装では行列が変わる。示唆は、factorizerが単に初期値を共有するだけでなく、共通方向とbucket固有方向の学習率の比にも影響するということ。これは原因候補であり、現状のloss上昇の実証ではない。

CUDAのSGD実装ではFTのFP16 momentumを `2^18` 倍して保存する。そのためFP16の最大有限値から逆算した保存可能なmomentumの絶対値は `65504 / 262144 = 0.2498779296875`。更新時に計算されたmomentumでそのstepの重みを更新した後、保存時に範囲外の値をクリップするため、次stepはクリップ後の値を使う。範囲内でもFP16への丸めは発生する。superbatchの警告件数はクリップされた**要素×更新の回数**で、異なる重みの個数ではない。過去の警告が今回のSGD実行時のものか、またlossにどれほど影響したかは別途ログと設定の照合が必要。FP16勾配のクリップは別の警告であり混同しない。

## 既存観測と次に必要な測定

以前調べた2 sbのcheckpoint比較では、L2の量子化値について開始時と終了時で異なる要素が65個だった。これは**途中での変化の総数ではない**。以前これを「ほとんどforwardに反映されていない」と説明したのは根拠不足だった。途中の往復、影響する局面、出力への寄与はこの比較からは分からない。

原因を絞るには、同一の保存状態と同一バッチを使い、通常lossとqlossの変化と対応させながら、以下を測る必要がある。

- L1/L2/L3、shared/residualごとの勾配、momentum、実際の重み更新量。重みの大きさだけでなく、出力変化への寄与も見る。
- 更新ごとに量子化値が変化した個数、同じ境界の往復回数、変化した重みの出力への影響。開始・終了差だけでは代用できない。
- QATのforwardと勾配の関係を、同じ状態・同じ入力で比べる。既存のQATなしのloss上昇は、QATがなくても上昇し得るという観測であり、各条件を完全に一致させた比較として扱わない。
- factorizerを含む個別・共有成分の更新と、切り替え前後のoptimizer状態。係数変更時にforwardが維持されても更新方向まで同一とは限らない。

これらの測定は必要な場合に限定して行う。対局による棋力比較は行わない。文献調査時点では新たな学習を開始していない。
