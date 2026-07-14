# 07. nnue-pytorch 互換性調査メモ

BulletOu の大きな NNUE / SFNN model の学習結果を、nodchip 版
`nnue-pytorch` の将棋用 branch と比較するときの差分メモ。

目的は、accuracy / loss の差が出たときに「model そのものの差」なのか
「学習条件の差」なのかを切り分けられるようにすること。

この文書は 2026-07-14 時点の調査結果であり、今後の実装ではここに列挙した
差分を 1 つずつ合わせ、実装前後を比較する。

## 比較対象 branch

`nnue-pytorch` の `master` は、将棋用 SFNN の比較対象としてそのまま使うには危険。
手元 clone の `master` は `model/features/__init__.py` が `halfka_hm` を import しているが、
同 checkout に `model/features/halfka_hm.py` が存在しない状態だった。

SFNN 比較では、実際に学習結果を出した shogi 系 branch を明示すること。

確認済みの代表 branch:

| branch | L1 | LayerStacks | bucket selector |
|---|---:|---:|---|
| `origin/shogi.2026-01-18.sfnnwop-1536` | 1536 | 9 | 自玉段 3 区分 × 敵玉段 3 区分 (`k3k3`) |
| `origin/shogi.2026-04-02.sfnnwop-2048` | 2048 | 8 | 進行度 bucket |

BulletOu の `SFNN_*_k3k3` と直接比較できるのは、基本的には 9 bucket の
`k3k3` 系 branch。8 bucket / progress bucket の branch と比較する場合は、
engine 側の bucket 選択も含めて別 architecture として扱う。

## 差分一覧

### 1. loss / target 変換

nnue-pytorch は WRM 形式の target / prediction 変換を使う。

```text
scorenet = model_output * nnue2score
qf = 0.5 * (1 + sigmoid((scorenet - in_offset) / in_scaling)
              - sigmoid((-scorenet - in_offset) / in_scaling))

pf = 0.5 * (1 + sigmoid((teacher_score - out_offset) / out_scaling)
              - sigmoid((-teacher_score - out_offset) / out_scaling))

target = pf * lambda + outcome * (1 - lambda)
loss   = mean(abs(target - qf) ^ pow_exp)
```

既定値:

| parameter | nnue-pytorch default |
|---|---:|
| `nnue2score` | 600 |
| `in_offset` | 270 |
| `out_offset` | 270 |
| `in_scaling` | 340 |
| `out_scaling` | 380 |
| `pow_exp` | 2.5 |

一方、現在の `examples/bulletou.rs` の NNUE / SFNN 経路は、
基本的に以下の sigmoid MSE で学習している。

```text
target = lambda * sigmoid(teacher_score / scale) + (1 - lambda) * outcome
loss   = (sigmoid(model_output) - target)^2
```

このため、BulletOu の `test_value_loss` と nnue-pytorch の `val_loss` は、
現状では同じ単位の数値ではない。accuracy は符号一致率なので比較しやすいが、
loss は loss 定義を揃えてから比較すること。

### 2. optimizer / learning-rate schedule

nnue-pytorch は `Ranger21` を使う。

主な条件:

| parameter | nnue-pytorch |
|---|---:|
| optimizer | `Ranger21` |
| beta | `(0.9, 0.999)` |
| eps | `1e-7` |
| weight decay | `0.0` |
| scheduler | `StepLR(gamma=0.992)` |
| default lr | `8.75e-4` |

BulletOu の現在の `examples/bulletou.rs` 経路は `AdamW` を使い、
`AdamWParams::default()` では `weight_decay=0.01` かつ全 weight を
一律 `[-1.98, 1.98]` に clip する。AdamW epsilon は従来 `1e-8`。

これは大きな差分。nnue-pytorch 互換性を比較する場合、最低限
`weight_decay=0.0` の ablation を行い、その後 Ranger21 相当を検討する。

scheduler 差分については、既存の BulletOu `--lr-schedule step` は
1 epoch の中で `lr -> lr_min` へ滑らかに落として epoch 境界で warm restart する
独自の geometric schedule であり、nnue-pytorch の `StepLR(gamma=0.992)` とは別物。
比較用に `--lr-schedule step_gamma` を追加した。これは指定局面数ごとに
`lr *= --lr-step-gamma` し、warm restart しない。

nnue-pytorch 既定に寄せる比較例:

```text
--lr 0.000875
--lr-schedule step_gamma
--lr-step-gamma 0.992
--lr-step-positions 100000000
--lr-min 0.00001
```

`--lr-step-positions` を省略した場合は 1 superbatch ごとの decay になる。
比較では明示したほうがよい。

### 3. SFNN の L1 factorized shared term

nnue-pytorch の `LayerStacks` は、L1 に `FactorizedStackedLinear` を使う。

概念的には:

```text
L1_bucket_effective = L1_bucket_specific + L1_shared
```

`L1_shared` は全 bucket 共通の factorized term で、初期値はゼロ。
bucket ごとの個別 weight と共有 weight の和が実効 L1 になる。

現在の `examples/bulletou.rs` の SFNN 経路では、この shared term を入れていない。
過去の実験用 `examples/shogi_layerstack.rs` には `l1f` として類似実装がある。

大きな SFNN で nnue-pytorch より accuracy が劣る場合、この差分はかなり重要。
まず `l1f` あり / なしを同条件で比較する。

### 4. 初期化

feature transformer の初期化は、既に nnue-pytorch 互換方向に修正済み。

nnue-pytorch:

```text
bound = sqrt(1 / num_inputs)
weight, bias ~ uniform(-bound, +bound)
```

BulletOu:

```text
init_nnue_pytorch_feature_transformer_scaled(fan_in, scale)
```

で同系統の初期化を行う。

StackedLinear についても、nnue-pytorch と同じく bucket 0 を初期化して全 bucket にコピーする
helper がある。

ただし、L1 factorized shared term が無い場合、初期化だけ揃えても model class は一致しない。

### 5. weight clipping / quantization scale

nnue-pytorch は量子化スケールから layer ごとに clip 範囲を決める。

| target | max |
|---|---:|
| hidden weight | `127 / 64 = 1.984375` |
| output weight | `127 * 127 / (600 * 16) = 1.680104...` |

また L1 bucket weight は、factorized shared weight を足した実効 weight が範囲内に入るように
clip される。

BulletOu の標準 AdamW は全 weight に一律 `[-1.98, 1.98]` を適用する。
output weight の上限が nnue-pytorch より広く、L1 factorized term も考慮していない。

`--nnue-pytorch-layer-clip` を付けると、hidden weight は `[-127/64, 127/64]`、
final output weight だけは `[-127*127/(600*16), 127*127/(600*16)]` にする。

nnue-pytorch の `WeightClippingCallback` は `l1.linear.weight`, `l2.linear.weight`,
`output.linear.weight` だけを clip 対象にしており、bias は clip しない。
BulletOu の AdamW は標準では bias も含めて全 parameter を clip するため、
`--nnue-pytorch-no-bias-clip` で bias tensor の clip を実質無効化できるようにした。

### 6. FeatureSet の一致

nnue-pytorch の sfnnwop 系 branch は `HalfKA_hm` を使う。

BulletOu で比較するときは、比較対象に応じて `--eval-type` を揃える。

| nnue-pytorch feature | BulletOu eval-type |
|---|---|
| `HalfKA_hm` | `SFNN_HALFKA2HM` など HM 系 |
| `HalfKA2` | `SFNN_HALFKA2` |

`SFNN_HALFKA2` と `HalfKA_hm` の結果を直接比較してはいけない。

### 7. LayerStack bucket

`k3k3` の 9 bucket は、以下の対応。

```text
       enemy king rank 0-2  3-5  6-8
friend king rank 0-2     0    1    2
friend king rank 3-5     3    4    5
friend king rank 6-8     6    7    8
```

`origin/shogi.2026-01-18.sfnnwop-1536` の C++ data loader と
BulletOu の `ShogiKingRankBucket<9>` は、この mapping が一致している。

一方 `origin/shogi.2026-04-02.sfnnwop-2048` は progress 由来の 8 bucket であり、
`k3k3` とは一致しない。

## 実装・比較の推奨順

一度に全部合わせると、どれが効いたかわからなくなる。
以下の順で 1 つずつ入れて、同じ教師・同じ検証 set・同じ seed 条件で比較する。

1. 比較対象 branch / architecture を固定する
   - 例: `shogi.2026-01-18.sfnnwop-1536`
   - BulletOu: `SFNN_HALFKA2HM`, `SFNN_halfkahm2_1536_15_32_k3k3`

2. L1 factorized shared term (`l1f`) を BulletOu の SFNN 経路に追加する
   - 初期値はゼロ
   - save 時には bucket weight に fold できる
   - 実装済み: `--sfnn-factorized-l1` で opt-in

3. loss を nnue-pytorch WRM 互換にする
   - `nnue2score=600`
   - `in_offset=270`, `out_offset=270`
   - `in_scaling=340`, `out_scaling=380`
   - `pow_exp=2.5`
   - 実装済み: `--nnue-pytorch-wrm-loss` で opt-in
   - 有効時は学習 loss だけでなく `test_value_loss` / plateau 判定も WRM loss に切り替わる

4. optimizer 条件を近づける
   - まず `AdamW weight_decay=0.0`
   - 実装済み: `--adamw-weight-decay 0.0` で opt-in
   - 実測では悪化する場合があるので、他の ablation と混ぜず単独で比較する
   - `eps=1e-7` は `--adamw-epsilon 0.0000001` で AdamW のまま単独比較する
   - `beta1` / `beta2` は `--adamw-beta1` / `--adamw-beta2` で AdamW のまま単独比較する
     - デフォルトは `0.9` / `0.999` で、nodchip nnue-pytorch の Ranger21 設定と同じ
   - その後 Ranger21 相当の実装・比較

5. layer-specific clipping を入れる
   - hidden と output を別範囲にする
   - L1 factorized term がある場合は実効 weight 基準で clip する
   - 実装済み: `--nnue-pytorch-layer-clip` で opt-in
   - 現時点の実装は final output weight の clip 差分だけを検証するためのもの。
     factorized L1 の実効 weight clip は未対応。
   - bias clip 無効化は `--nnue-pytorch-no-bias-clip` で別 ablation として試す

6. data loader / epoch / scheduler 条件を揃える
   - nnue-pytorch default epoch size は 100,000,000 positions
   - `StepLR(gamma=0.992)` と BulletOu 側 schedule を区別する
   - 実装済み: `--lr-schedule step_gamma --lr-step-gamma 0.992 --lr-step-positions 100000000`

## 比較時の注意

- `test_value_accuracy` は符号一致率なので比較可能。
- `test_value_loss` は loss 定義を揃えるまでは比較しない。
- `SFNN_HALFKA2` と `SFNN_HALFKA2HM` は別 feature set。
- 8 bucket progress model と 9 bucket `k3k3` model は別 architecture。
- 現在 checkout している nnue-pytorch の branch 名と commit hash をログに残す。

## 関連コード

nnue-pytorch 側:

- `train.py`
- `model/lightning_module.py`
- `model/model.py`
- `model/modules/layer_stacks.py`
- `model/modules/stacked_linear.py`
- `model/modules/feature_transformer/module.py`
- `model/quantize.py`
- `training_data_loader.cpp`

BulletOu 側:

- `examples/bulletou.rs`
- `examples/shogi_layerstack.rs`
- `crates/trainer/src/model/builder.rs`
- `crates/trainer/src/optimiser/adam.rs`
- `crates/bulletou_lib/src/value/loader.rs`
- `crates/bulletou_lib/src/validate.rs`
- `crates/bulletou_lib/src/game/outputs.rs`
