# 08. cuda-oxide による BulletOu 高速化調査

BulletOu を tatara 方式で高速化するための調査メモ。

調査対象は、同一 workspace に clone した `../tatara` と、現在の
BulletOu の GPU / trainer 実装である。目的は「cuda-oxide に書き換えると
何が速くなるのか」「何を移植すべきか」「既存 BulletOu とどう共存させるか」を
仕様レベルで整理すること。

## 結論

tatara が速い主因は、単に CUDA kernel を cuda-oxide で書いていることではない。
主因は以下の 3 点である。

1. NNUE / SFNN 用に architecture を固定している
2. forward / backward / loss / optimizer を手書き融合 kernel にしている
3. dataloader と GPU workspace が固定形状で、汎用 map / TensorIR の overhead を避けている

そのため、BulletOu の `bullet-gpu` backend を cuda-oxide に単純置換するだけでは
tatara 相当にはならない。むしろ、runtime fusion を失って遅くなる可能性がある。

推奨する方針は、既存の汎用 BulletOu backend は残し、shogi NNUE 専用の
cuda-oxide backend を別経路として追加することである。

```text
既存経路:
  BulletOu CLI
    -> bullet-trainer
    -> bullet-gpu (CUDA Driver API + NVRTC / ROCm)
    -> 汎用 TensorIR / PointwiseIR

追加する経路:
  BulletOu CLI または bulletou-cuda-train
    -> Shogi NNUE 専用 trainer
    -> cuda-oxide runtime
    -> build-time PTX + 手書き fused kernels
```

## tatara の構成

tatara は Rust 製の NNUE 高速学習器で、GPU kernel を cuda-oxide で
build-time に PTX へ compile する。

代表的な構成:

| path | 役割 |
|---|---|
| `bins/nnue_train` | training binary。cuda-oxide の `#[kernel]` 定義もここに置く |
| `crates/nnue-train` | host-side training loop / dataloader / schedule |
| `crates/gpu-runtime` | cuda-oxide host API の薄い wrapper |
| `crates/gpu-kernels` | CPU reference 実装と kernel 単位の検証用実装 |
| `crates/shogi-features` | shogi NNUE feature extraction |
| `scripts/build-kernels.sh` | cargo-oxide で NVVM IR を作り、LLVM で PTX 化 |

cuda-oxide は build-time compiler であり、BulletOu のように runtime に
kernel source を生成して NVRTC で compile する方式ではない。
この制約により、device `#[kernel]` は実行 binary から到達可能な場所に置く必要がある。

tatara はこの制約を受け入れ、NNUE 専用の fused kernel を手書きしている。

## BulletOu の現行 GPU 実装

現在の BulletOu は、上流 bullet 系の汎用 GPU runtime を使う。

代表的な構成:

| path | 役割 |
|---|---|
| `crates/gpu/src/runtime/cuda.rs` | CUDA Driver API + NVRTC による runtime compile |
| `crates/gpu/src/runtime/rocm.rs` | HIPRTC による ROCm backend |
| `crates/gpu/src/pointwise/ir.rs` | pointwise kernel の IR |
| `crates/gpu/src/pointwise/transforms.rs` | pointwise fusion と codegen |
| `crates/trainer/src/model.rs` | TensorIR から生成された forward / backward graph 実行 |
| `crates/trainer/src/run.rs` | 汎用 training loop |
| `crates/bulletou_lib/src/value.rs` | shogi value trainer wrapper |

現行方式の利点:

- 任意の `ModelBuilder` graph を扱える
- CUDA / ROCm の両 backend を持てる
- optimizer / loss / layer 構成を比較的自由に差し替えられる

現行方式の弱点:

- batch input が `BTreeMap<String, TValue>` 経由で汎用的
- forward / backward は generic graph execution で、NNUE 専用ではない
- loss readback が batch ごとに同期点になりやすい
- optimizer は tensor group ごとの汎用更新で、NNUE 全体を見た融合が難しい
- runtime NVRTC compile と PointwiseIR fusion の仕組みが cuda-oxide と相性が悪い

## tatara が速い理由

### 1. 手書き fused kernel

tatara は pointwise sequence を cuda-oxide kernel に手で融合している。

代表例:

| kernel pattern | 効果 |
|---|---|
| `loss_wdl` / `loss_wrm` | sigmoid / target blend / loss reduction をまとめる |
| `screlu_grad` | activation gradient をまとめる |
| `radam_step` | m / v / bias correction / weight update を 1 kernel 化 |
| `ranger_lookahead_lerp` | Lookahead slow weight 更新を専用化 |
| `ft_post_perspective_*` | bias add / CReLU / pairwise-mul / scale をまとめる |
| sparse FT backward 系 | active feature の inverse index を作って atomic / gather を制御 |

cuda-oxide は runtime fusion ができないため、これをやらずに 1 op = 1 kernel に
分解すると、memory bandwidth bound で大きく遅くなる。

### 2. 固定 workspace

tatara の `GpuTrainer` は、weight / grad / optimizer state / activation /
temporary buffer を trainer 構築時に確保し、step ごとには再利用する。

BulletOu も tensor 自体は再利用しているが、graph 実行・入力 map・汎用 tensor
operation の abstraction が残る。tatara は shogi NNUE 専用の buffer layout に
寄せているため、kernel launch と memory traffic の見通しが良い。

### 3. 非同期 loss ring / input upload ring

tatara は loss の readback を毎 step 同期しないように、非同期 ring を使う。
また input upload も ring 化し、H2D copy と compute の overlap を狙っている。

BulletOu の現行 loop は double-buffering を持つが、loss 取得や汎用 batch 処理で
同期点が増えやすい。

### 4. dataloader の固定形状化

tatara の dataloader は `PackedSfenValue` を decode し、fixed-capacity の
`Batch` に sparse index / score / wdl / bucket を詰める。
`BTreeMap<String, TValue>` のような汎用構造を GPU step の手前に置かない。

BulletOu は HCPE / HCPE3 / PSV など複数形式を扱うため汎用性が高いが、
cuda-oxide 専用 backend では固定 batch struct に落としたほうが速い。

## 移植でやってはいけないこと

`crates/gpu` を cuda-oxide に置き換えて、既存 TensorIR をそのまま動かす設計は
避けるべきである。

理由:

- cuda-oxide は build-time PTX なので、runtime に任意 graph から kernel を
  生成できない
- PointwiseIR の runtime fusion が使えなくなる
- ROCm backend が失われる
- 既存の汎用 `ModelBuilder` を維持しながら cuda-oxide の利点を出すのは難しい
- 手書き kernel に落とせない graph は結局遅くなる

cuda-oxide 化は「汎用 backend の置換」ではなく「shogi NNUE 専用 backend の追加」と
考えるべきである。

## 推奨アーキテクチャ

### backend 分離

最初は既存 CLI に `--backend cuda-oxide` を足すより、専用 binary を作るほうが安全。

候補:

```text
examples/bulletou.rs                 # 既存の汎用 backend
bins/bulletou_cuda_train             # 新規 cuda-oxide 専用 backend
```

ただし、最終的には CLI から以下のように選べる形が望ましい。

```text
--backend bullet       # 既定。現行 backend
--backend cuda-oxide   # NVIDIA / Linux-WSL2 専用の高速 backend
```

### crate 構成案

```text
crates/cuda_oxide_runtime/
  cuda-oxide host API wrapper
  PTX / CUBIN loader
  kernel launch helper

crates/shogi_nnue_cuda_kernels/
  CPU reference kernels
  数値同等性テスト

bins/bulletou_cuda_train/
  cuda-oxide #[kernel] definitions
  ShogiNnueCudaTrainer
  build.rs / PTX artifact handling
```

cuda-oxide の bin-entry reachability 制約により、device `#[kernel]` は
binary crate 側に置く。CPU reference と host 側 helper は library crate に置く。

## 実装フェーズ

### Phase 0: baseline 計測

まず現行 BulletOu で以下を固定して記録する。

| 項目 | 内容 |
|---|---|
| GPU | GPU 名、driver、CUDA toolkit |
| teacher | 同じ teacher path |
| eval type | まず `SFNN_HALFKA2` または `NNUE_HALFKP` |
| arch | 例: `SFNN_halfka2_1024_7_64_k3k3` |
| batch size | 現行 default と明示値 |
| positions / superbatch | 明示値 |
| optimizer / scheduler | 同一条件 |
| metrics | positions/sec, accuracy, loss |

高速化の比較では、accuracy / loss が変わったら速度比較にならない。
まず fp32 の数値同等性を優先する。

### Phase 0.5: 既存 backend での即効改善

cuda-oxide 専用 backend へ進む前の低リスクな改善として、`--batch-size` 省略時の
デフォルトを `65536` に寄せる。

この変更の狙い:

- tatara の代表的な学習条件と同じ batch 粒度に近づける
- 同じ局面数を処理するときの batch 数を減らし、host 側 batch 構築・GPU step 呼び出し・
  ログ処理の overhead を減らす
- `--positions-per-superbatch` 指定時は `floor(positions / batch_size) * batch_size`
  へ丸める既存仕様を維持する

これは fused kernel 化ではないため、tatara 相当の速度にはまだ届かない。
本命の改善は Phase 1 以降の NNUE/SFNN 専用 cuda-oxide backend で行う。

### Phase 0.6: dataloader queue tuning

`--batch-queue-size` は CLI / `LocalSettings` には存在していたが、低層の
training loop では固定値 `32` の同期 channel が使われていた。

このため、ユーザーが大きい batch queue を指定しても dataloader の先読み量は
増えず、CPU decode / batch 構築が GPU に追いつかない条件で調整できなかった。

既存 backend では、`LocalSettings::batch_queue_size` を
`bullet_trainer::run::TrainingSchedule` に渡し、`sync_channel` の容量として
そのまま使う。`0` が渡った場合は安全側で `1` に丸める。

これは kernel 自体を速くする変更ではないが、HCPE / HCPE3 decode が重い条件で
GPU 側を空転させにくくするための土台である。cuda-oxide backend でも同じ考え方で、
host 側の decode queue と device upload ring を明示的に分離する。

### Phase 0.7: H2D copy / compute overlap

現行 loop では、次 batch の `copy_to_device_async()` の返り値を即 `drop` していた。
この返り値は `SyncOnDrop` を含むため、即 drop は copy stream の即時同期を意味する。

その結果、次 batch の H2D copy は current batch の GPU compute と十分に overlap せず、
copy 完了を待ってから current batch の compute sync に進む形になっていた。

既存 backend 側では、次 batch の host buffer を生存させたまま H2D copy を enqueue し、
current batch の compute sync を先に待ってから copy stream を同期する。これにより、
次 batch の upload と current batch の backward / optimizer update を重ねられる。

これは cuda-oxide の本命である input upload ring の小さい前段である。
完全な ring 化ではないが、既存の double buffer 構造のまま安全に overlap を増やせる。

さらに、次 batch が dataloader queue にすでに到着している場合は、current batch の
backward/update を enqueue する前に `try_recv` で取得し、`next_on_device` への H2D copy を
先行 enqueue する。`try_recv` なので dataloader が遅い場合には compute 開始をブロックしない。
queue に余裕がある条件では、next batch upload と current batch compute の overlap 時間を
より長く取れる。

あわせて、複数 input tensor の H2D copy を tensor ごとの `SyncOnDrop` ではなく、
1 つの `SyncOnDrop` にまとめる。これにより、同じ copy stream への同期呼び出しを
input tensor 数ぶん繰り返す無駄を避ける。

また、`SyncOnDrop::sync(self)` は明示 sync 後に `Drop` で再度 sync しないようにする。
従来は `sync(self)` が `self` を消費したあと `synced=false` のまま drop されるため、
明示 `sync()` が二重の stream sync になっていた。

learning-rate scalar と gradient-factor scalar の upload も同じ copy stream 上で
連続して enqueue される。個別の `SyncOnDrop` を個別に `sync()` すると同じ stream を
2 回待つだけなので、`SyncOnDrop::merge()` で guard をまとめ、update 前に 1 回だけ
copy stream を同期する。

optimizer update でも同じ問題がある。weight ごとの update kernel は同じ compute stream に
enqueue されるため、`OptimiserUpdateSync::sync()` が kernel guard ごとに `value()` を呼ぶと、
同じ stream を weight 数ぶん待つことになる。kernel / copy guard を `SyncOnDrop::merge()` で
1 つに集約し、optimizer update 全体につき 1 回だけ compute stream を同期する。
この集約用の `OptimiserUpdateSync` も batch ごとの短命な container なので、weight 数や
optimizer 種別から分かる範囲で容量を事前確保し、update 時の `Vec` 伸長を減らす。
optimizer state と model weights は同じ重み集合から作られるため、update 時は key lookup ではなく
sorted map 同士を zip し、weight 取得の `BTreeMap::get` を避ける。
通常の学習経路では gradient map も同じ重み集合を持つため、key 集合が一致する場合は
state / weight / gradient をまとめて zip し、gradient 取得の `BTreeMap::get` も避ける。
一致しない場合は従来の lookup 経路へ落として互換性を保つ。
さらに hot path では optimizer state が小さな `OptimiserUpdateSync` を作って返す代わりに、
呼び出し側の集約syncへ直接pushする `update_into` を使う。これにより weight ごとの短命な
kernel/copy list allocation を避ける。

### Phase 0.8: one-batch delayed loss readback

loss readback は `outputs/loss` buffer を GPU から CPU に戻すための同期点になる。
ただし、`outputs/loss` を 1 面だけで使っている場合、D2H copy を未同期のまま次 batch の
backward に進むと、次 batch が同じ output buffer を compute stream で取得できない。

既存 backend では、backward output tensor を 2 面化し、通常 batch では loss D2H を
1 batch だけ遅延する。

```text
batch N:
  output slot 0 に backward
  loss slot 0 の D2H を loss stream に enqueue して待たない

batch N+1:
  output slot 1 に backward / update
  batch N の loss D2H をここで回収
```

このとき、loss readback 用に upload stream とは別の `loss_stream` を使う。
同じ stream を使うと、次 batch 冒頭の learning-rate scalar upload が loss readback を
待ってしまい、遅延の意味が薄くなるためである。

checkpoint の重みタイミングを壊さないため、superbatch 末尾の loss は遅延しない。
末尾 batch は即座に loss を回収し、running loss を確定してから save callback を呼ぶ。
したがって、checkpoint は従来通り「その superbatch の最後の update 直後」の重みになる。
`outputs/loss` buffer は output tensor 作成後に変わらないため、batch loop の外で `Arc` を
保持しておき、loss readback のたびに output map を lookup しない。

### Phase 0.9: avoid per-batch prefixed tensor map rebuild

現行 `Model::forward` / `Model::backward` は、毎 batch で以下のような中間 map を作っていた。

```text
weights/<id>   -> weight buffer
inputs/<id>    -> input buffer
gradients/<id> -> gradient buffer
<output id>    -> output buffer
```

この方式は単純だが、weight 数が多い network では batch ごとに prefix 付き key 文字列と
`BTreeMap` を作り直すことになる。既存 backend のままでも、ここは `Function` が要求する
tensor 名を prefix 解決して、直接該当する map から buffer を引けばよい。

この変更は NodeId に渡す buffer 対応を変えず、host 側の余分な allocation / lookup だけを
減らす。cuda-oxide backend ではさらに進めて、name lookup 自体を build-time layout に
落とす。

さらに、`Function::execute` に `BTreeMap<NodeId, Buffer>` を渡す必要もない。
`Function` 側は入力順序を要求せず、NodeId ごとに内部 pointer slot へ配置しているだけなので、
`execute_bindings` で `(NodeId, Buffer)` の iterator を受け取れるようにする。
これにより `Model::forward` / `Model::backward` は NodeId→Buffer の中間 `BTreeMap` を作らず、
小さな binding 配列から直接 kernel 実行へ進める。

最後に、tensor 名の prefix 分類も graph 構築時に固定する。
`weights/` / `inputs/` / `gradients/` / output のどこを見るかを `TensorSource` として
保持しておけば、batch 中の `strip_prefix()` と一時 binding 配列 allocation も不要になる。
batch 中は固定済み binding を走査し、該当する tensor map から `Arc<Buffer>` を clone するだけでよい。
さらに weight tensor は学習中に同じ buffer を使い続けるため、`TensorBinding` 構築時に
`Arc<Buffer>` へ直接解決しておく。これにより forward/backward の各 batch で weight 名を
`BTreeMap` lookup する処理を避け、動的 lookup は input / output / gradient だけに限定する。

kernel launch 引数も同様に、launch ごとに小さな `Vec` を作る必要はない。
`Function::execute` 内で `max_num_args` 分の引数 buffer を 1 つ確保し、各 kernel launch で
`clear()` して再利用する。これは GPU 計算順序や数値に影響しない host-side allocation 削減である。

optimizer の weight ごとの update kernel は `CompiledKernel::execute` を大量に呼ぶため、
ここでも小さな allocation が積み重なる。可変 batch size の一致確認は `BTreeSet` ではなく
`Option<usize>` で十分なので、set allocation を避ける。また launch 引数 `Vec` は必要容量を
事前に確保する。
さらに `CompiledKernel` にも scratch buffer を持たせ、input/output device pointer、
launch 引数、device pointer 引数配列を kernel 呼び出しごとに再利用する。guard は
`SyncOnDrop` に直接 attach し、scratch には launch 中だけ必要な host-side pointer 配列だけを
置く。

さらに `CompiledKernel::execute_slices` を追加し、Adam/RAdam/Ranger の update 呼び出しでは
`vec![...]` ではなく stack 上の配列参照を渡す。各 weight ごとの optimizer kernel launch で
入力/出力 buffer リスト用の heap allocation を避けるための変更であり、GPU kernel の内容や
同期順序は変えない。
同じく optimizer の補助 kernel である weight decay / weight clipping も
`execute_slices` 経由に揃え、pre/post update の小さな allocation を避ける。

forward/backward では `TensorBinding` を事前計算しているが、実行時の binding 解決で
`Arc<Buffer>` を clone して `Function` に渡す必要はない。`Function::execute_binding_refs`
を追加し、`Model::forward/backward` は `&Arc<Buffer>` を渡す。`Buffer::acquire` が必要な
guard を作るため、ownership/sync の意味は変わらず、binding 解決時の atomic refcount 操作を
減らせる。

`Function` の alias check は入力数が小さいため、毎回 `BTreeMap` を構築するよりも
`Vec<(DevicePtr, bool)>` の線形探索で十分である。これにより forward/backward 呼び出しごとの
map node allocation を避ける。

kernel launch の size 引数用 buffer は `max_num_args` 個をゼロ初期化する必要はない。
`Vec::with_capacity(max_num_args)` で容量だけ確保し、各 launch で実際に必要な size 引数だけを
push する。事前確保により launch 直前の pointer は安定し、GPU に渡す値は従来と同じである。

さらに `Function` に `FunctionScratch` を持たせ、`ptrs` / `aliases` / `sizes` /
`kernel_args` を forward/backward 呼び出しごとに再利用する。kernel launch は host pointer 配列を
launch 呼び出し中に読むだけなので、launch 後に scratch を再利用しても GPU 側の実行内容は変わらない。
これにより graph 実行ごとの短命な `Vec` allocation を避ける。

`SyncOnDrop` は attach される guard 数が分かっている呼び出し経路では容量を事前確保する。
`Function::execute_binding_refs` と `CompiledKernel::execute_slices` で guard list の再確保を避ける。
batch upload の `PreparedBatchHost::copy_to_device_async` でも、device tensor 数に合わせて
同じく guard list を事前確保する。
単一 buffer の H2D/D2H copy helper では容量 1 で確保する。
`Buffer::acquire` は返す `BufferGuard` 内で `Arc<Buffer>` を保持するため、呼び出し前に
余分な `Arc` clone を行わない。

`Function` 内部の NodeId→pointer slot 対応も、実行時に `BTreeMap` を引く必要はない。
構築時にだけ map を作り、実行時はソート済み slice の binary search にすることで、
forward/backward の binding 解決で tree node を辿る overhead を減らす。
Model の forward/backward 経路ではさらに、`TensorBinding` 構築時に `FunctionInput`
slot まで解決しておき、batch 中は NodeId lookup 自体を行わない。

training loop の learning-rate / gradient-factor scalar は 1 要素 `TValue` を batch ごとに
新規作成する必要はない。loop 外で 2 つの scalar host buffer を確保し、batch ごとに値だけ
書き換えて H2D copy することで、小さな `Vec<f32>` allocation を避ける。

`PreparedBatchHost` の host-side input container も、batch ごとに小さな
`BTreeMap<String, TValue>` を作る必要はない。現行の shogi trainer では input 名は固定なので、
`Vec<(Cow<'static, str>, TValue)>` にして固定名は borrowed key として持つ。device 側の
`TensorMap<String, Buffer>` はそのまま維持し、copy 時だけ線形探索で該当 input を引く。
input 数は 5〜6 個なので tree 構築より安い。

### Phase 1: cuda-oxide runtime skeleton

tatara の `crates/gpu-runtime` 相当を BulletOu 側に最小移植する。

必要なもの:

- CUDA context / stream / module / device buffer wrapper
- PTX loader
- kernel launch macro
- CUDA error を `anyhow` / `thiserror` へ流す薄い変換
- build artifact 探索

この段階では kernel は smoke test 用でよい。

### Phase 1a: backend selector

実装順の最初として、CLI に `--backend` を追加する。

```text
--backend bullet      # 既定。現行の汎用 Bullet backend
--backend cuda-oxide  # NNUE/SFNN 専用高速 backend 用に予約
```

`cuda-oxide` は KPPT / KPP_KKPT を対象にしない。Phase 1a の時点では runtime /
kernel が未接続なので、`--backend cuda-oxide` は silent fallback せず明示エラーにする。
これにより、後続 Phase で実装を差し込む CLI 形状だけ先に固定し、既存の `bullet`
backend の挙動を変えない。

### Phase 2: dataloader adapter

最初の実験では、既存 BulletOu dataloader を完全流用するより、
cuda backend 用の固定 batch struct を作る。

候補:

```text
struct CudaNnueBatch {
    stm_indices: Vec<i32>,
    nstm_indices: Vec<i32>,
    offsets: Vec<i32>,
    buckets: Vec<u8>,
    scores: Vec<i16>,
    outcomes: Vec<i8>,
    weights: Vec<f32>,
}
```

初期対応形式は PSV だけでもよいが、BulletOu の実運用では HCPE / HCPE3 が重要なので、
次の順で対応する。

1. PSV
2. HCPE
3. HCPE3

HCPE / HCPE3 は既存 loader の decode 実装を reuse し、GPU backend に渡す直前で
固定 batch struct に詰め替える。

Phase 2a として、既存 `PreparedData` から変換できる `FastBatchHost` /
`FastBatchLayout` を `bulletou_lib::value::fast_batch` に追加する。これはまだ
GPU kernel を起動しないが、`stm` / `nstm` / `buckets` / `targets` / `weights` /
`hand_count` を name-keyed tensor map ではなく固定 field として保持する。後続の
cuda-oxide backend はこの layout を device buffer へ直接 upload する。

### Phase 3: 最小 NNUE forward / backward

最初の kernel 対象は、実装量が少ないほうから始める。

候補:

1. `NNUE_HALFKP_256x2_32_32`
2. `SFNN_halfka2_1024_7_64_k3k3`

ただし最終目的が SFNN 高速化なら、Phase 3 で `NNUE_HALFKP` に時間を使いすぎない。
`NNUE_HALFKP` は runtime / optimizer / checkpoint の smoke test 用と割り切る。

最小構成:

- sparse FT forward
- dense forward
- sigmoid loss
- backward
- Ranger / RAdam update
- `nn.bin` save
- `state.bin` resume

### Phase 4: tatara 型 fused kernel の導入

効果が大きい順:

1. loss + target transform + reduction
2. optimizer step
3. FT post activation
4. sparse FT backward
5. bucketed dense forward / backward
6. async loss ring
7. input upload ring

この段階から speedup が見えるはずである。

### Phase 5: opt-in 高速化

数値が変わる可能性があるものは opt-in にする。

| option | 内容 | 既定 |
|---|---|---|
| `--cuda-oxide-ft-fp16` | FT activation / workspace を FP16 化 | off |
| `--cuda-oxide-opt-fp16-state` | optimizer state を FP16 化 | off |
| `--cuda-oxide-tf32` | dense GEMM で TF32 使用 | off |
| `--cuda-oxide-bucket-sort` | bucket ごとに batch を並べ替える | off |

まず fp32 baseline を一致させ、そのあと opt-in の強さを評価する。

## checkpoint 互換性

cuda-oxide backend が別 binary になっても、出力は既存 BulletOu と互換にする。

最低条件:

- `nn.bin` は `docs/spec/02-nnue-binary.md` と一致
- checkpoint dir numbering は `docs/spec/04-checkpoint-layout.md` と一致
- `learn.log` / summary log の列は既存と互換
- resume 可能な `state.bin` を保存

ただし optimizer state の内部形式は backend 固有でもよい。
その場合は `state.bin` に backend marker を持たせ、異なる backend からの resume は
明示的に拒否する。

```text
state backend = bullet
state backend = cuda-oxide
```

異なる backend 間で model weight だけ引き継ぎたい場合は、`nn.bin` から初期化する
別経路を用意する。

## build / 実行環境

cuda-oxide backend は既存 BulletOu より環境要求が厳しい。

必須:

- NVIDIA GPU
- Linux または WSL2
- CUDA Toolkit 12.x
- LLVM 21 以上。LLVM 22 推奨
- cuda-oxide と整合する nightly Rust
- `cargo-oxide`

現行 BulletOu のように、Windows native でそのまま動く保証はしない。
macOS GPU / ROCm も対象外。

この制約のため、既存 backend は残す必要がある。

## 検証方針

### kernel 単位

各 fused kernel に CPU reference を用意し、GPU と比較する。

例:

| kernel | 比較内容 |
|---|---|
| loss | per-position loss と reduction |
| optimizer | weight / m / v / slow weight |
| FT forward | accumulator / activation |
| FT backward | feature weight gradient |

許容誤差:

- fp32 path: `1e-5` 程度
- fp16 / TF32 path: 別基準を設ける

### training 単位

小さい teacher で、現行 BulletOu と cuda-oxide backend を比較する。

比較項目:

- 同じ seed
- 同じ initial weight
- 同じ batch
- 1 batch 後の weight 差
- 1 superbatch 後の loss / accuracy
- positions/sec

optimizer や loss の実装が完全一致しない段階では、速度比較をしてはいけない。

## リスク

### cuda-oxide 自体が alpha

tatara も cuda-oxide の rev と nightly Rust を pin している。
BulletOu に入れる場合も、Cargo.lock と toolchain を固定しないと壊れやすい。

### runtime graph の自由度を失う

cuda-oxide backend は build-time kernel なので、任意の新 architecture を
即座に試す用途には向かない。
新 architecture ごとに kernel または host layout の追加が必要になる。

### データ形式対応が重い

tatara は PSV 中心で設計されているが、BulletOu では HCPE / HCPE3 が重要。
ここを雑にすると、実運用で使えない高速 backend になる。

### 数値差分が入りやすい

FP16 / TF32 / fused reduction / async loss readback は、全て数値差分の原因になる。
まず fp32 で既存 backend と一致させ、opt-in 高速化は別実験にする。

## 推奨する最初の実装

最初の実装は以下に限定する。

1. `bins/bulletou_cuda_train` を追加
2. `crates/cuda_oxide_runtime` を追加
3. smoke kernel を 1 つだけ build / launch
4. `NNUE_HALFKP_256x2_32_32` の 1 batch forward だけ実装
5. CPU / 現行 BulletOu との forward 出力一致テストを作る

この段階ではまだ高速化を狙わない。
cuda-oxide build chain と device buffer / kernel launch / batch layout の土台を
安定させることが目的である。

その後、SFNN 用に以下を順に入れる。

1. sparse FT forward
2. dense forward
3. loss kernel
4. backward
5. optimizer
6. fused kernel 化
7. async ring

## まとめ

BulletOu を tatara 方式で高速化するなら、cuda-oxide は「置換先」ではなく
「NNUE 専用高速 backend のための実装手段」と見るべきである。

BulletOu の強みは、HCPE / HCPE3 対応、やねうら王向け出力、複数 architecture の
実験容易性にある。これを捨てて tatara を丸ごと移植するのではなく、既存 backend を
維持しながら、頻繁に使う shogi NNUE / SFNN だけを cuda-oxide backend へ
段階的に逃がすのが最も安全である。
