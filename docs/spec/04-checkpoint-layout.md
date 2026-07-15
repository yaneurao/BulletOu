# 04. Checkpoint Layout 仕様

`bulletou` がトレーニング中・終了時に `<output>/` 配下に書く成果物の構造と、resume プロトコル。

## 全体レイアウト

```
<output>/
├── summary-learn.log                  ← トップレベル累積ログ (= 全 run の sb 境界行だけを連結)
├── 0001/                              ← 1 個目の save
│   ├── (eval-type specific files)
│   ├── state.bin                      ← resume 用 (重み + Adam moments)
│   ├── dataloader_pos.txt             ← HCPE/HCPE3/pack の byte offset resume 用 (= 同教師継続再開)
│   └── learn.log                      ← この save 時点までの per-batch loss snapshot
├── 0002/
├── ...
└── 000N/                              ← 最新の save (resume 元、engine が指すべき dir)
    ├── (eval-type specific files)
    ├── state.bin
    ├── dataloader_pos.txt
    └── learn.log
```

番号は **save が走るごとに 1 ずつインクリメント**。デフォルトでは `--save-rate=1` で 1 superbatch ごとに save、`--save-rate=10` なら 10 superbatch ごと。

resume 時は **既存番号の続きから連番**。例えば前回 `0005/` まで存在する dir に対して再実行すると、新規 save は `0006/`, `0007/`, ... となる。

トップレベル log の名前が `summary-learn.log` なのは per-save 配下の `learn.log` (= per-batch 行を含む詳細版) と区別するため。`summary-learn.log` には各 sb の最終行 (= sb 境界の代表行) のみ抽出されて連結される。

`learn.log` (各 save 配下) は **その save 時点までの loss 履歴 snapshot**。同一 run 内では cumulative。run を跨ぐ (resume する) と loss snapshot はその run 単位で start し直す。

`dataloader_pos.txt` は HCPE / HCPE3 / pack の byte offset 再開用。各 save の callback で「consumer がここまで処理した」位置を `<byte_offset>,<plies_within_unit>` 形式 1 行で書く (固定長 HCPE / PSV では plies は常に 0)。auto-resume は最新 dir のこのファイルを読み、bullet の dataloader を該当 offset から再開させる (= 同教師継続再開時)。教師が変わった場合は無視して新教師の先頭から読む。`--superbatches N` 指定時の完走後継続学習では、これを使って教師位置も継続する。

## eval-type 別の per-save ファイル

### KPPT / KPP_KKPT
```
0NNN/
├── KK_synthesized.bin                 ← int32 × 2、6,561 entries (= 81 × 81)
├── KKP_synthesized.bin                ← int32 × 2、10,156,428 entries (= 81 × 81 × 1548)
├── KPP_synthesized.bin                ← int16 × 2 (KPPT) or int16 (KPP_KKPT)、194,100,624 entries
├── state.bin
└── learn.log
```

3 ファイル組すべてを engine が要求する。詳細は `bulletou_lib::value::yaneuraou_kppt` モジュール doc-comment。

### NNUE_HALFKP / NNUE_KP
```
0NNN/
├── nn.bin                             ← YaneuraOu / Stockfish 互換 NNUE バイナリ (詳細 [02-nnue-binary.md])
├── state.bin
└── learn.log
```

## `state.bin`

resume 用のバイナリ。bullet の既存 record format を流用。

### バイナリ format

各 record は以下の形:
```
<ID string>\n              ← UTF-8、改行で終端
<N: usize little-endian>   ← 8 byte (64bit)
<f32 × N little-endian>    ← 重みの実体
```

record を必要なだけ連結したものが `state.bin`。

### record ID 命名

各 record の `<ID string>` は以下のスラッシュ区切り構造:

```
<component>/<section>/<weight_id>
```

| 部位 | 取りうる値 |
|---|---|
| `<component>` | KPPT: `kk` / `kkp` / `kpp`、NNUE: `nnue` |
| `<section>` | `weights` / `momentum` / `velocity` (Adam moments) |
| `<weight_id>` | bullet のモデル内重み ID (例: `kkw` / `kkpb` / `l0w` / `l0b` 等) |

例:
- KPPT save (3 component): `kk/weights/kkw`, `kk/momentum/kkw`, `kk/velocity/kkw`, `kkp/weights/kkpw`, ..., `kpp/velocity/kppw`
- NNUE_HALFKP save (1 component): `nnue/weights/l0w`, `nnue/weights/l0b`, `nnue/weights/l1w`, ..., `nnue/velocity/outb`

### 生成・展開 API

`bulletou_lib::value::yaneuraou_kppt` モジュール内 (歴史的経緯で KPPT モジュール下にあるが、API 自体は generic):

| 関数 | 用途 |
|---|---|
| `bundle_component_state(out, component, optimiser_state_dir)` | `optimiser_state/{weights,momentum,velocity}.bin` を読んで `<component>/<section>/<id>` namespace で state.bin に追記 |
| `parse_model_weights_bin(bytes)` | state.bin を parse して `BTreeMap<ID string, Vec<f32>>` に展開 |
| `unbundle_component_state(records, component, optimiser_state_dir)` | parse 結果から特定 component を取り出して `<dir>/optimiser_state/{weights,momentum,velocity}.bin` に書き戻す |

## resume プロトコル

`bulletou` 起動時、`--output` に既に numbered dir + `state.bin` が存在する場合に自動 resume する。**ユーザー側に override flag は無い** (旧 `--start-superbatch` は v1.0 で削除済み)。

### 検出

`find_latest_state_bin(output_dir)`:
1. `output_dir` 配下の subdir をリストアップ
2. 名前が `usize::parse` できるものに絞る (= 番号付き dir)
3. その中で最も大きい番号 + `state.bin` が存在する dir を返す

### 復元

1. `state.bin` 全体を読んで `parse_model_weights_bin` で `BTreeMap` 化
2. スクラッチ dir (`<output>/.bulletou_resume/`) を確保
3. 各 component に対し `unbundle_component_state` でスクラッチ dir に展開
4. 各 component の trainer 構築直後に `trainer.load_from_checkpoint(<scratch>/<component>)` を呼ぶ
5. 学習終了後、`<output>/.bulletou_resume/` を削除

### 連番継続

新規 save は **既存番号の続き** から書く。例: `<output>/0005/` までが既存なら、新 save の最初は `<output>/0006/`。

実装: `count_existing_numbered_dirs(output_dir)` で既存数 N をカウントし、新 save の番号を `N + 1, N + 2, ...` とする。

### 起動時の自動分岐 (3 ケース)

`bulletou` 起動時、auto-resume は前回 run の状態を読んで以下のいずれかに分岐する。bullet 内部の `start_superbatch` と HCPE dataloader の byte offset がケースごとに変わる。LR scheduler は positions-based なので、いずれも `cb_prior_position` (= `summary-learn.log` の最大 positions) を carry-over するだけで連続性を保つ。

| ケース | 検出条件 | bullet `start_sb` | dataloader offset | log 表示 |
|---|---|---|---|---|
| **mid-epoch resume** | 教師同じ & 前回 last_sb < `--superbatches` | `last_sb + 1` | `dataloader_pos.txt` から | sb 列 = `last_sb+1..N` で再開、次 epoch 以降 sb=1..N |
| **clean continuation** | 教師同じ & 前回 last_sb >= `--superbatches` (= 完走後の追加学習) | `1` | `dataloader_pos.txt` から | sb 列 = 1..N (= 新 epoch の自然なカウント) |
| **teacher-changed** | 教師パス変更 (= `summary-learn.log` 最終行と現 `--teacher` 不一致) | `1` | `0` | sb 列 = 1..N |
| **fresh first run** | numbered dir 無し | `1` | `0` | sb 列 = 1..N |

`--superbatches N` を明示した run では、epoch は教師1周ではなく LR/validation cycle である。そのため clean continuation でも教師位置は `dataloader_pos.txt` から継続し、epoch 境界で教師先頭へは戻さない。教師EOFに到達した場合だけ、同じ epoch のまま教師先頭へ cyclic に戻る。

`--superbatches` 未指定の非 plateau run だけは、従来通り「教師EOF = epoch終了」として扱う。このモードで次 epoch を開始する場合は、教師先頭から読む。

各 epoch の chunk loop 開始時、bullet `start_sb` は **epoch 1 のみ** 上記の値、**epoch 2 以降は常に 1** にリセットされる。これは表示上の sb と LR/validation cycle のリセットであり、`--superbatches N` 指定時の教師位置リセットではない。

### epoch カウンタの cross-run 連続化

bullet 内部の `for epoch in 1..=max_epochs` は invocation ごとに 1 にリセットされる。継続学習で表示 epoch 列が `1, 2, 3, ...` に戻ると視認性が悪いため、`LogContext.epoch_offset` で表示時のみシフトする:

| ケース | `epoch_offset` |
|---|---|
| fresh first run | `0` |
| clean continuation / teacher-changed | `max_epoch_in_summary_log` (= 新 local epoch=1 が表示 max+1) |
| mid-epoch resume | `max_epoch_in_summary_log - 1` (= 中断 epoch の続きは同じ epoch 番号で表示、次 epoch から +1) |

これは **表示のみ**の補正で、`positions` 列や lr 計算には影響しない。

### sb カウンタの per-epoch 性質

sb 列は intrinsic に **per-epoch カウンタ** (= 各 epoch で 1..`--superbatches`)。cross-run の累積カウンタではない。継続学習でも各 epoch は sb=1 から表示される。

(歴史的経緯: `--max-epochs` 導入前の v0.x では sb は invocation を跨ぐ累積カウンタで、cross-run の `sb_offset` 補正が必要だった。v1.0 で sb は per-epoch 化され、`sb_offset` 補正は削除された。)

## `learn.log` / `summary-learn.log` フォーマット

2 種類の CSV ログがあり、列数が違う。両方ともヘッダ行つき、区切り文字はカンマ、行ごとの末尾改行あり。pandas / Excel でそのまま load 可能。

### per-save `<output>/000N/learn.log` (= 12 列、per-batch snapshot)

```
eval,epoch,superbatch,curr_batch,test_value_accuracy,test_value_loss,train_value_loss,lr_start,lr_end,lambda,positions,teacher
NNUE_HALFKP-NNUE_halfkp_256x2_32_32,1,1,32,-,-,0.234,0.001000,0.000999,1.000000,524288,teachers/
NNUE_HALFKP-NNUE_halfkp_256x2_32_32,1,1,64,-,-,0.231,0.000999,0.000998,1.000000,1048576,teachers/
...
NNUE_HALFKP-NNUE_halfkp_256x2_32_32,1,1,6104,0.576647,0.181778,0.071046,0.001000,0.000934,1.000000,100007936,teachers/
```

bullet は 32 batch ごとに 1 行 loss を記録するので、1 sb 内に約 191 行 (= `--batches-per-superbatch` ÷ 32)。`test_value_accuracy` / `test_value_loss` は **sb 境界の最終行のみ実値**、その他の per-batch 行は `-` (= save event でのみ validation が走るため)。

### top-level `<output>/summary-learn.log` (= 11 列、sb 境界のみ抽出)

```
eval,epoch,superbatch,test_value_accuracy,test_value_loss,train_value_loss,lr_start,lr_end,lambda,positions,teacher
NNUE_HALFKP-NNUE_halfkp_256x2_32_32,1,1,0.576647,0.181778,0.071046,0.001000,0.000934,1.000000,100007936,teachers/
NNUE_HALFKP-NNUE_halfkp_256x2_32_32,1,2,0.583300,0.174947,0.077046,0.000934,0.000753,1.000000,200015872,teachers/
```

per-save 版から `curr_batch` 列を除いたもの (= 各 sb の最終行 = sb 境界の代表行のみ)。複数 run / 複数 epoch を跨いで連結される。新規 save callback で 1 行ずつ追記される。

### 列の意味

| 列 | 意味 |
|---|---|
| `eval` | 出力ディレクトリ名と同じ `<eval-type>[-<arch>]` 形式 + マルチ component (KPPT 系) ではさらに `/<component>` を付加。NNUE 系 (シングル component、`--arch` を使う) は `NNUE_HALFKP-NNUE_halfkp_256x2_32_32` のように eval-type と arch を `-` で結合。KPPT 系 (`--arch` を使わない、3 component 連続学習) は `KPPT/kk` / `KPPT/kkp` / `KPPT/kpp` (または `KPP_KKPT/kk` 等) を行ごとに記録 |
| `epoch` | この save の epoch 番号。継続学習を跨いで連続化される (= `LogContext.epoch_offset` 補正後の値)。1 始まり |
| `superbatch` | 現在 epoch 内の 1 始まり superbatch カウンタ。`--batches-per-superbatch` (デフォルト 6104) batch ごとに +1 される。**per-epoch カウンタ**で、cross-run 累積ではない。新 epoch ごとに 1 にリセット |
| `curr_batch` | (per-save 版のみ) 現在 superbatch 内の 1 始まり batch カウンタ。bullet は 32 batch ごとに 1 行記録するので 32, 64, 96, ... の値を取る |
| `test_value_accuracy` | `--test-teacher` 検証局面に対する **draw-excluded sign agreement** (詳細は [06-validation-metrics.md])。sb 境界行のみ実値、それ以外は `-`。`--test-teacher` 未指定なら全行 `-` |
| `test_value_loss` | `--test-teacher` 検証局面に対する average loss (sigmoid + WDL の合成 target に対する MSE。draw は loss 側には含まれる)。sb 境界行のみ実値、それ以外は `-` |
| `train_value_loss` | bullet が最後の 32 batch で観測した training loss (移動平均ではなく 32 batch ウィンドウの即値) |
| `lr_start` | その行が表す区間の開始時点の学習率。summary 行ではその superbatch の開始 LR |
| `lr_end` | その行が表す区間の最後の batch で使った学習率。summary 行ではその superbatch の終端側 LR |
| `lambda` | その時点の `--lambda` (1 run 内では定数)。**小数点以下 6 桁固定** で出力 (`1.000000`、`0.500000` など) |
| `positions` | この component で消費した累計教師局面数。**resume / epoch 跨ぎで累積される** (run 開始時に既存 `summary-learn.log` の最大値を読み取って続きから書く)。full save の sb 境界行では、bullet の raw log が 32 batch 刻みで途中までしか出ていなくても、正確な `superbatch × batches_per_superbatch × batch_size` を書く。常に単調増加 |
| `teacher` | CLI の `--teacher` 値そのまま (RFC 4180 escape: 値内にカンマ/ダブルクォート/改行があるときは `"..."` で囲む) |

### 累積ロジック

- 1 run 内で各 component が消費する局面数: `positions = cb_prior_position + (local_superbatch − 1) × batches_per_superbatch × batch_size + curr_batch × batch_size`
- `cb_prior_position` は run 開始時 + 各 epoch 境界で `read_prior_positions()` から再ロードされる (= component 別の最大 `positions`)
- 各 save dir の `0NNN/learn.log` は「**その save 時点までの累積**」(bullet が log.txt を逐次更新するため)。最新番号 dir の `learn.log` を読めばその run の全貌が分かる
- トップレベル `<output>/summary-learn.log` は各 save の callback で sb 境界行を 1 行ずつ追記する (新規作成時のみ 1 度ヘッダを書く)。ファイル内のヘッダは常に 1 行のみ
- resume を跨いでも `epoch` / `superbatch` / `positions` 全部が連続表示される (= `epoch_offset` + 自然な per-epoch sb + `cb_prior_position` のおかげ)。Pandas で `positions` を sort key にすれば確実に時系列順

## ファイル名規約 (一時)

学習中、bullet 自体は `<output>/<net_id>-<superbatch>/` という命名の dir に書き出す (`net_id` は `--net-id` または `--eval-type` 由来のデフォルト)。

各 save の callback でこれを以下に変換する:

| eval-type | 変換 |
|---|---|
| KPPT / KPP_KKPT | `KK_synthesized.bin` 等を produce、`state.bin` を bundle |
| NNUE 系 | `quantised.bin` を `nn.bin` に rename、`state.bin` を bundle |

run 終了後、これらの per-component subdir を `<output>/0NNN/` 形式の番号付き dir に rename / assemble する (`assemble_numbered_dirs` for KPPT, `finalize_nnue_dirs` for NNUE)。
