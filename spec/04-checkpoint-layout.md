# 04. Checkpoint Layout 仕様

`bulletou` がトレーニング中・終了時に `<output>/` 配下に書く成果物の構造と、resume プロトコル。

## 全体レイアウト

```
<output>/
├── learn.log                          ← トップレベル累積ログ (全 run / resume を連結)
├── 0001/                              ← 1 個目の save
│   ├── (eval-type specific files)
│   ├── state.bin                      ← resume 用 (重み + Adam moments)
│   └── learn.log                      ← この save 時点の loss snapshot
├── 0002/
├── ...
└── 000N/                              ← 最新の save (resume 元、engine が指すべき dir)
    ├── (eval-type specific files)
    ├── state.bin
    └── learn.log
```

番号は **save が走るごとに 1 ずつインクリメント**。デフォルトでは `--save-rate=1` で 1 superbatch ごとに save、`--save-rate=10` なら 10 superbatch ごと。

resume 時は **既存番号の続きから連番**。例えば前回 `0005/` まで存在する dir に対して再実行すると、新規 save は `0006/`, `0007/`, ... となる。

`learn.log` (各 save 配下) は **その save 時点までの loss 履歴 snapshot**。同一 run 内では cumulative。run を跨ぐ (resume する) と loss snapshot はその run 単位で start し直す。

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

`bulletou` 起動時、`--output` に既に numbered dir + `state.bin` が存在する場合に自動 resume する。

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

## `learn.log` フォーマット

各 save dir の `learn.log` も、トップレベル `<output>/learn.log` も、**同じ 9 列 CSV (ヘッダ行つき)**。pandas / Excel でそのまま load 可能。区切りやセクションヘッダは入らない。

```
eval,epoch,superbatch,curr_batch,value_loss,lr,lambda,positions,teacher
NNUE_HALFKP-256x2-32-32,1,1,32,0.234,0.001,1.000,524288,teachers/
NNUE_HALFKP-256x2-32-32,1,1,64,0.231,0.001,1.000,1048576,teachers/
...
KPPT/kk,1,1,32,0.234,0.001,0.500,524288,teachers/
KPPT/kkp,1,1,32,0.156,0.001,0.500,524288,teachers/
KPPT/kpp,1,1,32,0.245,0.001,0.500,524288,teachers/
...
```

### 列の意味

| 列 | 意味 |
|---|---|
| `eval` | 出力ディレクトリ名と同じ `<eval-type>[-<arch>]` 形式 + マルチ component (KPPT 系) ではさらに `/<component>` を付加。NNUE 系 (シングル component、`--arch` を使う) は `NNUE_HALFKP-256x2-32-32` のように eval-type と arch を `-` で結合。KPPT 系 (`--arch` を使わない、3 component 連続学習) は `KPPT/kk` / `KPPT/kkp` / `KPPT/kpp` (または `KPP_KKPT/kk` 等) を行ごとに記録 |
| `epoch` | この run 内の 1 始まり epoch カウンタ (`--max-epochs`) |
| `superbatch` | 現在 epoch 内の 1 始まり superbatch カウンタ。`--batches-per-superbatch` (デフォルト 6104) batch ごとに +1 される |
| `curr_batch` | 現在 superbatch 内の 1 始まり batch カウンタ。bullet は 32 batch ごとに 1 行記録するので 32, 64, 96, ... の値を取る |
| `value_loss` | bullet が 32 batch ごとに計算する loss 値 |
| `lr` | その superbatch における学習率 (StepLR 由来) |
| `lambda` | その時点の `--lambda` (1 run 内では定数)。**小数点以下 3 桁固定** で出力 (`1.000`、`0.500` など) |
| `positions` | この component で消費した累計教師局面数。**resume 跨ぎで累積される** (run 開始時に既存トップレベル `learn.log` の最大値を読み取って続きから書く)。multi-epoch (`--max-epochs > 1`) 内では epoch 境界で reset する (v1 制限) |
| `teacher` | CLI の `--teacher` 値そのまま (RFC 4180 escape: 値内にカンマ/ダブルクォート/改行があるときは `"..."` で囲む) |

### 行の頻度

bullet は 32 batch ごとに 1 行 loss を記録する。デフォルト `--batches-per-superbatch ≒ 6104` だと、1 superbatch あたり約 191 行。

### 累積ロジック

- 1 run 内で各 component が消費する局面数は単調増加: `positions = (superbatch − 1) × batches_per_superbatch × batch_size + curr_batch × batch_size + prior_offset`
- `prior_offset` は run 開始時に `read_prior_positions()` で既存トップレベル `learn.log` から取得する (component 別の最大 `positions`)
- 各 save dir の `0NNN/learn.log` は「**その save 時点までの累積**」(bullet が log.txt を逐次更新するため)。最新番号 dir の `learn.log` を読めばその run の全貌が分かる
- トップレベル `<output>/learn.log` は run 終了時に最新 dir の内容を **ヘッダ行を除いて** 追記 (新規作成時のみ 1 度ヘッダを書く)。結果として 1 ファイル内のヘッダは常に 1 行のみ
- resume を跨ぐと `superbatch` カウンタは 1 から再開するが、`epoch` も 1 から始まる。`positions` だけが累積を保つので、Pandas で sort key にすれば順序を保てる

## ファイル名規約 (一時)

学習中、bullet 自体は `<output>/<net_id>-<superbatch>/` という命名の dir に書き出す (`net_id` は `--net-id` または `--eval-type` 由来のデフォルト)。

各 save の callback でこれを以下に変換する:

| eval-type | 変換 |
|---|---|
| KPPT / KPP_KKPT | `KK_synthesized.bin` 等を produce、`state.bin` を bundle |
| NNUE 系 | `quantised.bin` を `nn.bin` に rename、`state.bin` を bundle |

run 終了後、これらの per-component subdir を `<output>/0NNN/` 形式の番号付き dir に rename / assemble する (`assemble_numbered_dirs` for KPPT, `finalize_nnue_dirs` for NNUE)。
