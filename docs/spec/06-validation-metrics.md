# 06. Validation Metrics 仕様

`bulletou` の `test_value_accuracy` / `test_value_loss` 列と、外部ツール (YaneuraOu `test eval_accuracy`、dlshogi `train.py` の検証) との **cross-tool 数値比較契約**。

3 経路で同じ数値が出ることを担保するための仕様定義。

## 対象とする 3 経路

| 経路 | 計算タイミング | 対象モデル |
|---|---|---|
| **BulletOu `test_value_accuracy`** | 学習中、各 validation event (= `--validation-rate` ごと。未指定なら `--save-rate`) および save event | 量子化前 f32 model output |
| **YaneuraOu `test eval_accuracy <psv>`** | 学習完了後、`nn.bin` を load した engine 上 | 量子化後 i16 NNUE + Eval::evaluate() |
| **dlshogi `train.py` の検証** | 学習中、本家 dlshogi トレーナーの validation phase | dlshogi の Python model |

Current cuda-cpp training can decouple validation from checkpoint saves:
`--validation-rate N` runs `test_value_accuracy` / `test_value_loss` every
N superbatches, while `--save-rate` controls checkpoint writes. If
`--validation-rate` is omitted, it defaults to `--save-rate` for backward
compatibility.

3 経路で **同じ局面集**を渡して **同じ accuracy 数値**が出るのが理想。量子化前後の差 (BulletOu vs YaneuraOu) は本物のモデル差、ツール間の数値差はバグ。

## `test_value_accuracy` の定義

**Draw-excluded sign agreement** (= W vs L の符号一致率)。

```text
对象局面 i ごとに:
  if (game_result == 0)                  → skip (draw は分母分子ともに除外)
  if (|teacher_score| >= score_drop_abs) → skip (mate stamp 除外)
  else:
    pred  = (model_output >= 0)            ← STM 視点での勝ち予測
    truth = (game_result   >  0)           ← STM 実際に勝った
    if (pred == truth): sign_matches++
    compared++

accuracy = sign_matches / compared
```

Diagnostic counters (`pred>=0`, `pred<0`, `zero`) are reported over the
same decisive subset as `compared`. They do not change the metric; they
exist to expose short-run cases where the model has not yet learned a
meaningful sign split and accuracy is effectively the held-out
Win/Loss class balance.

`game_result == 0` (引き分け) は **必ず両側から除外**する。理由:
- dlshogi 作者の検証局面集に引き分けが含まれていない (= 本家準拠)
- 引き分けを `truth=1` 側にバケットすると「model_output ≈ 0 を出すモデル」が draw + win に対し機械的に正解扱いされる構造的バイアスが入る (= `--scale` を小さくすると単調に accuracy が上がる現象の主因)
- W vs L の純粋な符号一致率の方が、scale 比較で artifact が出にくい

### scale 比較における不変性

`test_value_accuracy` は scale-invariant (= `--scale` を変えても unit が変わらない)。

`pred` は `model_output >= 0` (= 0 を閾値にした符号判定) のみで決まり、scale の値を使わない。`truth` は teacher の対局結果から決まり、scale を使わない。よって異なる `--scale` で学習した model 同士の accuracy を直接比較できる。

ただし scale を極端に下げると model 自体が「符号予測タスクに退化」してマグニチュード情報が消える (= sigmoid が saturate して大局のスコア差が無くなる)。accuracy は良くなるが対局強度は別途検証が必要。

## `test_value_loss` の定義

検証局面集に対する **average squared loss** (= bullet の training loss と同じ形式)。

```text
对象局面 i ごとに:
  if (|teacher_score| >= score_drop_abs): skip (mate stamp 除外)
  else:
    blend       = 1 - lambda
    result_norm = result == +1 ? 1.0 : result == -1 ? 0.0 : 0.5
    score_norm  = sigmoid(teacher_score / scale)
    target      = blend * result_norm + (1 - blend) * score_norm
    pred        = sigmoid(model_output)
    loss_i      = (pred - target)^2

test_value_loss = mean(loss_i over compared positions)
```

### accuracy との違い: draw を含む

`test_value_loss` は **draw 局面も含む** (= `result_norm = 0.5` で素直に評価)。これは bullet の training loss と subset を揃えるため (= training subset と同じものを test に流して比較可能にする)。

### scale 比較における不変性: なし

`test_value_loss` は **scale-dependent**。target が `sigmoid(score/scale)` 経由で scale に依存するため、異なる scale で学習した model の loss は単位が違って直接比較できない。

scale 比較には accuracy を使い、loss は同一 scale 内の sb 進行・lambda 比較などに使う。

## `score_drop_abs` フィルタ

`|teacher_score| >= --score-drop-abs` の局面は accuracy / loss 両方から除外する。これは mate stamp (= ±32000 等の決定的局面マーク) を training 評価から外すためのもの。

デフォルト未指定 (= 0) のときは全局面 evaluate される。

## YaneuraOu `test eval_accuracy` との対応

YaneuraOu 側の実装は `source/testcmd/normal_test_cmd.cpp::eval_accuracy` (= `YANEURAOU_ENGINE` 配下のみ build)。

```cpp
if (rec.game_result == 0) { drawn++; continue; }   // draw 除外、BulletOu と同じ
Value v = engine.evaluate();                       // 量子化後 NNUE forward
bool pred  = v >= VALUE_ZERO;
bool truth = rec.game_result > 0;
if (pred == truth) sign_match++;
compared++;
```

PSV (40 byte) 形式の `--test-teacher` を引数で受ける (= HCPE ではなく psv)。`game_result` は STM 視点 (= s8 で -1/0/+1)、これは BulletOu の HCPE デコード後の `game_result` と同じ意味。

### 既知の差異要因

| 要因 | BulletOu | YaneuraOu | 差異の方向 |
|---|---|---|---|
| 量子化 | f32 (= 量子化前) | i16 (= 量子化後) | YaneuraOu accuracy がわずかに低い (= 通常 0.1〜1%) |
| 局面フィルタ | `score_drop_abs` | なし (= 全局面 evaluate) | 同じ集合を渡す前提なら影響なし。ユーザーが事前フィルタ要 |
| accumulator 更新 | バッチ forward 1 回 | `set_from_packed_sfen` のたび full refresh | 数値的に同一 (= incremental は決定論的に折りたたまれる) |
| Eval::evaluate の post-processing | なし | post-evaluate (= 評価値の clamp 等) | 通常無視できる |

## dlshogi `train.py` との対応

dlshogi の本家 train.py 内の validation 関数 (`binary_accuracy`) は元々:

```python
pred  = (y >= 0).float()
truth = (t >= 0.5).float()   # t は [0, 1] の sigmoid 値、>= 0.5 で win 側
match = (pred == truth).float().mean()
```

draw 局面 (t == 0.5) は `truth = True` 側にバケット。**これが旧 dlshogi convention** (= BulletOu でも v0.x ではこれを採用していた)。

ただし dlshogi 作者が **実際に検証用に使っている局面集には引き分けが含まれていない**ため、上記式の `truth=True bucket` 効果は機械的にはゼロ。

→ 仕様としては「draw 局面が検証集にあった場合の挙動」が dlshogi と BulletOu で異なるが、**「draw を含まない検証集を使う」運用に揃えれば 3 経路の数値が一致する**。

### 推奨: 検証集の draw 除外

検証 hcpe / psv ファイルから事前に draw を除いておく:

```bash
# YaneuraOu-ScriptCollection/teacher/filter_drawn_games.py
python teacher/filter_drawn_games.py test.hcpe test.no-drawn.hcpe
python teacher/filter_drawn_games.py test.psv  test.no-drawn.psv
```

これで 3 経路すべてが draw 0 局面で動くため、accuracy 数値が直接 cross-validate 可能になる。

詳細は `YaneuraOu-ScriptCollection/teacher/README.md` の「引き分け局面を取り除く」セクション。

## 関連

- BulletOu 実装: `crates/bulletou_lib/src/validate.rs::compute_sign_accuracy`
- YaneuraOu 実装: `YaneuraOu/source/testcmd/normal_test_cmd.cpp::eval_accuracy`
- 事前フィルタ: `YaneuraOu-ScriptCollection/teacher/filter_drawn_games.py`
- log 列の所在: [04-checkpoint-layout.md](04-checkpoint-layout.md) の `learn.log` フォーマット節
