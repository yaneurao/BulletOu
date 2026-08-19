# SFNN factorizer

<a href="../../en/advanced/sfnn-factorizer.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

このページでは、SFNN の LayerStack で使う `--sfnn-factorizer` を説明します。

まず学習を1回動かしたいだけなら、このページを読む必要はありません。`hand1024`、`k29k29`、`progress8` のように bucket 数が多い architecture を比較したいときに読んでください。

## 1. factorizer は何をするものか

LayerStack は、局面ごとに使う後段 network を切り替える仕組みです。たとえば次の architecture は、手駒、玉位置、進行度を組み合わせます。

```text
SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4
```

この場合、stack 数は次のようになります。

```text
hand1024 * k3k3 * progress4 = 1024 * 9 * 4 = 36,864 stacks
```

各 stack が完全に独立した重みを持つと、表現力は上がります。その一方で、1つの stack に届く教師局面は少なくなります。rare な bucket は学習が薄くなり、検証 loss や量子化後 loss が不安定になりやすいです。

factorizer は、この問題を緩和するために、stack 間で共通成分を持たせる仕組みです。ざっくり言うと、各 stack の重みを「個別成分 + 共通成分」の足し算で表します。

```text
W_effective = W_base + W_shared + W_axis + W_pair
```

`W_effective` が実際に forward で使われる重みです。ここでの `W` は、SFNN の L1/L2/L3 の stack ごとの weight や bias の1要素だと思ってください。

## 2. bucket 軸と stack 番号

BulletOu では、LayerStack の stack 番号を次の順序で合成します。

```text
stack = ((hand_bucket * king_bucket_count) + king_bucket) * progress_bucket_count
      + progress_bucket
```

architecture にその軸がない場合、その軸の bucket 数は1として扱われます。

| 軸 | 例 | 意味 |
|---|---|---|
| hand | `hand4`, `hand16`, `hand64`, `hand64z`, `hand256`, `hand1024` | 手番側と非手番側の手駒状態 |
| king | `k3k3`, `k9k9`, `k9k9z`, `k13k13z`, `k21k21`, `k29k29` | 玉位置 bucket |
| progress | `progress2`, `progress4`, `progress8`, `progress16`, `progress32` | 進行度 bucket |

factorizer は、この hand / king / progress の軸を使って、どの stack 同士で成分を共有するかを決めます。

## 3. `shared`

`shared` は、全 stack で1つの共通成分を足します。

```text
W_effective[hand, king, progress]
  = W_base[hand, king, progress]
  + alpha_shared * W_shared
```

全 stack に同じ成分が足されるので、最も粗い共有です。BulletOu のデフォルトは `--sfnn-factorizer shared` です。

## 4. `axis`

`axis` は、bucket の単独軸ごとに成分を足します。

```text
W_effective[hand, king, progress]
  = W_base[hand, king, progress]
  + alpha_shared   * W_shared
  + alpha_hand     * W_hand_axis[hand]
  + alpha_king     * W_king_axis[king]
  + alpha_progress * W_progress_axis[progress]
```

実際には、architecture に存在する軸だけが使われます。たとえば `k3k3` だけなら hand/progress axis はありません。`hand1024_k3k3_progress4` なら hand / king / progress の3軸すべてがあります。

### hand axis の分解

hand bucket は、手番側と非手番側の手駒 bucket を掛け合わせたものです。

| 指定 | 片側 bucket 数 `D` | 合計 hand bucket 数 |
|---|---:|---:|
| `hand4` | 2 | 4 |
| `hand16` | 4 | 16 |
| `hand64` | 8 | 64 |
| `hand64z` | 8 | 64 |
| `hand256` | 16 | 256 |
| `hand1024` | 32 | 1024 |

合成式は次の形です。

```text
hand_bucket = stm_hand_bucket * D + non_stm_hand_bucket
```

`hand=axis` は、この `hand_bucket` をそのまま1024個の独立成分として持つのではなく、手番側と非手番側の2方向へ分解して持ちます。

```text
W_hand_axis[hand_bucket]
  = W_hand_stm_axis[stm_hand_bucket]
  + W_hand_non_stm_axis[non_stm_hand_bucket]
```

たとえば `hand1024` なら、片側32 bucketなので、hand-axis の成分数は `32 + 32 = 64` です。1024個を直接持つよりかなり小さいため、rare な手駒組み合わせでも共有が効きます。

### king axis / progress axis

king axis と progress axis も考え方は同じです。

```text
W_king_axis[king_bucket]
W_progress_axis[progress_bucket]
```

`k3k3` の king bucket は、先手玉側3区分と後手玉側3区分の組み合わせです。BulletOu の factorizer では、king axis も内部的にはその2方向へ分解して使います。`k3k3` なら king-axis の成分数は `3 + 3 = 6` です。

`progress8` なら progress-axis の成分数は8です。

## 5. `pair`

`pair` は、単独軸だけでなく、2軸の組み合わせでも成分を共有します。

```text
W_effective[hand, king, progress]
  = W_base[hand, king, progress]
  + alpha_shared   * W_shared
  + alpha_hand     * W_hand_axis[hand]
  + alpha_king     * W_king_axis[king]
  + alpha_progress * W_progress_axis[progress]
  + alpha_pair     * W_king_hand_pair[hand, king]
  + alpha_pair     * W_king_progress_pair[king, progress]
  + alpha_pair     * W_hand_progress_pair[progress, hand]
```

`--sfnn-factorizer pair` と書くと、`shared` と使える axis 成分も同時に有効になります。つまり、`pair` は「2軸だけを使う」という意味ではありません。

`hand1024_k3k3_progress4` の場合、使える pair 成分は次の3つです。

| pair 成分 | 共有の意味 | 成分数 |
|---|---|---:|
| `king-hand` | 同じ hand と king なら progress をまたいで共有 | `1024 * 9 = 9,216` |
| `king-progress` | 同じ king と progress なら hand をまたいで共有 | `9 * 4 = 36` |
| `hand-progress` | 同じ hand と progress なら king をまたいで共有 | `1024 * 4 = 4,096` |

たとえば `hand-progress` は、「同じ手駒状態かつ同じ進行度なら、玉位置が違っても使える成分」を持つ、という意味です。

## 6. 指定方法

よく使う指定は次の通りです。

| 指定 | 意味 |
|---|---|
| `--sfnn-factorizer shared` | 全 stack 共通成分だけを使う。デフォルト |
| `--sfnn-factorizer none` | factorizer を使わない |
| `--sfnn-factorizer axis` | architecture に存在する hand / king / progress の単独軸を使う |
| `--sfnn-factorizer pair` | `shared`、使える axis、使える pair をまとめて使う |
| `--sfnn-factorizer king=axis,hand=axis` | 軸ごとに指定する |
| `--sfnn-factorizer king-hand,hand-progress` | pair 成分を個別に指定する |

`hand1024_k3k3_progress4` で pair まで使う例:

```bash
--arch SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4
--sfnn-factorizer pair
```

この指定では、architecture に対応する範囲で次の成分が使われます。

```text
shared
king-axis
hand-axis
progress-axis
king-hand
king-progress
hand-progress
```

## 7. `--sfnn-factorizer-alpha`

`--sfnn-factorizer-alpha` は、factorizer 成分を forward でどれだけ足すかを変える係数です。

```text
W_effective = W_base + alpha * W_factorizer
```

`alpha=1.0` が標準です。`alpha=2.0` なら、その成分を2倍して足します。同時に、その factorizer tensor へ流れる勾配も2倍になります。指定範囲は `0.0` から `10.0` です。

全部を同じ強さにする場合:

```bash
--sfnn-factorizer pair
--sfnn-factorizer-alpha all=3.0
```

単独軸と2軸を同じ強さにする場合:

```bash
--sfnn-factorizer pair
--sfnn-factorizer-alpha axis=4.0,pair=4.0
```

hand axis だけを弱める場合:

```bash
--sfnn-factorizer axis
--sfnn-factorizer-alpha hand=0.80
```

`hand=` は hand-axis の強さを変えます。`hand-progress` や `king-hand` のような pair 成分の強さは `pair=` で変えます。

```bash
--sfnn-factorizer pair
--sfnn-factorizer-alpha hand=0.80,pair=2.0
```

この例では、hand-axis は0.8倍、pair 成分は2.0倍です。

`all=` と個別指定を組み合わせることもできます。後ろに書いた指定が優先されます。

```bash
--sfnn-factorizer pair
--sfnn-factorizer-alpha all=3.0,pair=4.0
```

この例では、`shared` と `axis` は3.0、`pair` は4.0です。

## 8. `nn.bin` 書き出し時の扱い

`nn.bin` を保存するときは、factorizer 成分は `W_effective` に畳み込まれます。

```text
W_export = W_base + alpha_shared * W_shared + ...
```

そのため、エンジン側は factorizer を知る必要がありません。エンジンが読むのは、畳み込み済みの通常の stack weight です。

一方、`state.bin` には base weight と factorizer tensor が分かれて保存されます。追加学習で factorizer 設定を変える場合は、開始時のログで factorizer の状態を確認してください。

## 9. 使い分けの目安

| 状況 | 試す候補 |
|---|---|
| `k3k3` だけで安定している | `shared` または `axis` |
| `k29k29` のように king bucket が多い | `king=axis` |
| `hand1024` を使う | `hand=axis`、または `pair` |
| `hand1024_k3k3_progress4` のように複数軸を掛け合わせる | `pair` |
| qloss が暴れる | `alpha` を上げる、または飽和ペナルティを試す |
| factorizer が強すぎて伸びない | `alpha` を下げる、または `none` で短く追加学習する |

大きな bucket 構成では、`none` は各 stack を独立に学習できますが、rare bucket が崩れやすくなります。`axis` や `pair` は自由度を少し制限する代わりに、似た bucket 同士で学習を共有できます。

## 10. 量子化飽和を抑える補助オプション

factorizer を強くしたり bucket 数を増やしたりすると、`nn.bin` へ量子化するときに i8 の上限付近へ張り付く重みが増えることがあります。その場合は、実験用に飽和ペナルティを指定できます。

```bash
--sfnn-saturation-penalty 1e-7
```

これはデフォルトでは無効です。量子化後の loss や accuracy だけが悪い場合の切り分けに使います。

## 11. rare bucket を count で弱く正則化する

`hand1024_k3k3_progress8` のように stack 数が多い構成では、ほとんど出現しない bucket があります。そういう bucket の個別成分を自由に動かしすぎると、少数の局面に引っ張られて崩れることがあります。

BulletOu では、教師データから bucket の出現回数を事前に数えて、その count に応じて base stack residual を弱く減衰できます。ここでいう residual は、factorizer で共有される成分ではなく、各 stack が個別に持っている base weight です。

まず count.bin を作ります。

```powershell
.\target\release\examples\bulletou.exe bucket-count `
  --teacher D:\sojoteam_datasets `
  --arch SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4 `
  --positions 500000000 `
  --buffer-mb 1024 `
  --read-buffers 3 `
  --output D:\BulletOu-snapshots\counts\hand1024-k3k3-progress4-count.bin
```

`--positions` を省略すると、指定した teacher path に含まれる全ファイルを1回だけ読んで count します。大きな教師データから一部だけサンプリングしたい場合は、上の例のように `--positions` を指定します。

`.psv` / `.bin` は固定長レコードなので、BulletOu は専用の高速経路で読みます。読み込み用 buffer を複数個用意し、片方を count している間に別の buffer へディスク読み込みします。実装上は queue で buffer を回しますが、動作としては ring buffer です。

| オプション | 意味 | 目安 |
|---|---|---|
| `--buffer-mb` | 1個の読み込み buffer の大きさ | デフォルト `1024` |
| `--read-buffers` | 読み込み buffer の個数 | デフォルト `3`、最低 `2` |

必要なメモリはおおよそ `--buffer-mb × --read-buffers` です。例えば `--buffer-mb 1024 --read-buffers 4` なら、読み込み buffer だけで約4GiB使います。大きくしすぎても必ず速くなるわけではありません。ディスク読み込みが波打つ場合は `3` か `4`、OSキャッシュ上の小さな入力では小さめの値が速いこともあります。

進捗表示では、開始からの平均速度と直近区間の速度を分けて出します。

```text
[count] ... avg_pos/s=... inst_pos/s=... read_wait=... count=...
```

`avg_pos/s` は開始からの平均、`inst_pos/s` は直近の進捗区間だけの速度です。`read_wait` が大きい場合は、count/decode 側ではなくディスク読み込み待ちが主なボトルネックです。`count` が大きい場合は、bucket のdecode/count側が主なボトルネックです。

学習時にそのファイルを指定します。

```powershell
--sfnn-factorizer pair `
--sfnn-factorizer-alpha all=1.0 `
--sfnn-bucket-counts D:\BulletOu-snapshots\counts\hand1024-k3k3-progress4-count.bin `
--sfnn-residual-count-confidence 1.0
```

`count.bin` には完全な LayerStack bucket ごとの出現回数が入ります。axis / pair factorizer に対する confidence も、この同じファイルから計算できます。BulletOu は、各 axis 行・pair 行を使う stack の count を合計して使います。

`--sfnn-*-count-confidence` は、指定しなければすべて無効です。`--sfnn-bucket-counts <count.bin>` だけを指定した場合は、count.bin の検証と統計表示だけを行い、学習内容は変えません。

### count に応じて residual を弱める

bucket 固有 residual だけを count に応じて抑えるには、次のように指定します。

```powershell
--sfnn-bucket-counts D:\...\count.bin `
--sfnn-residual-count-confidence 1.0
```

stack ごとの減衰係数は次の式です。

```text
decay_stack = max_decay * min(1, sqrt((confidence_count + 1) / (count_stack + 1)))
```

`max_decay` は最大減衰量です。`--sfnn-residual-count-confidence` を指定し、`--sfnn-residual-count-decay` を省略した場合、`max_decay` は `1e-7` になります。最大減衰量そのものを調整したいときだけ `--sfnn-residual-count-decay <値>` を指定します。

`confidence_count` は次のように計算されます。

```text
residual_params_per_bucket = 1 bucket が個別に持つ residual パラメーター数
confidence_count = residual_params_per_bucket * --sfnn-residual-count-confidence
```

つまり `--sfnn-residual-count-confidence 1.0` は、「bucket 固有 residual のパラメーター数と同じぐらいの出現回数があるまでは、その bucket 固有成分をまだ強く信用しない」という意味です。教師データ全体に対する割合ではなく、モデル側の自由度を基準にします。

| count | 挙動 |
|---:|---|
| `count <= confidence_count` | 最大の `max_decay` で residual を抑える |
| `count = 4 * confidence_count` | 約 `max_decay / 2` になる |
| count が十分多い | ほとんど効かなくなる |

この正則化は factorizer tensor には直接かけません。`shared` / `axis` / `pair` の共有成分は残し、bucket 固有の residual だけを count に応じて抑えます。そのため、rare bucket を完全に無視するのではなく、「まず共有成分を信じ、十分な出現回数がある bucket だけ個別成分を強く学習する」という挙動になります。

`--sfnn-factorizer-alpha all=1.0` の場合、forward で使う重みは次のように考えます。

```text
W_effective = W_residual + W_factorizer
```

count decay は `W_residual` にだけかかります。したがって、出現回数が少ない bucket では `W_residual` が小さく抑えられ、`W_effective` は factorizer 側の共有成分に寄ります。出現回数が多い bucket では decay が弱くなるので、必要なら `W_residual` を大きく学習できます。

つまり `all=1.0` は factorizer 成分を普通に足す設定で、count decay は「bucket 固有成分の自由度を count に応じて変える」設定です。count が多い bucket ほど `none` に近い自由度を持ち、count が少ない bucket ほど factorizer に寄ります。

### count に応じて axis / pair factorizer を弱める

axis 行・pair 行そのものを count に応じて弱めるには、次のように指定します。

```powershell
--sfnn-bucket-counts D:\...\count.bin `
--sfnn-axis-count-confidence 1.0 `
--sfnn-pair-count-confidence 1.0
```

BulletOu は、それぞれの axis 行・pair 行を使う LayerStack bucket の出現回数を合計します。そして、factorizer の足し込み量に次の係数を掛けます。

```text
confidence = count_term / (count_term + term_params * option_value)
```

`term_params` は、1つの axis 行または pair 行が L1/L2/L3 に持つパラメーター数です。option value が `0` なら係数は `1` になり、その factorizer 行は弱まりません。option を有効にしていて count が `0` の行は、係数が `0` になります。

alpha と count confidence を同時に使うと、実効重みは次のように考えられます。

```text
W_effective =
    W_residual
  + shared_alpha * W_shared
  + axis_alpha   * confidence_axis * W_axis
  + pair_alpha   * confidence_pair * W_pair
```

`shared` を広い prior として残しつつ、ほとんど出現していない axis / pair 行だけを弱くしておきたいときに使います。

### `count.bin` のファイル形式

通常は `bulletou.exe bucket-count` で作るので、手で書く必要はありません。外部ツールで読む場合の形式は次の通りです。整数はすべて little-endian です。

| 順序 | 型 | 内容 |
|---:|---|---|
| 1 | `u8[8]` | magic。ASCIIで `BOUCNT1\0` |
| 2 | `u32` | version。現在は `1` |
| 3 | `u32` | architecture名のbyte長 |
| 4 | `u8[arch_len]` | UTF-8のarchitecture名。例: `SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4` |
| 5 | `u64` | countに使った局面数 |
| 6 | `u32` | stack数 |
| 7 | `u32[stack数]` | 各LayerStack bucketの出現回数 |

`counts[i]` は LayerStack bucket index `i` の出現回数です。学習時に `--sfnn-bucket-counts` で指定すると、BulletOu はファイル内のarchitecture名と stack数が現在の `--arch` と一致するか確認します。

出現回数は `u32` なので、1つのbucketに 4,294,967,295 回を超えて入るような集計はできません。その場合は `--positions` を減らして集計範囲を小さくしてください。
