# 4. 蟄ｦ鄙偵ｒ襍ｰ繧峨○繧・窶・`bulletou` 繧ｳ繝槭Φ繝峨・螳溯｡・

<a href="../../en/tutorial/4-train.md"><img alt="Read in English" src="https://img.shields.io/badge/Lang-English-DC2626?style=flat-square"></a>

繧ｴ繝ｼ繝ｫ: 逕ｨ諢上＠縺滓蕗蟶ｫ繝・・繧ｿ縺九ｉ縲√ｄ縺ｭ縺・ｉ邇倶ｺ呈鋤繧ｨ繝ｳ繧ｸ繝ｳ縺瑚ｪｭ縺ｿ霎ｼ繧√ｋ隧穂ｾ｡髢｢謨ｰ繝舌う繝翫Μ繧貞ｭｦ鄙偵☆繧九・

縺薙・遶縺ｯ [3. 謨吝ｸｫ繝・・繧ｿ繧堤畑諢上☆繧犠(3-data.md) 繧貞ｮ御ｺ・＠縺ｦ縺・ｋ蜑肴署 窶・謨吝ｸｫ繝輔ぃ繧､繝ｫ (`.pack` / `.hcpe` / `.hcpe3` / `.psv`) 縺檎畑諢上＆繧後∽ｺ句燕縺ｫ繧ｷ繝｣繝・ヵ繝ｫ縺輔ｌ縺ｦ縺・ｋ迥ｶ諷九・

## 4.1 繝薙Ν繝・(1 蝗槭□縺・

縺ｾ縺・`bulletou` 繧偵ン繝ｫ繝峨☆繧九ゅた繝ｼ繧ｹ縺ｫ螟画峩縺檎┌縺代ｌ縺ｰ蛻晏屓 1 蝗槭□縺代〒 OK縲る・↓縲。ulletOu 縺ｮ繧ｽ繝ｼ繧ｹ繧呈峩譁ｰ縺励◆逶ｴ蠕後・縲∵里蟄倥・ `.\target\release\examples\bulletou.exe` 縺ｯ蜿､縺・∪縺ｾ縺ｪ縺ｮ縺ｧ蠢・★蜀阪ン繝ｫ繝峨☆繧・

```bash
cargo build --release --features cuda-cpp-backend --example bulletou
```

Windows 縺ｮ蝣ｴ蜷医∫函謌舌＆繧後ｋ繝舌う繝翫Μ縺ｯ `.\target\release\examples\bulletou.exe`縲ゆｻ･荳九・繧ｳ繝槭Φ繝我ｾ九・ Unix 蠖｢蠑上〒譖ｸ縺上・縺ｧ驕ｩ螳懆ｪｭ縺ｿ譖ｿ縺医・
## 4.2 譛蟆上さ繝槭Φ繝・(NNUE HalfKP)

```bash
./target/release/examples/bulletou \
    --arch NNUE_halfkp_256x2_32_32 \
    --teacher teachers/
```

縺薙ｌ縺縺代〒蜍輔￥縲Ａ--output` 繧堤怐逡･縺励※縺・ｋ縺ｮ縺ｧ縲…heckpoint 縺ｯ `checkpoints/NNUE_HALFKP-NNUE_halfkp_256x2_32_32/` 驟堺ｸ九↓譖ｸ縺九ｌ繧・(`--arch` 縺九ｉ閾ｪ蜍募多蜷・縲ょ挨縺ｮ蝣ｴ謇縺ｫ譖ｸ縺阪◆縺・ｴ蜷医・ `--output checkpoints/my-halfkp` 縺ｮ繧医≧縺ｫ譏守､ｺ縺吶ｋ縲・
## 4.3 `--arch` 繧呈欠螳壹☆繧・

蟄ｦ鄙貞ｯｾ雎｡縺ｯ `--arch` 縺縺代〒謖・ｮ壹☆繧九・PPT 邉ｻ縺ｪ繧・`KPPT` / `KPP_KKPT`縲¨NUE / SFNN 邉ｻ縺ｪ繧・**繧・・縺・ｉ邇九・ Makefile edition 蜷阪°繧・`YANEURAOU_ENGINE_` 繧貞叙繧企勁縺・◆蜷榊燕**繧呈欠螳壹☆繧九ゅ◆縺ｨ縺医・ HalfKP 縺ｮ 256x2-32-32 縺ｪ繧・`NNUE_halfkp_256x2_32_32`縲゜-P 縺ｮ 256x2-32-32 縺ｪ繧・`NNUE_kp_256x2_32_32`縲ヾFNN 縺ｪ繧・`SFNN_halfka2_1024_7_64_k3k3` 縺ｮ繧医≧縺ｫ譖ｸ縺上ょ商縺・洒邵ｮ蠖｢ `256x2-32-32` 縺ｯ蜿励￠莉倥￠縺ｪ縺・・
NNUE 邉ｻ縺ｮ繧ｵ繧､繧ｺ驛ｨ蛻・・ `<L1>x2_<L2>_<L3>` 縺ｧ縲～L1` (perspective 縺斐→縺ｮ accumulator 繧ｵ繧､繧ｺ) 縺ｯ **32 縺ｮ蛟肴焚** (FT SIMD 繝代ョ繧｣繝ｳ繧ｰ隕∽ｻｶ) 縺ｧ豁｣縺ｮ謨ｴ謨ｰ縲～L2` / `L3` 縺ｯ豁｣縺ｮ謨ｴ謨ｰ縺ｪ繧我ｽ輔〒繧ょ女縺台ｻ倥￠繧九ゅｈ縺丈ｽｿ繧上ｌ繧九し繧､繧ｺ縺ｯ莉･荳・

| 繧ｵ繧､繧ｺ繧ｵ繝輔ぅ繝・け繧ｹ | L1 (accumulator) | L2 | L3 | 逕ｨ騾斐・逶ｮ螳・|
|---|---|---|---|---|
| `256x2-32-32` (繝・ヵ繧ｩ繝ｫ繝・ | 256 | 32 | 32 | 蜿､蜈ｸ逧・↑蟆丞梛 NNUE縲ょｭｦ鄙呈凾髢薙′遏ｭ縺乗嫌蜍慕｢ｺ隱榊髄縺・|
| `384x2-8-96` | 384 | 8 | 96 | |
| `512x2-8-64` | 512 | 8 | 64 | 荳ｭ蝙・|
| `768x2-16-64` | 768 | 16 | 64 | |
| `1024x2-8-32` | 1024 | 8 | 32 | 螟ｧ蝙・(謗ｨ隲悶さ繧ｹ繝医・蠅励∴繧・ |
| `1024x2-8-64` | 1024 | 8 | 64 | 螟ｧ蝙・|
| `SFNN_halfkahm2_1536_15_32_k3k3` | 1536 | 15 | 32 | k3k3(king3-by-king3) LayerStacks 縺ｮ SFNN-1536 |
| `SFNN_halfka2_4096_3_64_c0_s1024x4_k3k3` | 4096 | 3 | 64 | grouped SFNN L1縲・096 繧・4 group 縺ｫ蛻・￠縲∝推 group 縺ｯ 1024 -> 1 |
| `SFNN_halfka2_8192_3_64_c0_s2048x4_k3k3` | 8192 | 3 | 64 | grouped SFNN L1縲・192 繧・4 group 縺ｫ蛻・￠縲∝推 group 縺ｯ 2048 -> 1 |
| `SFNN_halfka2_4096_7_64_c0_s1024x4_k3k3` | 4096 | 7 | 64 | grouped SFNN L1縲・096 繧・4 group 縺ｫ蛻・￠繧・|
| `SFNN_halfka2_1024_7_64_hand64` | 1024 | 7 | 64 | 繧・・縺・ｉ邇・hand64 LayerStack bucket (64 stacks) |
| `SFNN_halfka2_1024_7_64_hand64_k3k3` | 1024 | 7 | 64 | hand64 ﾃ・k3k3 LayerStack bucket (576 stacks縲√°縺ｪ繧雁､ｧ縺阪＞) |
| `SFNN_halfka2_1024_7_64_k9k9` | 1024 | 7 | 64 | king9-by-king9 LayerStack bucket (81 stacks) |
| `SFNN_halfka2_1024_7_64_k29k29` | 1024 | 7 | 64 | king29-by-king29 LayerStack bucket (841 stacks) |
| `SFNN_halfka2_1024_7_64_hand64_k9k9` | 1024 | 7 | 64 | hand64 ﾃ・k9k9 LayerStack bucket (5184 stacks縲・撼蟶ｸ縺ｫ螟ｧ縺阪＞) |
| `SFNN_halfka2_1024_7_64_hand64_k29k29` | 1024 | 7 | 64 | hand64 ﾃ・k29k29 LayerStack bucket (53824 stacks縲∝ｷｨ螟ｧ) |
| `SFNN_halfka2_1024_7_64_hand256` | 1024 | 7 | 64 | hand256 謇矩ｧ呈怏辟｡ LayerStack bucket (256 stacks) |
| `SFNN_halfka2_1024_7_64_hand256_k3k3` | 1024 | 7 | 64 | hand256 ﾃ・k3k3 LayerStack bucket (2304 stacks縲・撼蟶ｸ縺ｫ螟ｧ縺阪＞) |
| `SFNN_halfka2_1024_7_64_hand1024` | 1024 | 7 | 64 | hand1024 謇矩ｧ呈怏辟｡ LayerStack bucket (1024 stacks) |
| `SFNN_halfka2_1024_7_64_hand1024_k3k3` | 1024 | 7 | 64 | hand1024 ﾃ・k3k3 LayerStack bucket (9216 stacks縲∝ｷｨ螟ｧ) |
| `SFNN_ka2_4096_15_64_c0_s256x16_k3k3` | 4096 | 15 | 64 | 霆ｽ驥上↑ KA2 蜈･蜉帙・ grouped SFNN |
| `SFNN_halfka2_8192_7_64_c0_s1024x8_k3k3` | 8192 | 7 | 64 | common+shard 陦ｨ險倥・ pure grouped L1縲・ common + 1024 x 8 shards |
| `SFNN_ka2_3072_7_64_c1024_s256x8_k3k3` | 3072 | 7 | 64 | common+shard SFNN L1縲・024 common + 256 x 8 shards |

```bash
./target/release/examples/bulletou \
    --arch NNUE_halfkp_1024x2_8_64 \
    --teacher teachers/
```

`--arch` 縺ｯ蟄ｦ鄙貞ｯｾ雎｡縺ｨ architecture 縺ｮ蜚ｯ荳縺ｮ謖・ｮ壹↑縺ｮ縺ｧ縲・壼ｸｸ縺ｮ蟄ｦ鄙偵〒縺ｯ蠢・医ゆｸ願ｨ倥・陦ｨ縺ｫ辟｡縺・し繧､繧ｺ繧ょｮ滄ｨ鍋畑騾斐〒蜿励￠莉倥￠繧九′縲∝ｭｦ鄙堤ｵ先棡縺ｮ `nn.bin` 繧・load 縺ｧ縺阪ｋ縺ｮ縺ｯ縲悟酔縺・architecture 繝倥ャ繝縺ｧ build 縺励◆繧・・縺・ｉ邇九阪□縺代Ａmake` 縺ｫ蟇ｾ蠢懊☆繧・edition 蜷阪ｒ貂｡縺励※繝薙Ν繝峨☆繧句ｿ・ｦ√′縺ゅｋ (隧ｳ邏ｰ縺ｯ [ﾂｧ8 Engine](8-engine.md))縲・
grouped SFNN 縺ｮ螳滄ｨ薙・ LayerStack suffix 縺ｮ蜑阪↓ `_c0_sMxG_` 縺ｧ譖ｸ縺代ｋ縲ゅ◆縺ｨ縺医・ `SFNN_halfka2_8192_7_64_c0_s1024x8_k3k3` 縺ｯ `FT=8192`, `L1 hidden=7`, `L2=64`, `L1 繧・1024 x 8 shards 縺ｫ蛻・牡` 縺ｨ縺・≧諢丞袖縲Ｄommon 縺碁撼繧ｼ繝ｭ縺ｮ common+shard L1 繧ょ酔縺・`_cN_sMxG_` 繧剃ｽｿ縺・ゆｾ・ `SFNN_ka2_3072_7_64_c1024_s256x8_k3k3` 縺ｯ 1024 common + 256 x 8 shards縲Ｔuffix 縺ｯ `k3k3`, `k9k9`, `k29k29`, `hand64`, `hand64_k3k3`, `hand64_k9k9`, `hand64_k29k29`, `hand256`, `hand256_k3k3`, `hand256_k9k9`, `hand256_k29k29`, `hand1024`, `hand1024_k3k3`, `hand1024_k9k9`, `hand1024_k29k29` 繧呈欠螳壹〒縺阪ｋ縲Ａka2` / `halfka2` 縺ｪ縺ｩ縺ｮ feature 蜷阪°繧牙・驛ｨ target 縺ｯ閾ｪ蜍慕噪縺ｫ豎ｺ縺ｾ繧九・
## 4.4 SFNN-1536 (繧・・縺・ｉ邇・NNUEwoSQPT1536) 繧貞ｭｦ鄙偵☆繧・

繧・・縺・ｉ邇九・ **`YANEURAOU_ENGINE_SFNN1536` 繝薙Ν繝・* 縺ｫ load 縺輔○繧玖ｩ穂ｾ｡髢｢謨ｰ繧貞ｭｦ鄙偵＠縺溘＞蝣ｴ蜷医・縲∝ｯｾ蠢懊☆繧・architecture 蜷阪ｒ謖・ｮ壹☆繧・

```bash
./target/release/examples/bulletou \
    --arch SFNN_halfkahm2_1536_15_32_k3k3 \
    --teacher teachers/
```

騾壼ｸｸ縺ｮ NNUE 縺ｨ驕輔▲縺ｦ **9 蛟九・繧ｵ繝悶ロ繝・ヨ繧貞ｱ髱｢縺斐→縺ｫ菴ｿ縺・・縺代ｋ** (LayerStacks=9)縲Ａ--arch` 縺ｮ `k3k3` suffix 縺後ｄ縺ｭ縺・ｉ邇倶ｺ呈鋤縺ｮ LayerStack 譁ｹ蠑上ｒ驕ｸ縺ｶ縲ゆｽｿ縺・婿縺ｮ隱ｬ譏弱・ [ﾂｧ9 LayerStack](9-layerstack.md)縲√い繝ｼ繧ｭ繝・け繝√Ε / 驥丞ｭ仙喧 / `nn.bin` 繝ｬ繧､繧｢繧ｦ繝医・莉墓ｧ倥・ [繝ｪ繝輔ぃ繝ｬ繝ｳ繧ｹ: SFNN-1536](../shogi/sfnn-1536.md)縲・

## 4.5 KPPT 繧貞ｭｦ鄙偵☆繧・

KPPT 邉ｻ縺ｧ縺ｯ蝗ｺ螳・target 蜷阪ｒ `--arch` 縺ｫ謖・ｮ壹☆繧・

```bash
./target/release/examples/bulletou \
    --arch KPPT \
    --teacher teachers/
```

繝・ヵ繧ｩ繝ｫ繝亥・蜉帛・縺ｯ `checkpoints/KPPT/`縲Ｇactorised 迚医↓縺励◆縺代ｌ縺ｰ `--arch KPP_KKPT` 縺ｫ螟峨∴繧九□縺代・
## 4.6 謨吝ｸｫ繝・・繧ｿ縺ｮ貂｡縺玲婿

`--teacher` 縺ｫ縺ｯ:
- 1 縺､縺ｮ繝輔ぃ繧､繝ｫ (`teachers/teacher.pack` 縺ｮ繧医≧縺ｪ繝輔Ν繝代せ)
- 繝・ぅ繝ｬ繧ｯ繝医Μ (荳願ｨ倅ｾ九ゆｸｭ縺ｮ蜷御ｸ諡｡蠑ｵ蟄舌ヵ繧｡繧､繝ｫ縺後☆縺ｹ縺ｦ騾｣邨舌＆繧後ｋ)
- 繧ｫ繝ｳ繝槫玄蛻・ｊ隍・焚謖・ｮ・

縺ｮ縺・★繧後ｂ貂｡縺帙ｋ縲・

## 4.7 蟄ｦ鄙偵′縺ｩ縺薙∪縺ｧ騾ｲ繧縺・

`--superbatches` 繧・`--max-epochs` 繧ら怐逡･縺励※縺・ｋ縺ｮ縺ｧ縲∵蕗蟶ｫ繝・・繧ｿ繧・1 蜻ｨ (dataloader 縺・EOF 繧定ｿ斐☆縺ｾ縺ｧ) 縺ｧ蟄ｦ鄙偵′邨ゆｺ・☆繧九り､・焚 epoch 蝗槭＠縺溘＞蝣ｴ蜷医・ `--superbatches` 縺ｧ epoch 髟ｷ繧呈ｱｺ繧√◆縺・∴縺ｧ `--max-epochs 3` 縺ｮ繧医≧縺ｫ謖・ｮ壹☆繧九Ａstep` / `geometric` / `cos` 縺ｯ epoch 蠅・阜縺ｧ `--lr` 縺ｫ謌ｻ繧九・

謨吝ｸｫ繧ｵ繧､繧ｺ縺御ｺ句燕縺ｫ繧上°縺｣縺ｦ縺・ｋ縺ｨ `--superbatches N` 縺ｧ縲・ epoch = N sb縲阪ｒ譏守､ｺ縺ｧ縺阪ｋ ([ﾂｧ6.1 蟄ｦ鄙偵せ繧ｱ繧ｸ繝･繝ｼ繝ｫ](6-tune.md#61-蟄ｦ鄙偵せ繧ｱ繧ｸ繝･繝ｼ繝ｫ) 蜿ら・)縲よ蕗蟶ｫ縺ｮ邱丞ｱ髱｢謨ｰ繧剃ｸ迸ｬ縺ｧ謨ｰ縺医ｋ `--count-teacher` 繝輔Λ繧ｰ縺後≠繧・

```bash
./target/release/examples/bulletou --count-teacher --teacher teachers/
# 竊・"Total: 461373440 positions, suggested --superbatches 4"
```

cosine annealing (`--lr-schedule cos`) 繧剃ｽｿ縺・→縺阪・迚ｹ縺ｫ驥崎ｦ・窶・1 cycle 縺・1 epoch 縺ｨ縺ｴ縺｣縺溘ｊ蜷医≧繧医≧縺ｫ `--superbatches` 繧帝∈縺ｶ縺ｨ縲∝推 epoch 譛ｫ縺ｧ lr_min 縺ｫ逹蝨ｰ縲∵ｬ｡ epoch 鬆ｭ縺ｧ lr_max 縺ｫ warm restart縲√→縺・≧縺阪ｌ縺・↑繧ｵ繧､繧ｯ繝ｫ縺ｫ縺ｪ繧九ゅ％縺ｮ蝣ｴ蜷医∵蕗蟶ｫ繝・・繧ｿ閾ｪ菴薙・ epoch 蠅・阜縺ｧ蜈磯ｭ縺ｸ謌ｻ繧峨↑縺・よ蕗蟶ｫEOF縺ｫ蛻ｰ驕斐＠縺溘→縺阪□縺大・鬆ｭ縺ｸ謌ｻ繧・cyclic stream 縺ｨ縺励※謇ｱ繧上ｌ繧九・

## 4.8 譛溷ｾ・＆繧後ｋ蜃ｺ蜉・

蜍輔＞縺ｦ縺・ｌ縺ｰ莉･荳九・繧医≧縺ｪ蜃ｺ蜉帙′豬√ｌ繧・

```
=== bulletou: running NNUE_HALFKP (256x2-32-32 ClippedReLU, dual-perspective) ===
Training Preamble
Net Name               : shogi_nnue_halfkp
Batch Size             : 65536
Batches / Superbatch   : 1525
Positions / Superbatch : 99942400
...
  cuda-cpp loss progress log = checkpoints/.../cuda-cpp-progress.log (step 1, every 10 step(s), checkpoint, final)
  cuda-cpp SFNN checkpoint: epoch=1 sb=1/36 batch=2543/2543 positions=41664512 pos/s=... dir=checkpoints/.../0001
  cuda-cpp SFNN validation summary: epoch=1, superbatch=1, test_value_accuracy=..., test_value_loss=...
  cuda-cpp SFNN direct train = ok: steps=..., positions=..., train_elapsed=...s, elapsed=...s, throughput=... pos/s, ...
```

cuda-cpp backend 縺ｮ stdout `pos/s` 縺ｯ checkpoint file save / validation / loss readback / progress-log write 縺ｮ譎る俣繧帝勁螟悶＠縺溽ｴ皮ｲ九↑蟄ｦ鄙・throughput縲Ｃatch 蛻･ loss 縺ｯ stdout 縺ｫ豬√＆縺壹～<output>/cuda-cpp-progress.log` 縺ｫ CSV 縺ｧ霑ｽ險倥＆繧後ｋ縲・
`pos/s` (1 遘偵≠縺溘ｊ蜃ｦ逅・ｱ髱｢謨ｰ) 縺悟ｭｦ鄙帝溷ｺｦ縺ｮ逶ｮ螳峨３TX 4090 1 譫壹〒謨ｰ蜊・ｸ・pos/s 蜃ｺ繧九ゆｸ倶ｽ・GPU 縺ｧ縺ｯ豈比ｾ九＠縺ｦ菴惹ｸ九・

---

谺｡縺ｸ:
- 蟄ｦ鄙偵ｒ荳ｭ譁ｭ縺励◆繧雁・髢九＠縺溘＞蝣ｴ蜷医・ [5. 荳ｭ譁ｭ繝ｻ蜀埼幕](5-resume.md)
- 蟄ｦ鄙偵・繧ｹ繧ｱ繧ｸ繝･繝ｼ繝ｫ繧・蕗蟶ｫ繧ｿ繝ｼ繧ｲ繝・ヨ繧定ｪｿ謨ｴ縺励◆縺・ｴ蜷医・ [6. 蟄ｦ鄙偵ｒ繝√Η繝ｼ繝九Φ繧ｰ](6-tune.md)
- 蟄ｦ鄙堤ｵ先棡縺後ｂ縺・焔蜈・↓縺ゅｋ縺ｪ繧・[7. 邨先棡繧堤｢ｺ隱江(7-result.md) 縺ｸ

蜑阪∈: [3. 謨吝ｸｫ繝・・繧ｿ繧堤畑諢上☆繧犠(3-data.md)
