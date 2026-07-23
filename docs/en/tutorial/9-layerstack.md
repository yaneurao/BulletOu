# 9. LayerStack — pick a different sub-network per position

<a href="../../ja/tutorial/9-layerstack.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

A standard NNUE uses a single MLP to evaluate any position. **LayerStack-family** evaluation functions instead keep **several sub-networks** and pick one per position:

- The right "shape" of evaluation can differ between opening / middlegame / endgame, or between different king-position relationships.
- So the model keeps several independent small sub-networks, and at inference time picks just one based on the position.
- The **bucket selection logic** has to agree between the engine and bulletou, and is encoded in the `--arch` suffix.

In bulletou, LayerStack is used by the **SFNN family**. The suffix at the end of `--arch` selects the same bucket algorithm that the matching YaneuraOu build uses.

## 9.1 Choosing the LayerStack Suffix

| `--arch` suffix | Buckets | YaneuraOu-loadable | Description |
|---|---|---|---|
| **`hand256`** | 256 | yes | Side-to-move / non-side 4-bit hand-presence buckets. |
| **`hand256_k3k3`** | 2304 | yes | `hand256` bucket × `k3k3` bucket. Very large. |
| **`hand256_k9k9`** | 20736 | yes | `hand256` bucket × `k9k9` bucket. Huge. |
| **`hand1024`** | 1024 | yes | Side-to-move / non-side 5-bit hand-presence buckets. |
| **`hand1024_k3k3`** | 9216 | yes | `hand1024` bucket × `k3k3` bucket. Huge. |
| **`hand1024_k9k9`** | 82944 | yes | `hand1024` bucket × `k9k9` bucket. Extreme; expect very large VRAM and checkpoint sizes. |
| **`hand64`** | 64 | yes | Side-to-move hand-score bucket (8 levels) × non-side hand-score bucket (8 levels). |
| **`hand64_k3k3`** | 576 | yes | `hand64` bucket × `k3k3` bucket. This is much larger on GPU/disk because every stack has its own MLP weights. |
| **`hand64_k9k9`** | 5184 | yes | `hand64` bucket × `k9k9` bucket. This is very large; use small FT/H1 sizes when experimenting. |
| **`k3k3(king3-by-king3)`** (default) | 9 | yes | Friend king's rank in 3 groups (1-3 / 4-6 / 7-9) × enemy king's rank in 3 groups = 9 combos. Matches YaneuraOu's `stack_index_for_nnue` exactly. |
| **`k9k9(king9-by-king9)`** | 81 | yes | Exact friend king rank × exact enemy king rank = 81 combos. |

`king9_by_king9`, `hand64_king3_by_king3`, `hand64_king9_by_king9`, `hand256_king3_by_king3`, `hand256_king9_by_king9`, `hand1024_king3_by_king3`, and `hand1024_king9_by_king9` are accepted as aliases for the corresponding short suffixes.

### The k3k3(king3-by-king3) bucket table

After perspective flipping, both kings' ranks are coarsened into three groups, then combined:

|  | enemy king rank 1-3 | enemy king rank 4-6 | enemy king rank 7-9 |
|---|---|---|---|
| **friend king rank 1-3** | bucket 0 | bucket 1 | bucket 2 |
| **friend king rank 4-6** | bucket 3 | bucket 4 | bucket 5 |
| **friend king rank 7-9** | bucket 6 | bucket 7 | bucket 8 |

Each bucket owns its own set of `fc_0 + fc_1 + fc_2` weights; during training, the weights of bucket *k* are only updated by positions classified into bucket *k*.

## 9.2 When it kicks in

LayerStack is **only meaningful for the SFNN family**. The other target families (`NNUE_halfkp_*` / `NNUE_kp_*` / `NNUE_halfkpe9_*` / `NNUE_halfkpvm_*` / `KPPT` family) use a single MLP.

```bash
# Train SFNN-1536 with k3k3(king3-by-king3) = 9 buckets
./target/release/examples/bulletou \
    --arch SFNN_halfkahm2_1536_15_32_k3k3 \
    --teacher teachers/

# Train HalfKA2 SFNN with the YaneuraOu hand64 bucket split
./target/release/examples/bulletou \
    --arch SFNN_halfka2_1024_7_64_hand64 \
    --teacher teachers/
```

Omitting `--output` puts checkpoints under `checkpoints/SFNN_HALFKA2HM-SFNN_halfkahm2_1536_15_32_k3k3/` (= the inferred target and arch values joined into the directory name).

Schedule flags (`--lr`, `--superbatches`, etc.) and the loss-log format are identical to the rest of the tutorial — see [§6 Tune the training](6-tune.md) and [§7 Inspect the result](7-result.md).

## 9.3 Verifying the load in YaneuraOu

LayerStack training writes a regular `nn.bin` (same path as any NNUE eval). To verify it loads, use the **YaneuraOu SFNNwoP1536 build** and run `setoption EvalDir → isready → bench`. An `info string Warning: NNUE hash mismatch` line on `isready` is expected (load continues).

Loading procedure is the same as in [§8 Load into an engine](8-engine.md). See the [SFNN-1536 reference](../shogi/sfnn-1536.md) for the YaneuraOu build setup that supports this family.

## 9.4 When you might want to skip LayerStack

Because LayerStack stores per-bucket weights, both training and inference are heavier than a single MLP. The hand-combined variants range from 576 stacks (`hand64_k3k3`) to 82944 stacks (`hand1024_k9k9`), so start with a small FT/H1 size or a hand-only suffix when testing the idea.

- If you only have a small teacher (e.g. < 100M positions), the per-bucket position count drops and each bucket trains less effectively.
- If you don't need YaneuraOu's SFNNwoP1536 build, sticking with `NNUE_HALFKP` / `NNUE_HALFKPVM` etc. is simpler.

Practical guidance:
- **Need a YaneuraOu SFNNwoP1536-compatible eval** → use `SFNN_HALFKA2HM` + LayerStack.
- **Otherwise** → a single-MLP NNUE architecture is enough.

## 9.5 Related

- [SFNN-1536 training reference](../shogi/sfnn-1536.md) — architecture, binary layout, quantisation scales
- Existing example: `examples/shogi_layerstack.rs` — has additional (experimental) bucket modes beyond k3k3(king3-by-king3) (rshogi-format output, kept parallel to bulletou)

---

Previous: [8. Load into an engine](8-engine.md)
