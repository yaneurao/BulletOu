# 9. LayerStack — pick a different sub-network per position

<a href="../../ja/tutorial/9-layerstack.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

A standard NNUE uses a single MLP to evaluate any position. **LayerStack-family** evaluation functions instead keep **several sub-networks** and pick one per position:

- The right "shape" of evaluation can differ between opening / middlegame / endgame, or between different king-position relationships.
- So the model keeps 9 independent small sub-networks, and at inference time picks just one based on the position.
- The **bucket selection logic** has to agree between the engine and bulletou, and is encoded in the `--arch` suffix.

In bulletou, LayerStack is currently used **only** for **YaneuraOu SFNNwoP1536-build training** (= `--eval-type SFNN_HALFKA1HM` / `SFNN_HALFKA2HM`). For the full spec see the [SFNN-1536 training reference](../shogi/sfnn-1536.md).

## 9.1 Choosing the LayerStack Suffix

| `--arch` suffix | Buckets | YaneuraOu-loadable | Description |
|---|---|---|---|
| **`k3k3(king3-by-king3)`** (default) | 9 | yes | Friend king's rank in 3 groups (1-3 / 4-6 / 7-9) × enemy king's rank in 3 groups = 9 combos. Matches YaneuraOu's `stack_index_for_nnue` exactly. |

This is currently the only supported suffix. If YaneuraOu adds a new bucket scheme in the future we'll add the matching suffix here.

### The k3k3(king3-by-king3) bucket table

After perspective flipping, both kings' ranks are coarsened into three groups, then combined:

|  | enemy king rank 1-3 | enemy king rank 4-6 | enemy king rank 7-9 |
|---|---|---|---|
| **friend king rank 1-3** | bucket 0 | bucket 1 | bucket 2 |
| **friend king rank 4-6** | bucket 3 | bucket 4 | bucket 5 |
| **friend king rank 7-9** | bucket 6 | bucket 7 | bucket 8 |

Each bucket owns its own set of `fc_0 + fc_1 + fc_2` weights; during training, the weights of bucket *k* are only updated by positions classified into bucket *k*.

## 9.2 When it kicks in

LayerStack is **only meaningful for the SFNN family**. The other eval-types (`NNUE_HALFKP` / `NNUE_KP` / `NNUE_HALFKPE9` / `NNUE_HALFKPVM` / `KPPT` family) use a single MLP.

```bash
# Train SFNN-1536 with k3k3(king3-by-king3) = 9 buckets
./target/release/examples/bulletou \
    --eval-type SFNN_HALFKA2HM \
    --arch SFNN_halfkahm2_1536_15_32_k3k3 \
    --teacher teachers/
```

Omitting `--output` puts checkpoints under `checkpoints/SFNN_HALFKA2HM-SFNN_halfkahm2_1536_15_32_k3k3/` (= the eval-type and arch values joined into the directory name).

Schedule flags (`--lr`, `--superbatches`, etc.) and the loss-log format are identical to the rest of the tutorial — see [§6 Tune the training](6-tune.md) and [§7 Inspect the result](7-result.md).

## 9.3 Verifying the load in YaneuraOu

LayerStack training writes a regular `nn.bin` (same path as any NNUE eval). To verify it loads, use the **YaneuraOu SFNNwoP1536 build** and run `setoption EvalDir → isready → bench`. An `info string Warning: NNUE hash mismatch` line on `isready` is expected (load continues).

Loading procedure is the same as in [§8 Load into an engine](8-engine.md). See the [SFNN-1536 reference](../shogi/sfnn-1536.md) for the YaneuraOu build setup that supports this family.

## 9.4 When you might want to skip LayerStack

Because LayerStack stores 9× the per-bucket weights, both training and inference are heavier than a single MLP.

- If you only have a small teacher (e.g. < 100M positions), the per-bucket position count drops and each bucket trains less effectively.
- If you don't need YaneuraOu's SFNNwoP1536 build, sticking with `NNUE_HALFKP` / `NNUE_HALFKPVM` etc. is simpler.

Practical guidance:
- **Need a YaneuraOu SFNNwoP1536-compatible eval** → use `SFNN_HALFKA2HM` + LayerStack.
- **Otherwise** → a single-MLP NNUE eval-type is enough.

## 9.5 Related

- [SFNN-1536 training reference](../shogi/sfnn-1536.md) — architecture, binary layout, quantisation scales
- Existing example: `examples/shogi_layerstack.rs` — has additional (experimental) bucket modes beyond k3k3(king3-by-king3) (rshogi-format output, kept parallel to bulletou)

---

Previous: [8. Load into an engine](8-engine.md)
