# 3. KPPT / KPP_KPPT Roadmap

<a href="../../ja/tutorial/3-kppt-roadmap.md"><img alt="日本語で読む" src="https://img.shields.io/badge/Lang-日本語-DC2626?style=flat-square"></a>

> **Status: Phase 1 (KK only) is in.** The full KPPT / KPP_KPPT support
> is rolling out in phases:
>
> - **Phase 1 (DONE)** — `ShogiKk` sparse input + `shogi_kk_train` example.
>   The training pipeline runs end-to-end on a KK-only minimal network.
>   Strength is expected to be poor; this phase only confirms wire-up.
> - **Phase 2 (planned)** — add `ShogiKkp` and combine with KK.
> - **Phase 3 (planned)** — add `ShogiKpp` (= full KPPT). Requires large
>   GPU memory: KPP weights alone are ~1.4 GB (5.6 GB+ with Adam state).
> - **Phase 4 (planned)** — emit YaneuraOu-compatible `KK_synthesized.bin`
>   / `KKP_synthesized.bin` / `KPPT_synthesized.bin`.
> - **Phase 5 (planned)** — `KPP_KPPT` factorised form.

## Why support KPPT / KPP_KPPT?

YaneuraOu has historically supported a family of evaluation functions before NNUE:

- **KK** — king vs king position only
- **KKP** — king × king × piece
- **KPP** — king × piece × piece (the original "Apery / Bonanza" style)
- **KPPT** — KPP plus a side-to-move tensor T
- **KPP_KPPT** — a factorised form of KPPT (KP + KPPT, sharing parameters)

Many strong shogi engines were built on these. There is still value in:

- Continuing to improve / re-train classical evals as a research baseline
- Using BulletOu's GPU pipeline to do what was historically very slow CPU-only training
- Comparing classical and NNUE evals on the same training data

## Structural difference from NNUE

NNUE is "**sparse feature transformer + small MLP**" — a (fairly) standard neural network shape that fits naturally into bullet's IR.

KPPT is "**sum of large sparse embedding tables, no hidden layers**":

```
eval(pos) = KK[bk][wk] 
          + Σ_i KKP[bk][wk][p_i] 
          + Σ_{i<j} KPP[bk][p_i][p_j]
          + (turn term T)
```

There is no "hidden layer" in the NN sense. It is a giant lookup-table sum.

The biggest single table (`KPP`) is roughly **184 M parameters** (`81 × 1548 × 1548 / 2 × 2 channels`), which is a different scale of memory than the typical 1–10 M parameter NNUE.

## What BulletOu needs to add

Tracking the spec discussion in [`docs/spec/bullet/shogi-port.md`](https://github.com/yaneurao/YaneuraOu) (yaneurao's workspace-side spec, not in this repo):

1. **Tuple `SparseInputType`s** for KK / KKP / KPP — each defines its own `num_inputs()`, `max_active()`, `map_features()`. The hard part is BonaPiece pair enumeration for KPP.
2. **Multi-input `ValueTrainerBuilder`** — currently `inputs(SingleInputType)` is a single trait object; KPPT needs `inputs((Kk, Kkp, Kpp))` style tuple support. This is a non-trivial change in the builder DSL.
3. **No-hidden-layer architecture** — directly sum embedding outputs, no MLP after the FT.
4. **YaneuraOu-format writer** — BulletOu's current `SavedFormat` produces rshogi-compatible quantised binaries. KPPT/KPP_KPPT need to produce YaneuraOu's `KK_synthesized.bin`, `KKP_synthesized.bin`, `KPPT_synthesized.bin` triplet with the exact layout YaneuraOu's `evaluate_kppt.cpp` expects.
5. **Training schedule tuning** — KPPT historically used ELMO-style WDL teacher, strong weight decay, and small lrs. The hyperparameters that work for NNUE will not directly transfer.

## Estimated size of work

From the workspace-side spec (`docs/spec/bullet/shogi-port.md`):

| Component | Lines (rough) |
|---|---|
| KK / KKP / KPP SparseInputType implementations | 600–1,200 |
| ValueTrainerBuilder tuple-input extension | 200–500 (in bullet core, may need to be upstreamed or maintained as a fork patch) |
| YaneuraOu-format weight writer | 300–500 |
| Factorisation helpers (for KP_KPPT) | 200–400 |
| Schedule and regularisation tuning | mostly experimental cost, not LoC |
| **Total** | **~1,300–2,600 LoC + experiment time** |

In contrast, NNUE shogi support (already done in upstream `bullet-shogi`) is ~17,000 LoC, so KPPT support is much smaller in pure LoC terms — but it touches the bullet core builder DSL, which is a sensitive area.

## What the user-visible interface will probably look like

When KPPT support lands, the expected end-user workflow will mirror the NNUE one:

```bash
cargo run --release --features cuda --example shogi_kppt_train -- \
  --data /data/shogi/train.pack \
  --output checkpoints/my-kppt-eval \
  --eval-format kppt \
  --epochs 1
```

with output files following YaneuraOu's KPPT layout (`KK_synthesized.bin`, `KKP_synthesized.bin`, `KPPT_synthesized.bin`) directly in the checkpoint directory.

`KPP_KPPT` would be a switch on top: `--eval-format kpp_kppt` (or a similar flag) plus an optional factorisation control.

## What you can do today

Until KPPT support is implemented:

- Use **YaneuraOu's own `learn` command** for KPPT-family training (the original implementation; CPU-only, slow but functional).
- Use **BulletOu for NNUE**, where the GPU acceleration is fully working.
- If you want to help implement KPPT support, the BonaPiece pair enumeration code in `crates/bullet_lib/src/shogi/bona_piece.rs` is the natural starting point. The math is well-documented in `docs/en/shogi/kp-absolute-progress.md` (BonaPiece layout) and `docs/spec/bullet/shogi-port.md` (overall design).

---

When KPPT lands, this page will be split into a real KPPT tutorial alongside [2. NNUE Tutorial](2-nnue-tutorial.md). Until then, it serves as a forward-looking design summary.
