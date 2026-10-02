# Persistent FT saturation penalty (experimental)

This opt-in linear hinge penalty targets FT units that remain almost always upper-saturated. It does not reset units or replace the ordinary clamp backward with an STE. Saturation alone does not prove a strength problem: compare OFF/ON experimentally.

| JSON key (replace underscores with hyphens for CLI) | Default | Meaning |
|---|---:|---|
| `sfnn_ft_saturation_penalty` | 0 | Finite nonnegative strength λ; zero disables all extra work |
| `sfnn_ft_saturation_rate` | 0.2 | Training upper-saturation fraction threshold, in (0,1]; 0.2 means 20% |
| `sfnn_ft_saturation_patience` | 1 | Positive number of consecutive training microbatches |

All three support epoch schedules. Future settings do not affect earlier epochs.

By default, a unit qualifies immediately in a batch with at least 20% upper saturation.

For each training microbatch, count upper-saturated activations per FT unit across both perspectives. Validation data is not used. Zero entry weights are excluded from detection; positive weights count equally. A unit becomes eligible on the Nth consecutive qualifying batch. Falling below the threshold, or an entirely filtered batch, resets its streak. With bpu=4, four microbatches count as four, not one optimizer update.

With pre-clamp activation $z_{isu}$, batch size $B$, FT width $F$, entry weight $w_i$, and a non-differentiated eligibility mask $m_u$, the additional gradient corresponds to

$$
L_{\mathrm{FT}}=\frac{\lambda}{2BF}\sum_{i,s,u}w_i m_u\max(0,z_{isu}-1).
$$

For $z>1$, the additional derivative is $\lambda w_i m_u/(2BF)$; for $z<1$, zero. At $z=1$, the positive boundary subgradient is chosen. This bypasses the zero derivative of the ordinary clamp. Clamped FP32 activations suffice to detect the boundary, avoiding an additional pre-activation array. Only the penalty gradient is computed, not its scalar value. Gradients go to FT weights and biases, including the FT factorizer alpha chain rule. Accumulation uses the existing microbatch averaging at optimizer update. Other layers receive no direct penalty.

Reported loss/qloss exclude this penalty. It is separate from the existing `sfnn_saturation_penalty` weight penalty.

Append to an existing grid command:

```powershell
  --grid sfnn_ft_saturation_penalty 0 0.001 0.01 0.1 1.0
```

These are starting points, not validated optimal strengths. Keep initialization, learning rate and training budget identical. Use `--verbose` to compare mean and maximum per-unit saturation as well as accuracy and playing strength.

The gradient is averaged over FT width as well. With F=1024 and strength=0.001, even a fully saturated unit receives at most about 0.000000977 additional bias gradient when entry weights are one. If small strengths have little effect, test a logarithmically wider range as above; 1.0 is not a recommended optimum.

Supported: cuda-cpp SFNN, update scope=all, FT factorization, QAT, L1/L2/L3 centering, effective L1 clipping and bpu>1. Frozen FT skips the penalty. Detection stays on GPU without readback. History requires about 8KiB for F=1024, but activation scanning and gradient application add training cost. OFF performs no extra scans or allocations.

History is transient: resume, worker snapshot restoration, configuration changes and disabling reset it. Checkpoints and nn.bin formats and inference are unchanged. Existing running jobs/settings are not modified; start or resume with the rebuilt executable to use the option.

The penalty is merged into FT activation gradients after ordinary clamp backward. Task and penalty weight gradients share ONE feature-index construction/gather, eliminating the separate penalty weight-gradient pass. Saturation counting and reduced bias gradients remain separate small GPU operations. Existing backward workspace is reused without an extra full FT activation array. The formula, threshold and strength semantics are unchanged. Parallel accumulation order changes, so floating-point results are not bitwise identical.

Developer diagnostics: `BULLETOU_FT_GUARD_AUDIT=1` deliberately uses the old separate penalty pass to compare gradients before/after addition, logging gradients, momentum and CUDA timing at microbatches 1/10/100/610/1220 after history initialization/reset. This mode introduces large CPU readbacks and synchronization and does not measure normal fused performance; leave it unset for normal training. Pre-clamp reconstruction is a HalfKA2 diagnostic using the first 4096 positions.
