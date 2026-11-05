# Training with BCE with Logits

[日本語](../../ja/advanced/bce-with-logits.md)

Supported by the cuda-cpp trainer, worker, validation, and GPU/CPU quantized validation. The default remains the existing squared probability-space WRM loss.

## Configuration

Add these fields to your training JSON, preserving the other training settings:

```json
{
  "loss_bce_with_logits": true,
  "wrm_in_offset": 0
}
```

The CLI equivalent is `--loss-bce-with-logits --wrm-in-offset 0`.
Remove or disable `win_rate_model` and `loss_sigmoid_mse`; explicit selection of either conflicts with BCE.

Standard BCE logits do not use a prediction-side WRM offset. A nonzero `wrm_in_offset` is rejected rather than silently ignored (the ordinary WRM default is270, so specify0). `loss_pow_exp` is unused for BCE and must remain at its default2.

## Formula

Let `y` be the train-scale network output, `s` the teacher score, and `σ(x)=1/(1+exp(-x))`.

```text
z = y * wrm_nnue2score / wrm_in_scaling
p = σ(z)
F(s) = (1 + σ((s - wrm_target_offset)/wrm_target_scaling)
          - σ((-s - wrm_target_offset)/wrm_target_scaling)) / 2
t_score = ε + (1 - 2ε) * F(s)       ε = wrm_target_epsilon
t = lambda * t_score + (1-lambda) * game_result
L = -t*log(p) - (1-t)*log(1-p)
  = max(z,0) - t*z + log(1+exp(-abs(z)))
dL/dy = (p-t) * wrm_nnue2score / wrm_in_scaling
```

`game_result` is1 for wins,0.5 for draws, and0 for losses. Teacher offset, scaling, epsilon, and lambda retain their existing meanings. The implementation evaluates a stable logit-space expression, avoiding log(0) for large outputs. Training applies the existing per-position weight and batch mean.

Loss/qloss now report BCE, excluding Norm loss and other regularizers. Accuracy definitions do not change. GPU approximate versus CPU integer qvalid inference remains unchanged. Pass the same BCE and WRM options to `quantized-test`. Train-scale qloss uses raw/8128 without FV_SCALE; CPU mode's separately reported engine-scale loss uses raw/FV_SCALE with nnue2score=1. GPU mode retains the existing behavior: only train-scale loss is calculated, and the engine-scale field repeats that value.

**BCE values are not directly comparable with squared-error loss values.** BCE does not automatically prevent saturation: targets0/1 have no finite optimal logit. Consider a separate teacher-epsilon experiment if needed. Gradient magnitudes differ, so the same learning rate is not necessarily appropriate.

## Grid search

Set `wrm_in_offset: 0`, `win_rate_model: false`, and `loss_sigmoid_mse: false` in the common JSON:

```powershell
python .\scripts\grid_search.py `
  --settings-file D:\BulletOu-snapshots\settings\bulletou-settings.json `
  --output-folder D:\BulletOu-snapshots\grid-bce `
  --grid loss_bce_with_logits false true
```

False selects the existing WRM squared error; true selects BCE. `grid_summary.csv` records `loss_bce_with_logits`. Do not rank different loss definitions by their numerical loss values. For continued-training comparisons, start each condition from the same checkpoint and teacher position.

## Error-weighted BCE training

`--bce-error-weight-k K` (JSON: `"bce_error_weight_k": K`) increases the relative contribution of samples with larger probability errors. The default **0 preserves ordinary BCE**. K must be finite and nonnegative; nonzero K requires `loss_bce_with_logits: true`. Teacher epsilon is not required.

```text
e_i = abs(p_i - t_i)
w_i = 1 + K * e_i
mean_w = sum(entry_weight_i * w_i) / sum(entry_weight_i)
a_i = stop_gradient(w_i / mean_w)
training_loss = mean(entry_weight_i * a_i * BCE(z_i, t_i))
gradient_i = entry_weight_i * a_i * (p_i - t_i) * nnue2score / in_scaling / batch_size
```

Both numerator and denominator are detached. Existing entry weights define the normalization population: with all ones this is the ordinary batch mean; zero-weight masked samples are excluded. An all-masked batch produces zero loss and gradients. Normalization is per mini-batch, not across all batches accumulated by bpu. Larger K emphasizes large errors, including possible teacher errors.

**Validation loss/qloss remain ordinary BCE, independent of K.** Training loss readback includes the weighting, but validation does not. Thus K trials retain a common validation objective. Accuracy definitions are unchanged; do not pass K to `quantized-test`.

Enable BCE in the common JSON, with `wrm_in_offset: 0`, `win_rate_model: false`, and `loss_sigmoid_mse: false`:

```powershell
python .\scripts\grid_search.py `
  --settings-file D:\BulletOu-snapshots\settings\bulletou-settings.json `
  --output-folder D:\BulletOu-snapshots\grid-bce-error-weight `
  --grid bce_error_weight_k 0 1 2
```

The aggregate includes `bce_error_weight_k`. K=0 retains the existing BCE kernel path. K>0 adds two GPU kernel launches, reusing existing scalar loss scratch space without additional VRAM allocation or CPU readback. Training throughput impact has not been measured.
