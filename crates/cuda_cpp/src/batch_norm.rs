//! Batch normalization before SFNN activations. Statistics are independent of
//! task-loss weights. FT can share one normalization across its two views;
//! stacked layers use one group per bucket. Empty groups leave running state
//! untouched; singleton groups use running statistics and do not update them.
use super::*;

fn optional_f32_buffer_ptr(b: Option<&F32Buffer>) -> *mut ffi::BulletOuCudaCppF32Buffer {
    b.map_or(std::ptr::null_mut(), F32Buffer::as_ptr)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub epsilon: f32,
    /// EMA weight of the new batch, not the old running estimate.
    pub momentum: f32,
    pub initial_gamma: f32,
    pub initial_beta: f32,
}
impl Default for Config {
    fn default() -> Self {
        Self { epsilon: 1e-5, momentum: 0.1, initial_gamma: 0.25, initial_beta: 0.5 }
    }
}
impl Config {
    pub fn validate(self) -> Result<()> {
        if !self.epsilon.is_finite()
            || self.epsilon <= 0.0
            || !self.momentum.is_finite()
            || self.momentum <= 0.0
            || self.momentum > 1.0
            || !self.initial_gamma.is_finite()
            || !self.initial_beta.is_finite()
        {
            return Err(CudaCppError::message("BN requires finite gamma/beta, epsilon > 0 and 0 < momentum <= 1"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct State {
    /// One-shot L2 revival calibration has already been performed on this lineage.
    pub revival_done: bool,
    pub zero_revival_done: bool,
    pub width: usize,
    pub groups: usize,
    pub config: Config,
    pub affine: Vec<f32>,
    pub running: Vec<f32>,
    pub optimizer: RangerParamStateReadback,
}
impl State {
    /// Change only the EMA update rate when resuming; preserve all learned state.
    fn with_resume_config(&self, config: Config) -> Result<Self> {
        config.validate()?;
        self.validate()?;
        let mut state = self.clone();
        state.config.momentum = config.momentum;
        if state.config != config {
            return Err(CudaCppError::message(
                "BN configuration differs from checkpoint: only momentum may change on resume",
            ));
        }
        Ok(state)
    }
    pub fn validate(&self) -> Result<()> {
        self.config.validate()?;
        let c = channels(self.width, self.groups)?;
        expect_len("BN affine", 2 * c, self.affine.len())?;
        expect_len("BN running statistics", 3 * c, self.running.len())?;
        for v in [&self.optimizer.momentum, &self.optimizer.velocity, &self.optimizer.slow_params] {
            expect_len("BN optimizer", 2 * c, v.len())?;
        }
        if self
            .affine
            .iter()
            .chain(&self.running)
            .chain(&self.optimizer.momentum)
            .chain(&self.optimizer.velocity)
            .chain(&self.optimizer.slow_params)
            .any(|x| !x.is_finite())
            || self.running[c..2 * c].iter().any(|x| *x < 0.0)
            || self.running[2 * c..].iter().any(|x| *x != 0.0 && *x != 1.0)
            || self.optimizer.velocity.iter().any(|x| *x < 0.0)
        {
            return Err(CudaCppError::message("invalid/non-finite BN checkpoint state"));
        }
        Ok(())
    }
    /// Inference y = scale * z + shift; fold this into the preceding affine.
    pub fn inference_affine(&self) -> Result<(Vec<f32>, Vec<f32>)> {
        self.validate()?;
        let c = self.width * self.groups;
        let scale: Vec<_> =
            (0..c).map(|i| self.affine[i] / (self.running[c + i] + self.config.epsilon).sqrt()).collect();
        let shift = (0..c).map(|i| self.affine[c + i] - scale[i] * self.running[i]).collect();
        Ok((scale, shift))
    }
    /// Versioned f32 tensor for the existing named-tensor checkpoint format.
    pub fn encode(&self) -> Result<Vec<f32>> {
        self.validate()?;
        let mut v = vec![
            1.0 + f32::from(self.revival_done) + 2.0 * f32::from(self.zero_revival_done),
            self.width as f32,
            self.groups as f32,
            self.config.epsilon,
            self.config.momentum,
            self.config.initial_gamma,
            self.config.initial_beta,
        ];
        for x in [
            &self.affine,
            &self.running,
            &self.optimizer.momentum,
            &self.optimizer.velocity,
            &self.optimizer.slow_params,
        ] {
            v.extend(x);
        }
        Ok(v)
    }
    pub fn decode(v: &[f32]) -> Result<Self> {
        if v.len() < 7
            || !matches!(v[0], 1.0 | 2.0 | 3.0 | 4.0)
            || v[1] < 1.0
            || v[2] < 1.0
            || !v[1].is_finite()
            || !v[2].is_finite()
            || v[1].fract() != 0.0
            || v[2].fract() != 0.0
        {
            return Err(CudaCppError::message("invalid BN checkpoint header"));
        }
        let (width, groups) = (v[1] as usize, v[2] as usize);
        let c = channels(width, groups)?;
        expect_len("BN checkpoint", 7 + 11 * c, v.len())?;
        let state = Self {
            revival_done: v[0] == 2.0 || v[0] == 4.0,
            zero_revival_done: v[0] == 3.0 || v[0] == 4.0,
            width,
            groups,
            config: Config { epsilon: v[3], momentum: v[4], initial_gamma: v[5], initial_beta: v[6] },
            affine: v[7..7 + 2 * c].to_vec(),
            running: v[7 + 2 * c..7 + 5 * c].to_vec(),
            optimizer: RangerParamStateReadback {
                momentum: v[7 + 5 * c..7 + 7 * c].to_vec(),
                velocity: v[7 + 7 * c..7 + 9 * c].to_vec(),
                slow_params: v[7 + 9 * c..].to_vec(),
            },
        };
        state.validate()?;
        Ok(state)
    }
}
fn channels(width: usize, groups: usize) -> Result<usize> {
    if width == 0 || groups == 0 || width > 65536 || groups > 65536 / width {
        Err(CudaCppError::message("BN requires 1..65536 total channels"))
    } else {
        Ok(width * groups)
    }
}

#[derive(Debug)]
pub struct Layer {
    pub revival_done: bool,
    pub zero_revival_done: bool,
    pub width: usize,
    pub groups: usize,
    pub config: Config,
    pub affine: F32Buffer,
    pub running: F32Buffer,
    pub gradients: F32Buffer,
    pub optimizer: RangerParamState,
}
#[derive(Debug)]
pub struct Workspace {
    rows: usize,
    stride: usize,
    width: usize,
    groups: usize,
    pub normalized: F32Buffer,
    pub second_normalized: Option<F32Buffer>,
    pub stats: F32Buffer,
}
impl Layer {
    pub fn new(ctx: &Context, width: usize, groups: usize, config: Config) -> Result<Self> {
        config.validate()?;
        let c = channels(width, groups)?;
        let mut affine = vec![config.initial_gamma; c];
        affine.extend(vec![config.initial_beta; c]);
        let mut running = vec![0.0; 3 * c];
        running[c..2 * c].fill(1.0);
        Self::from_state(
            ctx,
            &State {
                revival_done: false,
                zero_revival_done: false,
                width,
                groups,
                config,
                affine: affine.clone(),
                running,
                optimizer: RangerParamStateReadback {
                    momentum: vec![0.0; 2 * c],
                    velocity: vec![0.0; 2 * c],
                    slow_params: affine,
                },
            },
        )
    }
    pub fn from_state(ctx: &Context, s: &State) -> Result<Self> {
        s.validate()?;
        let gradients = F32Buffer::new(ctx, s.affine.len())?;
        gradients.fill(ctx, 0.0)?;
        Ok(Self {
            revival_done: s.revival_done,
            zero_revival_done: s.zero_revival_done,
            width: s.width,
            groups: s.groups,
            config: s.config,
            affine: F32Buffer::from_host(ctx, &s.affine)?,
            running: F32Buffer::from_host(ctx, &s.running)?,
            gradients,
            optimizer: RangerParamState::from_host_state(
                ctx,
                s.affine.len(),
                RangerParamHostState {
                    momentum: &s.optimizer.momentum,
                    velocity: &s.optimizer.velocity,
                    slow_params: &s.optimizer.slow_params,
                },
            )?,
        })
    }
    pub fn read_state(&self, ctx: &Context) -> Result<State> {
        Ok(State {
            revival_done: self.revival_done,
            zero_revival_done: self.zero_revival_done,
            width: self.width,
            groups: self.groups,
            config: self.config,
            affine: self.affine.download(ctx)?,
            running: self.running.download(ctx)?,
            optimizer: self.optimizer.download(ctx)?,
        })
    }
    pub fn workspace(&self, ctx: &Context, rows: usize, stride: usize, two_views: bool) -> Result<Workspace> {
        if rows == 0 || stride < self.width || rows.checked_mul(stride).is_none() {
            return Err(CudaCppError::message("invalid BN workspace dimensions"));
        }
        Ok(Workspace {
            rows,
            stride,
            width: self.width,
            groups: self.groups,
            normalized: F32Buffer::new(ctx, rows * stride)?,
            second_normalized: if two_views { Some(F32Buffer::new(ctx, rows * stride)?) } else { None },
            stats: F32Buffer::new(ctx, (4 + 4 * rows.div_ceil(1024)) * self.width * self.groups)?,
        })
    }
    fn validate_workspace(&self, w: &Workspace, second: bool, ids: Option<&I32Buffer>) -> Result<()> {
        if w.width != self.width
            || w.groups != self.groups
            || second != w.second_normalized.is_some()
            || (self.groups > 1 && ids.is_none())
        {
            return Err(CudaCppError::message("BN workspace/view/group mismatch"));
        }
        Ok(())
    }
    pub fn forward(
        &self,
        ctx: &Context,
        w: &Workspace,
        a: &F32Buffer,
        b: Option<&F32Buffer>,
        ids: Option<&I32Buffer>,
        training: bool,
    ) -> Result<()> {
        self.validate_workspace(w, b.is_some(), ids)?;
        check(unsafe {
            bulletou_bn_forward(
                ctx.as_ptr(),
                a.as_ptr(),
                optional_f32_buffer_ptr(b),
                w.normalized.as_ptr(),
                optional_f32_buffer_ptr(w.second_normalized.as_ref()),
                ids.map_or(std::ptr::null_mut(), I32Buffer::as_ptr),
                self.affine.as_ptr(),
                self.running.as_ptr(),
                w.stats.as_ptr(),
                w.rows,
                w.stride,
                self.width,
                self.groups,
                self.config.epsilon,
                self.config.momentum,
                training as i32,
            )
        })
    }
    /// Gradients include loss reduction already. Parameter gradients accumulate
    /// across calls so batches_per_update does not silently change semantics.
    pub fn backward(
        &self,
        ctx: &Context,
        w: &Workspace,
        da: &F32Buffer,
        db: Option<&F32Buffer>,
        ids: Option<&I32Buffer>,
    ) -> Result<()> {
        self.validate_workspace(w, db.is_some(), ids)?;
        check(unsafe {
            bulletou_bn_backward(
                ctx.as_ptr(),
                da.as_ptr(),
                optional_f32_buffer_ptr(db),
                w.normalized.as_ptr(),
                optional_f32_buffer_ptr(w.second_normalized.as_ref()),
                ids.map_or(std::ptr::null_mut(), I32Buffer::as_ptr),
                self.affine.as_ptr(),
                w.stats.as_ptr(),
                self.gradients.as_ptr(),
                w.rows,
                w.stride,
                self.width,
                self.groups,
            )
        })
    }
}
unsafe extern "C" {
    fn bulletou_bn_bind(
        ctx: *mut ffi::BulletOuCudaCppContext,
        layer: i32,
        params: *mut ffi::BulletOuCudaCppF32Buffer,
        running: *mut ffi::BulletOuCudaCppF32Buffer,
        stats: *mut ffi::BulletOuCudaCppF32Buffer,
        gradients: *mut ffi::BulletOuCudaCppF32Buffer,
        xa: *mut ffi::BulletOuCudaCppF32Buffer,
        xb: *mut ffi::BulletOuCudaCppF32Buffer,
        rows: usize,
        stride: usize,
        width: usize,
        groups: usize,
        epsilon: f32,
        momentum: f32,
        training: i32,
    ) -> i32;
    fn bulletou_bn_forward(
        ctx: *mut ffi::BulletOuCudaCppContext,
        a: *mut ffi::BulletOuCudaCppF32Buffer,
        b: *mut ffi::BulletOuCudaCppF32Buffer,
        xa: *mut ffi::BulletOuCudaCppF32Buffer,
        xb: *mut ffi::BulletOuCudaCppF32Buffer,
        ids: *mut ffi::BulletOuCudaCppI32Buffer,
        params: *mut ffi::BulletOuCudaCppF32Buffer,
        running: *mut ffi::BulletOuCudaCppF32Buffer,
        stats: *mut ffi::BulletOuCudaCppF32Buffer,
        rows: usize,
        stride: usize,
        width: usize,
        groups: usize,
        epsilon: f32,
        momentum: f32,
        training: i32,
    ) -> i32;
    fn bulletou_bn_backward(
        ctx: *mut ffi::BulletOuCudaCppContext,
        a: *mut ffi::BulletOuCudaCppF32Buffer,
        b: *mut ffi::BulletOuCudaCppF32Buffer,
        xa: *mut ffi::BulletOuCudaCppF32Buffer,
        xb: *mut ffi::BulletOuCudaCppF32Buffer,
        ids: *mut ffi::BulletOuCudaCppI32Buffer,
        params: *mut ffi::BulletOuCudaCppF32Buffer,
        stats: *mut ffi::BulletOuCudaCppF32Buffer,
        gradients: *mut ffi::BulletOuCudaCppF32Buffer,
        rows: usize,
        stride: usize,
        width: usize,
        groups: usize,
    ) -> i32;
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct NetworkState(pub [Option<State>; 3]);
#[derive(Debug)]
pub struct Network {
    pub layers: [Option<(Layer, Workspace)>; 3],
}
impl Network {
    pub fn new(
        ctx: &Context,
        shape: SfnnForwardShape,
        rows: usize,
        enabled: [bool; 3],
        config: Config,
        saved: &NetworkState,
    ) -> Result<Self> {
        Self::with_layout(ctx, rows, enabled, config, saved, [
            (shape.ft_size, shape.ft_size, 1),
            (shape.l1_hidden, shape.l1_out(), shape.num_stacks),
            (shape.l2_size, shape.l2_size, shape.num_stacks),
        ])
    }
    pub fn new_nnue(ctx: &Context, shape: NnueForwardShape, rows: usize,
        enabled: [bool; 3], config: Config, saved: &NetworkState) -> Result<Self> {
        Self::with_layout(ctx, rows, enabled, config, saved, [
            (shape.l1, shape.l1, 1), (shape.l2, shape.l2, 1), (shape.l3, shape.l3, 1),
        ])
    }
    fn with_layout(ctx: &Context, rows: usize, enabled: [bool; 3], config: Config,
        saved: &NetworkState, layout: [(usize, usize, usize); 3]) -> Result<Self> {
        let mut layers = [None, None, None];
        for i in 0..3 {
            if !enabled[i] {
                if saved.0[i].is_some() {
                    return Err(CudaCppError::message(
                        "cannot disable saved BN without an explicit folded-weight conversion",
                    ));
                }
                continue;
            }
            let (width, stride, groups) = layout[i];
            let l = if let Some(s) = &saved.0[i] {
                if s.width != width || s.groups != groups {
                    return Err(CudaCppError::message("BN checkpoint shape mismatch"));
                }
                let resumed = s.with_resume_config(config)?;
                if s.config.momentum != config.momentum {
                    eprintln!("  BN statistics EMA: layer={i} momentum={} -> {} (running statistics, gamma/beta and optimizer state preserved)",
                        s.config.momentum, config.momentum);
                }
                Layer::from_state(ctx, &resumed)?
            } else {
                Layer::new(ctx, width, groups, config)?
            };
            let w = l.workspace(ctx, rows, stride, i == 0)?;
            layers[i] = Some((l, w));
        }
        Ok(Self { layers })
    }
    pub fn read_state(&self, ctx: &Context) -> Result<NetworkState> {
        let mut s = NetworkState::default();
        for i in 0..3 {
            if let Some((l, _)) = &self.layers[i] {
                s.0[i] = Some(l.read_state(ctx)?);
            }
        }
        Ok(s)
    }
    pub(super) fn bind<'a>(&self, ctx: &'a Context, rows: usize, training: bool) -> Result<Binding<'a>> {
        let guard = Binding(ctx);
        for (i, lw) in self.layers.iter().enumerate() {
            if let Some((l, w)) = lw {
                if rows > w.rows {
                    return Err(CudaCppError::message("BN validation batch exceeds allocated training batch"));
                }
                check(unsafe {
                    bulletou_bn_bind(
                        ctx.as_ptr(),
                        i as i32,
                        l.affine.as_ptr(),
                        l.running.as_ptr(),
                        w.stats.as_ptr(),
                        l.gradients.as_ptr(),
                        w.normalized.as_ptr(),
                        optional_f32_buffer_ptr(w.second_normalized.as_ref()),
                        rows,
                        w.stride,
                        l.width,
                        l.groups,
                        l.config.epsilon,
                        l.config.momentum,
                        training as i32,
                    )
                })?;
            }
        }
        Ok(guard)
    }
    pub(super) fn update(&self, ctx: &Context, p: RangerUpdateParams) -> Result<()> {
        self.update_with_lr_multiplier(ctx, p, 1.0)
    }
    pub(super) fn update_with_lr_multiplier(&self, ctx: &Context, mut p: RangerUpdateParams, multiplier: f32) -> Result<()> {
        // Gamma/beta are affine normalization parameters, not quantized weights.
        p.radam.min_weight = -f32::MAX;
        p.radam.max_weight = f32::MAX;
        p.radam.decay = 0.0;
        for (l, _) in self.layers.iter().flatten() {
            update_param_group_with_lr_multiplier(ctx, p, &l.gradients, &l.affine, &l.optimizer, multiplier)?;
        }
        Ok(())
    }
}
pub(super) struct Binding<'a>(&'a Context);
impl Drop for Binding<'_> {
    fn drop(&mut self) {
        for i in 0..3 {
            unsafe {
                bulletou_bn_bind(
                    self.0.as_ptr(),
                    i,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    0,
                    0,
                    0,
                    0,
                    0.0,
                    0.0,
                    0,
                );
            }
        }
    }
}

impl NnueTrainWeightsReadback {
    /// NNUE affine tensors are input-major, including the sparse FT.
    pub fn fold_batch_norm(&mut self, shape: NnueForwardShape) -> Result<()> {
        for (i, state) in self.batch_norm.0.iter().enumerate() {
            let Some(state) = state else { continue };
            let (w, b, width) = match i {
                0 => (&mut self.l0w, &mut self.l0b, shape.l1),
                1 => (&mut self.l1w, &mut self.l1b, shape.l2),
                _ => (&mut self.l2w, &mut self.l2b, shape.l3),
            };
            if state.width != width || state.groups != 1 {
                return Err(CudaCppError::message("NNUE BN fold shape mismatch"));
            }
            let (scale, shift) = state.inference_affine()?;
            for row in w.chunks_exact_mut(width) {
                for (u, value) in row.iter_mut().enumerate() { *value *= scale[u]; }
            }
            for (u, value) in b.iter_mut().enumerate() { *value = *value * scale[u] + shift[u]; }
        }
        self.batch_norm = NetworkState::default();
        Ok(())
    }
}

impl SfnnTrainWeightsReadback {
    /// Fold inference BN after all active L1 shared terms. Keep zero shared
    /// tensors so existing exporter/shape contracts remain unchanged.
    pub fn fold_batch_norm(&mut self, shape: SfnnForwardShape, shared_alpha: f32) -> Result<()> {
        for (i, s) in self.batch_norm.0.clone().into_iter().enumerate() {
            let Some(s) = s else { continue };
            let (width, groups) = match i {
                0 => (shape.ft_size, 1),
                1 => (shape.l1_hidden, shape.num_stacks),
                _ => (shape.l2_size, shape.num_stacks),
            };
            if s.width != width || s.groups != groups {
                return Err(CudaCppError::message("BN inference fold shape mismatch"));
            }
            let (scale, shift) = s.inference_affine()?;
            if i == 0 {
                for row in self.l0w.chunks_exact_mut(shape.ft_size) {
                    for (u, v) in row.iter_mut().enumerate() {
                        *v *= scale[u];
                    }
                }
                for (u, v) in self.l0b.iter_mut().enumerate() {
                    *v = *v * scale[u] + shift[u];
                }
            } else {
                let (w, b, fw, fb, input, output) = if i == 1 {
                    (&mut self.l1w, &mut self.l1b, &mut self.l1fw, &mut self.l1fb, shape.ft_size, shape.l1_out())
                } else {
                    (&mut self.l2w, &mut self.l2b, &mut self.l2fw, &mut self.l2fb, shape.l2_in(), shape.l2_size)
                };
                for bucket in 0..shape.num_stacks {
                    for u in 0..output {
                        // L1 skip is not normalized but still fold its shared term.
                        let normalized = u < s.width;
                        let (r, t) = if normalized {
                            (scale[bucket * s.width + u], shift[bucket * s.width + u])
                        } else {
                            (1.0, 0.0)
                        };
                        let row = bucket * output + u;
                        b[row] = (b[row] + fb.as_ref().map_or(0.0, |v| shared_alpha * v[u])) * r + t;
                        for k in 0..input {
                            let f = fw
                                .as_ref()
                                .map_or(0.0, |v| shared_alpha * v[if i == 1 { k * output + u } else { u * input + k }]);
                            w[row * input + k] = (w[row * input + k] + f) * r;
                        }
                    }
                }
                if let Some(f) = fw {
                    f.fill(0.0);
                }
                if let Some(f) = fb {
                    f.fill(0.0);
                }
            }
        }
        self.batch_norm = NetworkState::default();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bn_resume_momentum_preserves_state_and_changes_ema() {
        let ctx=Context::new(0).unwrap();
        let mut state=Layer::new(&ctx,1,1,Config::default()).unwrap().read_state(&ctx).unwrap();
        state.running=vec![2.0,3.0,1.0];
        state.affine=vec![0.4,0.6];
        state.optimizer.momentum=vec![0.2,-0.1];
        state.optimizer.velocity=vec![0.3,0.4];
        state.optimizer.slow_params=vec![0.35,0.55];
        let config=Config{momentum:0.01,..state.config};
        let resumed=state.with_resume_config(config).unwrap();
        let mut expected=state.clone();expected.config.momentum=0.01;
        assert_eq!(resumed,expected);
        assert_eq!(resumed.inference_affine().unwrap(),state.inference_affine().unwrap());
        assert!(state.with_resume_config(Config{epsilon:0.001,..config}).is_err());
        assert!(state.with_resume_config(Config{initial_beta:0.1,..config}).is_err());
        for invalid in [0.0,-0.1,1.1,f32::NAN] {
            assert!(state.with_resume_config(Config{momentum:invalid,..config}).is_err());
        }
        let layer=Layer::from_state(&ctx,&resumed).unwrap();
        assert_eq!(layer.read_state(&ctx).unwrap(),expected);
        let workspace=layer.workspace(&ctx,2,1,false).unwrap();
        let x=F32Buffer::from_host(&ctx,&[4.0,8.0]).unwrap();
        layer.forward(&ctx,&workspace,&x,None,None,true).unwrap();
        let after=layer.read_state(&ctx).unwrap();
        close(after.running[0],2.0*0.99+6.0*0.01,1e-5);
        close(after.running[1],3.0*0.99+8.0*0.01,1e-5);
        assert_eq!(after.affine,state.affine);
        assert_eq!(after.optimizer.momentum,state.optimizer.momentum);
        assert_eq!(after.optimizer.velocity,state.optimizer.velocity);
        assert_eq!(State::decode(&after.encode().unwrap()).unwrap(),after);
    }
    #[test]
    fn bn_affine_lr_multiplier_scales_and_freezes_optimizer() {
        let ctx=Context::new(0).unwrap();
        let shape=NnueForwardShape {input_size:4,l1:2,l2:2,l3:1};
        let mut results=Vec::new();
        for m in [1.0,0.1,0.0] {
            let bn=Network::new_nnue(&ctx,shape,2,[true,false,false],Config::default(),&NetworkState::default()).unwrap();
            let l=&bn.layers[0].as_ref().unwrap().0;
            l.gradients.fill(&ctx,0.25).unwrap();
            let before=l.read_state(&ctx).unwrap();
            let p=RangerUpdateParams {radam:RAdamUpdateParams {step:1,learning_rate:0.01,..Default::default()},..Default::default()};
            bn.update_with_lr_multiplier(&ctx,p,m).unwrap();
            let after=l.read_state(&ctx).unwrap();
            if m==0.0 {
                assert_eq!(before,after);
                assert!(l.gradients.download(&ctx).unwrap().iter().all(|&v|v==0.0));
                // Also skip Lookahead: a stale slow copy must not move a frozen affine parameter.
                l.optimizer.slow_params.fill(&ctx,10.0).unwrap();
                let saved=l.read_state(&ctx).unwrap();
                bn.update_with_lr_multiplier(&ctx,RangerUpdateParams {radam:RAdamUpdateParams {step:6,..p.radam},..p},0.0).unwrap();
                assert_eq!(saved,l.read_state(&ctx).unwrap());
            }
            results.push(after.affine.iter().zip(&before.affine).map(|(a,b)|a-b).collect::<Vec<_>>());
        }
        for (full,small) in results[0].iter().zip(&results[1]) { assert!(full.abs()>0.0);close(*small,*full*0.1,1e-7); }
        assert!(SfnnLayerLrMultipliers {bn_affine:-1.0,..Default::default()}.validate().is_err());
    }
    #[test]
    fn nnue_bn_all_layer_masks_fold_and_backward() {
        let ctx = Context::new(0).unwrap();
        let shape = NnueForwardShape { input_size: 4, l1: 2, l2: 2, l3: 1 };
        let host = crate::tests::tiny_nnue_weights(shape);
        let batch = NnueForwardDeviceBatch::from_host(&ctx, NnueForwardHostBatch {
            stm_indices: &[0, 1, 2, 3, 0, 2], nstm_indices: &[3, 2, 1, 0, 2, 1],
            batch_size: 6, max_active: 1,
        }).unwrap();
        for mask in 0..8 {
            let mut runner = NnueTrainStepRunner::new(&ctx, host, 6, 1).unwrap();
            runner.configure_batch_norm(&ctx, [mask & 1 != 0, mask & 2 != 0, mask & 4 != 0],
                Config { initial_gamma: 0.1, ..Default::default() }, &NetworkState::default()).unwrap();
            // Train-mode end-to-end derivative, through every enabled BN and clamp.
            let guard = runner.batch_norm.as_ref().map(|b| b.bind(&ctx, 6, true)).transpose().unwrap();
            nnue_forward_device(&ctx, &batch, &runner.weights, &runner.forward_workspace).unwrap();
            let dy = [0.2, -0.3, 0.1, 0.4, -0.2, 0.5];
            runner.loss_workspace.mean_output_gradients.upload(&ctx, &dy).unwrap();
            nnue_backward_device(&ctx, &batch, &runner.weights, &runner.forward_workspace,
                &runner.loss_workspace, &runner.backward_workspace).unwrap();
            for (weight, grad, values) in [
                (&runner.weights.l0w, &runner.backward_workspace.l0w_gradients, host.l0w),
                (&runner.weights.l1w, &runner.backward_workspace.l1w_gradients, host.l1w),
                (&runner.weights.l2w, &runner.backward_workspace.l2w_gradients, host.l2w),
            ] {
                let expected = grad.download(&ctx).unwrap();
                for j in 0..values.len() {
                    let mut objective = [0.0; 2];
                    for (k, sign) in [-1.0, 1.0].iter().enumerate() {
                        let mut v = values.to_vec(); v[j] += sign * 0.0001;
                        weight.upload(&ctx, &v).unwrap();
                        nnue_forward_device(&ctx, &batch, &runner.weights, &runner.forward_workspace).unwrap();
                        objective[k] = runner.forward_workspace.output.download(&ctx).unwrap().iter()
                            .zip(dy).map(|(a,b)| a*b).sum::<f32>();
                    }
                    weight.upload(&ctx, values).unwrap();
                    close(expected[j], (objective[1]-objective[0])/0.0002, 0.006);
                }
            }
            drop(guard);
            let saved = runner.read_weights(&ctx).unwrap();
            for s in saved.batch_norm.0.iter().flatten() {
                assert_eq!(State::decode(&s.encode().unwrap()).unwrap(), *s);
            }
            let guard = runner.batch_norm.as_ref().map(|b| b.bind(&ctx,6,false)).transpose().unwrap();
            nnue_forward_device(&ctx,&batch,&runner.weights,&runner.forward_workspace).unwrap();
            let expected = runner.forward_workspace.output.download(&ctx).unwrap();
            drop(guard);
            let mut folded = saved;
            folded.fold_batch_norm(shape).unwrap();
            let weights = NnueForwardDeviceWeights::from_host(&ctx, NnueForwardHostWeights {
                shape, l0w: &folded.l0w, l0b: &folded.l0b, l1w: &folded.l1w, l1b: &folded.l1b,
                l2w: &folded.l2w, l2b: &folded.l2b, outw: &folded.outw, outb: &folded.outb,
            }).unwrap();
            nnue_forward_device(&ctx,&batch,&weights,&runner.forward_workspace).unwrap();
            for (a,b) in expected.iter().zip(runner.forward_workspace.output.download(&ctx).unwrap()) {
                close(*a,b,2e-5);
            }
        }
    }
    #[test]
    fn nnue_bn_step_paths_update_and_restore() {
        let ctx = Context::new(0).unwrap();
        let upload = Context::new(0).unwrap();
        let shape = NnueForwardShape { input_size: 4, l1: 2, l2: 2, l3: 1 };
        let host = crate::tests::tiny_nnue_weights(shape);
        let batch = NnueTrainStepHostBatch {
            stm_indices: &[0,1,2,3,0,2], nstm_indices: &[3,2,1,0,2,1],
            targets: &[0.1,0.3,0.7,0.9,0.2,0.8], entry_weights: &[1.0;6], batch_size:6,max_active:1,
        };
        let mut reference: Option<NnueTrainWeightsReadback> = None;
        for path in 0..3 {
            let mut runner = NnueTrainStepRunner::new(&ctx,host,6,1).unwrap();
            runner.warmup(&ctx).unwrap();
            runner.configure_batch_norm(&ctx,[true;3],Config::default(),&NetworkState::default()).unwrap();
            for step in 1..=2 {
                let mut p = RangerUpdateParams::default(); p.radam.step=step;
                match path {
                    0 => runner.step_no_readback(&ctx,p,ScalarLossKind::BceWithLogits,1.0,batch).unwrap(),
                    1 => runner.step_pipelined_no_readback(&ctx,&upload,p,ScalarLossKind::BceWithLogits,1.0,batch).unwrap(),
                    _ => { runner.step_profiled_no_readback(&ctx,p,ScalarLossKind::BceWithLogits,1.0,batch).unwrap(); },
                }
            }
            let result = runner.read_weights(&ctx).unwrap();
            for state in result.batch_norm.0.iter().flatten() {
                assert!(state.optimizer.momentum.iter().any(|v| *v!=0.0));
                assert!(state.running[2*state.width..].iter().all(|v| *v==1.0));
            }
            if let Some(r) = &reference {
                for (a,b) in r.l0w.iter().zip(&result.l0w) {close(*a,*b,1e-6);}
                assert_eq!(r.batch_norm,result.batch_norm);
            } else {reference=Some(result.clone());}
            runner.configure_batch_norm(&ctx,[true;3],Config::default(),&result.batch_norm).unwrap();
            assert_eq!(runner.read_weights(&ctx).unwrap().batch_norm,result.batch_norm);
            assert!(runner.configure_batch_norm(&ctx,[false;3],Config::default(),&result.batch_norm).is_err());
        }
    }
    #[test]
    #[ignore = "explicit serial reference/optimized comparison and GPU timing"]
    fn bn_reference_parity_and_benchmark() {
        let ctx=Context::new(0).unwrap();
        for (rows,width,groups,views) in [(257,13,3,false),(259,32,1,true),(259,64,3,false),(65536,1024,1,true),(65536,8,8,false),(65536,64,8,false)] {
            let mut reference=None;
            for old in [true,false] {
                unsafe extern "C" {fn bulletou_bn_reference_mode(enabled:i32);}
                struct ResetReference;
                impl Drop for ResetReference {fn drop(&mut self) {unsafe {bulletou_bn_reference_mode(0);}}}
                let _reset=ResetReference;
                unsafe {bulletou_bn_reference_mode(old as i32);}
                let l=Layer::new(&ctx,width,groups,Config::default()).unwrap();
                let w=l.workspace(&ctx,rows,width,views).unwrap();
                let x:Vec<_>=(0..rows*width).map(|i|((i*17%997) as f32-498.0)*0.002).collect();
                let dy:Vec<_>=(0..rows*width).map(|i|((i*13%101) as f32-50.0)*0.00001).collect();
                let a=F32Buffer::from_host(&ctx,&x).unwrap();
                let b=views.then(||F32Buffer::from_host(&ctx,&x).unwrap());
                let da=F32Buffer::from_host(&ctx,&dy).unwrap();
                let db=views.then(||F32Buffer::from_host(&ctx,&dy).unwrap());
                let ids=I32Buffer::from_host(&ctx,&(0..rows).map(|i|(i%groups) as i32).collect::<Vec<_>>()).unwrap();
                l.forward(&ctx,&w,&a,b.as_ref(),Some(&ids),true).unwrap();
                l.backward(&ctx,&w,&da,db.as_ref(),Some(&ids)).unwrap();
                if rows<1000 {
                    let values=[a.download(&ctx).unwrap(),da.download(&ctx).unwrap(),l.gradients.download(&ctx).unwrap(),l.read_state(&ctx).unwrap().running];
                    if old {reference=Some(values);} else {
                        for (r,v) in reference.take().unwrap().iter().zip(&values) {
                            for (&r,&v) in r.iter().zip(v) {close(r,v,2e-5);}
                        }
                    }
                }
                ctx.synchronize().unwrap();
                let start=std::time::Instant::now();
                for _ in 0..5 {
                    l.forward(&ctx,&w,&a,b.as_ref(),Some(&ids),true).unwrap();
                    l.backward(&ctx,&w,&da,db.as_ref(),Some(&ids)).unwrap();
                }
                ctx.synchronize().unwrap();
                eprintln!("BN rows={rows} width={width} groups={groups} reference={old} forward+backward={:.3}ms",start.elapsed().as_secs_f64()*200.0);
            }
        }
    }
    fn close(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() < tol, "{a} != {b}");
    }
    fn objective(x: &[f32], dy: &[f32], gamma: f32, beta: f32, eps: f32) -> f32 {
        let n = x.len() as f64;
        let m = x.iter().map(|&v| v as f64).sum::<f64>() / n;
        let v = x.iter().map(|&v| (v as f64 - m).powi(2)).sum::<f64>() / n;
        x.iter()
            .zip(dy)
            .map(|(&x, &g)| ((gamma as f64 * (x as f64 - m) / (v + eps as f64).sqrt() + beta as f64) * g as f64) as f32)
            .sum()
    }
    #[test]
    fn bn_state_rejects_corruption() {
        assert!(State::decode(&[]).is_err());
        assert!(State::decode(&[1.0, f32::NAN, 1.0, 1e-5, 0.1, 1.0, 0.0]).is_err());
        assert!(Config { epsilon: 0.0, ..Default::default() }.validate().is_err());
    }
    #[test]
    fn bn_gpu_backward_matches_finite_differences_two_views() {
        let ctx = Context::new(0).unwrap();
        let config = Config { initial_gamma: 0.4, initial_beta: 0.3, ..Default::default() };
        let l = Layer::new(&ctx, 1, 1, config).unwrap();
        let w = l.workspace(&ctx, 3, 1, true).unwrap();
        let x = vec![-0.7, 0.2, 1.9, 0.1, -1.0, 0.8];
        let g = vec![0.2, -0.6, 0.8, 0.9, -0.3, 0.1];
        let a = F32Buffer::from_host(&ctx, &x[..3]).unwrap();
        let b = F32Buffer::from_host(&ctx, &x[3..]).unwrap();
        let da = F32Buffer::from_host(&ctx, &g[..3]).unwrap();
        let db = F32Buffer::from_host(&ctx, &g[3..]).unwrap();
        l.forward(&ctx, &w, &a, Some(&b), None, true).unwrap();
        l.backward(&ctx, &w, &da, Some(&db), None).unwrap();
        let mut dx = da.download(&ctx).unwrap();
        dx.extend(db.download(&ctx).unwrap());
        let h = 0.001;
        for i in 0..x.len() {
            let mut p = x.clone();
            let mut m = x.clone();
            p[i] += h;
            m[i] -= h;
            let fd =
                (objective(&p, &g, 0.4, 0.3, config.epsilon) - objective(&m, &g, 0.4, 0.3, config.epsilon)) / (2.0 * h);
            close(dx[i], fd, 0.0002);
        }
        let grad = l.gradients.download(&ctx).unwrap();
        close(
            grad[0],
            (objective(&x, &g, 0.4 + h, 0.3, config.epsilon) - objective(&x, &g, 0.4 - h, 0.3, config.epsilon))
                / (2.0 * h),
            0.0002,
        );
        close(grad[1], g.iter().sum(), 0.00001);
        l.backward(
            &ctx,
            &w,
            &F32Buffer::from_host(&ctx, &g[..3]).unwrap(),
            Some(&F32Buffer::from_host(&ctx, &g[3..]).unwrap()),
            None,
        )
        .unwrap();
        let accumulated = l.gradients.download(&ctx).unwrap();
        close(accumulated[0], 2.0 * grad[0], 0.00001);
        close(accumulated[1], 2.0 * grad[1], 0.00001);
    }
    #[test]
    fn bn_gpu_bucket_empty_singleton_and_inference_fold() {
        let ctx = Context::new(0).unwrap();
        let l = Layer::new(&ctx, 1, 3, Default::default()).unwrap();
        // Extra column models an unnormalized L1 skip output.
        let w = l.workspace(&ctx, 3, 2, false).unwrap();
        let x = [1.0, 9.0, 3.0, 8.0, 7.0, 6.0];
        let a = F32Buffer::from_host(&ctx, &x).unwrap();
        let ids = I32Buffer::from_host(&ctx, &[0, 0, 1]).unwrap();
        l.forward(&ctx, &w, &a, None, Some(&ids), true).unwrap();
        let out = a.download(&ctx).unwrap();
        close(out[1], 9.0, 1e-6);
        close(out[3], 8.0, 1e-6);
        let s = l.read_state(&ctx).unwrap();
        assert_eq!(&s.running[..3], &[2.0, 0.0, 0.0]);
        assert_eq!(&s.running[3..6], &[2.0, 1.0, 1.0]);
        assert_eq!(&s.running[6..], &[1.0, 0.0, 0.0]);
        let encoded = s.encode().unwrap();
        let restored = State::decode(&encoded).unwrap();
        assert_eq!(s, restored);
        let l = Layer::from_state(&ctx, &restored).unwrap();
        a.upload(&ctx, &x).unwrap();
        l.forward(&ctx, &w, &a, None, Some(&ids), false).unwrap();
        assert_eq!(s, l.read_state(&ctx).unwrap());
        let (scale, shift) = s.inference_affine().unwrap();
        let out = a.download(&ctx).unwrap();
        for (row, bucket) in [0, 0, 1].into_iter().enumerate() {
            close(out[2 * row], scale[bucket] * x[2 * row] + shift[bucket], 1e-5);
        }
        let mut bad = encoded;
        bad[7 + 2 * 3 + 3] = -1.0;
        assert!(State::decode(&bad).is_err());
    }
    #[test]
    fn bn_gpu_constant_input_is_finite() {
        let ctx = Context::new(0).unwrap();
        let l = Layer::new(&ctx, 2, 1, Default::default()).unwrap();
        let w = l.workspace(&ctx, 4, 2, false).unwrap();
        let a = F32Buffer::from_host(&ctx, &[1000.0; 8]).unwrap();
        l.forward(&ctx, &w, &a, None, None, true).unwrap();
        for v in a.download(&ctx).unwrap() {
            close(v, 0.5, 1e-6);
        }
        let g = F32Buffer::from_host(&ctx, &[1.0; 8]).unwrap();
        l.backward(&ctx, &w, &g, None, None).unwrap();
        for v in g.download(&ctx).unwrap() {
            close(v, 0.0, 1e-6);
        }
    }
    #[test]
    fn bn_sfnn_all_layers_backward_fold_and_resume() {
        let ctx = Context::new(0).unwrap();
        let shape = crate::tests::tiny_sfnn_shape();
        let initial = crate::tests::tiny_sfnn_weights(shape);
        let mut r = SfnnTrainStepRunner::new(&ctx, initial, 4, 1).unwrap();
        r.configure_batch_norm(&ctx, [true; 3], Default::default(), &Default::default()).unwrap();
        let batch = SfnnTrainStepHostBatch {
            stm_indices: &[0, 1, 2, 3],
            nstm_indices: &[3, 2, 1, 0],
            buckets: &[0, 0, 1, 1],
            targets: &[0.1, 0.3, 0.7, 0.9],
            entry_weights: &[1.0; 4],
            batch_size: 4,
            max_active: 1,
        };
        r.step_no_readback_with_loss_finalize_update_and_lr_multipliers(
            &ctx,
            Default::default(),
            ScalarLossKind::SigmoidPow { pow_exp: 2.0 },
            1.0,
            batch,
            true,
            false,
            Default::default(),
        )
        .unwrap();
        let g0 = r.backward_workspace.l0w_gradients.download(&ctx).unwrap();
        let g1 = r.backward_workspace.l1w_gradients.download(&ctx).unwrap();
        let g2 = r.backward_workspace.l2w_gradients.download(&ctx).unwrap();
        let evaluate = || {
            let _bind = r.batch_norm.as_ref().unwrap().bind(&ctx, 4, true).unwrap();
            sfnn_forward_device_with_factorizer_and_alpha(
                &ctx,
                &r.device_batch,
                &r.weights,
                &r.forward_workspace,
                r.factorizer,
                r.factorizer_alpha,
            )
            .unwrap();
            r.forward_workspace
                .output
                .download(&ctx)
                .unwrap()
                .iter()
                .zip(batch.targets)
                .map(|(&y, &t)| {
                    let p = 1.0 / (1.0 + (-y).exp());
                    (p - t).powi(2) / 4.0
                })
                .sum::<f32>()
        };
        // Includes the fused FT normalize/clamp/product path, not just primitive BN.
        let optimized_loss = evaluate();
        let optimized_output = r.forward_workspace.output.download(&ctx).unwrap();
        unsafe extern "C" { fn bulletou_bn_reference_mode(enabled: i32); }
        struct ResetReference;
        impl Drop for ResetReference {
            fn drop(&mut self) { unsafe { bulletou_bn_reference_mode(0); } }
        }
        {
            let _reset = ResetReference;
            unsafe { bulletou_bn_reference_mode(1); }
            close(optimized_loss, evaluate(), 2e-6);
            for (a,b) in optimized_output.iter().zip(r.forward_workspace.output.download(&ctx).unwrap()) {
                close(*a,b,2e-5);
            }
        }
        let h = 0.0005;
        for (w, g) in [(&r.weights.l0w, &g0), (&r.weights.l1w, &g1), (&r.weights.l2w, &g2)] {
            let original = w.download(&ctx).unwrap();
            for i in 0..original.len() {
                let mut x = original.clone();
                x[i] += h;
                w.upload(&ctx, &x).unwrap();
                let plus = evaluate();
                x[i] -= 2.0 * h;
                w.upload(&ctx, &x).unwrap();
                let minus = evaluate();
                w.upload(&ctx, &original).unwrap();
                close(g[i], (plus - minus) / (2.0 * h), 0.00015);
            }
        }
        r.forward_current_weights(&ctx, &r.device_batch, &r.forward_workspace).unwrap();
        let expected = r.forward_workspace.output.download(&ctx).unwrap();
        let state = r.read_batch_norm_state(&ctx).unwrap();
        let mut folded = r.read_weights(&ctx).unwrap();
        folded.fold_batch_norm(shape, r.factorizer_alpha.shared).unwrap();
        let host = SfnnForwardHostWeights {
            l0w: &folded.l0w,
            l0b: &folded.l0b,
            l1w: &folded.l1w,
            l1b: &folded.l1b,
            l1fw: folded.l1fw.as_deref(),
            l1fb: folded.l1fb.as_deref(),
            l2w: &folded.l2w,
            l2b: &folded.l2b,
            l2fw: folded.l2fw.as_deref(),
            l2fb: folded.l2fb.as_deref(),
            ..initial
        };
        let dev = SfnnForwardDeviceWeights::from_host(&ctx, host).unwrap();
        let gpu_proxy=SfnnForwardDeviceWeights::new_dense(&ctx,shape).unwrap();
        let cpu_fold_proxy=SfnnForwardDeviceWeights::new_dense(&ctx,shape).unwrap();
        r.build_quantized_proxy(&ctx,shape.input_size,0,&gpu_proxy).unwrap();
        sfnn_build_quantized_proxy_device(&ctx,shape.input_size,0,&dev,&cpu_fold_proxy,
            r.factorizer,r.factorizer_alpha,None,None).unwrap();
        for (a,b) in [(&gpu_proxy.l0w,&cpu_fold_proxy.l0w),(&gpu_proxy.l0b,&cpu_fold_proxy.l0b),
            (&gpu_proxy.l1w,&cpu_fold_proxy.l1w),(&gpu_proxy.l1b,&cpu_fold_proxy.l1b),
            (&gpu_proxy.l2w,&cpu_fold_proxy.l2w),(&gpu_proxy.l2b,&cpu_fold_proxy.l2b),
            (&gpu_proxy.l3w,&cpu_fold_proxy.l3w),(&gpu_proxy.l3b,&cpu_fold_proxy.l3b)] {
            for (a,b) in a.download(&ctx).unwrap().iter().zip(b.download(&ctx).unwrap()) {
                close(*a,b,2e-6); // Legacy GPU dequantization uses fast-math division.
            }
        }
        let base=shape.input_size-1;
        let virtual_shape=SfnnForwardShape {input_size:base,..shape};
        let gpu_virtual=SfnnForwardDeviceWeights::new_dense(&ctx,virtual_shape).unwrap();
        let cpu_virtual=SfnnForwardDeviceWeights::new_dense(&ctx,virtual_shape).unwrap();
        for alpha in [0.0,0.3,1.0,2.0] {
            let mut alphas=r.factorizer_alpha;alphas.ft=alpha;
            {
                let _bn=r.batch_norm.as_ref().unwrap().bind(&ctx,1,false).unwrap();
                sfnn_build_quantized_proxy_device(&ctx,base,1,&r.weights,&gpu_virtual,
                    r.factorizer,alphas,None,None).unwrap();
            }
            sfnn_build_quantized_proxy_device(&ctx,base,1,&dev,&cpu_virtual,
                r.factorizer,alphas,None,None).unwrap();
            for (a,b) in gpu_virtual.l0w.download(&ctx).unwrap().iter().zip(cpu_virtual.l0w.download(&ctx).unwrap()) {
                close(*a,b,2e-6);
            }
        }
        sfnn_forward_device_with_factorizer_and_alpha(
            &ctx,
            &r.device_batch,
            &dev,
            &r.forward_workspace,
            r.factorizer,
            r.factorizer_alpha,
        )
        .unwrap();
        for (a, b) in expected.iter().zip(r.forward_workspace.output.download(&ctx).unwrap()) {
            close(*a, b, 2e-5);
        }
        r.restore_batch_norm(&ctx, &state).unwrap();
        r.forward_current_weights(&ctx, &r.device_batch, &r.forward_workspace).unwrap();
        for (a, b) in expected.iter().zip(r.forward_workspace.output.download(&ctx).unwrap()) {
            close(*a, b, 1e-6);
        }
    }

    #[test]
    fn bn_wide_ft_and_l1_gradients_match_reference_with_accumulation() {
        let ctx=Context::new(0).unwrap();
        let initial=crate::tests::tiny_sfnn_weights(crate::tests::tiny_sfnn_shape());
        for (hidden,skip,stacks) in [(7,true,2),(8,false,8),(8,true,16)] {
        let shape=SfnnForwardShape {input_size:8,ft_size:128,l1_hidden:hidden,l1_skip:skip,num_stacks:stacks,..initial.shape};
        let values=|n:usize| (0..n).map(|i|((i*17%101) as f32-50.0)*0.003).collect::<Vec<_>>();
        let l0w=values(shape.input_size*shape.ft_size);
        let l0b=vec![0.1;shape.ft_size];
        let l1w=values(shape.l1w_len().unwrap());
        let l1fw=values(shape.ft_size*shape.l1_out());
        let l1b=values(shape.num_stacks*shape.l1_out());
        let l1fb=values(shape.l1_out());
        let l2w=values(shape.num_stacks*shape.l2_size*shape.l1_hidden*2);
        let l2fw=values(shape.l2_size*shape.l1_hidden*2);
        let l2b=values(stacks*shape.l2_size);
        let l3w=values(stacks*shape.l2_size);
        let l3b=values(stacks);
        let weights=SfnnForwardHostWeights {shape,l0w:&l0w,l0b:&l0b,l1w:&l1w,l1fw:Some(&l1fw),
            l1b:&l1b,l1fb:Some(&l1fb),l2w:&l2w,l2fw:Some(&l2fw),l2b:&l2b,l3w:&l3w,l3b:&l3b,..initial};
        let rows=1027;
        let stm=(0..rows).map(|i|(i%4) as i32).collect::<Vec<_>>();
        let nstm=(0..rows).map(|i|((i+1)%4) as i32).collect::<Vec<_>>();
        let buckets=(0..rows).map(|i|((i/3)%(stacks-1)) as i32).collect::<Vec<_>>();
        let targets=(0..rows).map(|i|(i%11) as f32/10.0).collect::<Vec<_>>();
        let entries=vec![1.0;rows];
        let batch=SfnnTrainStepHostBatch {stm_indices:&stm,nstm_indices:&nstm,buckets:&buckets,
            targets:&targets,entry_weights:&entries,batch_size:rows,max_active:1};
        unsafe extern "C" {fn bulletou_bn_reference_mode(enabled:i32);}
        struct Reset;
        impl Drop for Reset {fn drop(&mut self){unsafe{bulletou_bn_reference_mode(0);}}}
        let _reset=Reset;
        let mut reference=None;
        for old in [true,false] {
            unsafe{bulletou_bn_reference_mode(old as i32);}
            let mut r=SfnnTrainStepRunner::new(&ctx,weights,rows,1).unwrap();
            r.configure_batch_norm(&ctx,[true;3],Default::default(),&Default::default()).unwrap();
            for _ in 0..2 {
                r.step_no_readback_with_loss_finalize_update_and_lr_multipliers(&ctx,
                    Default::default(),ScalarLossKind::SigmoidPow{pow_exp:2.0},1.0,batch,true,false,Default::default()).unwrap();
            }
            let b=&r.backward_workspace;
            let result=[b.l0w_gradients.download(&ctx).unwrap(),b.l0b_gradients.download(&ctx).unwrap(),
                b.l1w_gradients.download(&ctx).unwrap(),b.l1b_gradients.download(&ctx).unwrap(),
                b.l1fw_gradients.download(&ctx).unwrap(),b.l1fb_gradients.download(&ctx).unwrap(),
                r.forward_workspace.output.download(&ctx).unwrap()];
            if old {reference=Some(result);} else {
                for (a,b) in reference.take().unwrap().iter().zip(result.iter()) {
                    for (&a,&b) in a.iter().zip(b) {close(a,b,3e-5);}
                }
            }
        }
        }
    }
}
