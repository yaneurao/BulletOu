//! Restore-time revival of constant L1 branch pairs (not the skip output).
use super::*;

/// Exact empirical MAD, not an EMA or a variance approximation. CPU-only storage.
pub struct Contribution {
    width: usize,
    values: Vec<Vec<f32>>,
    pub means: Vec<f64>,
    pub utility: Vec<f64>,
    pub relative: Vec<f64>,
    pub threshold: f64,
}
impl Contribution {
    pub fn new(width:usize, groups:usize) -> Self {
        Self {width,values:vec![Vec::new();groups],means:vec![0.0;width*groups],
            utility:Vec::new(),relative:Vec::new(),threshold:0.01}
    }
    pub fn add(&mut self,b:usize,y:&[f32]) { self.values[b].extend_from_slice(y); }
    /// Weights are [bucket, downstream unit, branch]; branches are square then normal for L1.
    pub fn finish(&mut self,w:&[f32],out:usize,branches:usize,threshold:f64) -> Result<()> {
        if !threshold.is_finite() || threshold<=0.0 || threshold>1.0 {
            return Err(CudaCppError::message("revive contribution threshold must be finite and in (0, 1]"));
        }
        expect_len("revive outgoing weights",self.values.len()*out*self.width,w.len())?;
        if w.iter().any(|x|!x.is_finite()) || branches==0 || self.width%branches!=0 {
            return Err(CudaCppError::message("invalid contribution weights or branch layout"));
        }
        let units=self.width/branches;
        self.utility=vec![0.0;self.values.len()*units];
        self.relative=self.utility.clone();self.threshold=threshold;
        for (b,ys) in self.values.iter().enumerate() {
            let n=ys.len()/self.width;
            if n==0 {continue;}
            let mean=&mut self.means[b*self.width..(b+1)*self.width];
            mean.fill(0.0);
            for row in ys.chunks_exact(self.width) {for (m,&y) in mean.iter_mut().zip(row) {*m+=y as f64;}}
            for m in mean.iter_mut() {*m/=n as f64;}
            for col in 0..self.width {
                let mad=ys.chunks_exact(self.width).map(|row|(row[col] as f64-mean[col]).abs()).sum::<f64>()/n as f64;
                let norm=(0..out).map(|j|w[(b*out+j)*self.width+col].abs() as f64).sum::<f64>();
                self.utility[b*units+col%units]+=mad*norm;
            }
            let max=self.utility[b*units..(b+1)*units].iter().copied().fold(0.0,f64::max);
            if max>0.0 {for i in b*units..(b+1)*units {self.relative[i]=self.utility[i]/max;}}
        }
        if self.utility.iter().chain(&self.relative).any(|v|!v.is_finite()) {
            return Err(CudaCppError::message("non-finite revival contribution"));
        }
        self.values.iter_mut().for_each(|v|*v=Vec::new());
        Ok(())
    }
    pub fn selected(&self,counts:&[usize],units:usize) -> Vec<usize> {
        (0..self.relative.len()).filter(|&i|counts[i/units]>=1024 && self.relative[i]<self.threshold).collect()
    }
}

pub struct Calibration {
    pub input: usize,
    pub width: usize,
    pub counts: Vec<usize>,
    pub upper: Vec<usize>,
    pub lower: Vec<usize>,
    upper_threshold: f64,
    zero_threshold: f64,
    sums: Vec<f64>,
    branch_sums: Vec<f64>,
    pub contribution: Contribution,
}

impl Calibration {
    pub fn new(input: usize, width: usize, groups: usize) -> Self {
        Self { input, width, counts: vec![0; groups], upper: vec![0; groups*width],
            lower: vec![0; groups*width], upper_threshold: 0.99, zero_threshold: 0.99,
            sums: vec![0.0; groups*input], branch_sums: vec![0.0; groups*2*width],
            contribution:Contribution::new(2*width,groups) }
    }
    pub fn set_thresholds(&mut self, upper: f64, zero: f64) -> Result<()> {
        if [upper,zero].iter().any(|v| !v.is_finite() || *v<=0.0 || *v>1.0) {
            return Err(CudaCppError::message("revival thresholds must be finite and in (0, 1]"));
        }
        self.upper_threshold=upper; self.zero_threshold=zero;
        Ok(())
    }
    pub fn is_upper(&self, i: usize, enabled: bool) -> bool {
        let n=self.counts[i/self.width];
        enabled && n>=1024 && self.upper[i] as f64/n as f64>=self.upper_threshold
    }
    fn is_zero(&self, i: usize, enabled: bool) -> bool {
        let n=self.counts[i/self.width];
        enabled && n>=1024 && self.lower[i] as f64/n as f64>=self.zero_threshold
    }
    /// x is the combined FT input; y is [squared branches, normal branches].
    pub fn add(&mut self, buckets: &[i32], x: &[f32], y: &[f32]) -> Result<()> {
        self.add_masked(buckets,x,y,None)
    }
    pub fn add_masked(&mut self, buckets: &[i32], x: &[f32], y: &[f32], weights: Option<&[f32]>) -> Result<()> {
        expect_len("L1 revival inputs", buckets.len()*self.input, x.len())?;
        expect_len("L1 revival branches", buckets.len()*2*self.width, y.len())?;
        if let Some(w)=weights {
            expect_len("L1 revival sample weights",buckets.len(),w.len())?;
            if w.iter().any(|v|!v.is_finite() || *v<0.0) {
                return Err(CudaCppError::message("invalid L1 revival sample weight"));
            }
        }
        if x.iter().chain(y).any(|v| !v.is_finite())
            || buckets.iter().any(|&b| b<0 || b as usize>=self.counts.len()) {
            return Err(CudaCppError::message("invalid L1 revival calibration"));
        }
        for (row,&bucket) in buckets.iter().enumerate() {
            if weights.is_some_and(|w|w[row]==0.0) { continue; }
            let b=bucket as usize;
            self.counts[b]+=1;
            self.contribution.add(b,&y[row*2*self.width..(row+1)*2*self.width]);
            for j in 0..self.input { self.sums[b*self.input+j]+=x[row*self.input+j] as f64; }
            for u in 0..self.width {
                let square=y[row*2*self.width+u];
                let normal=y[row*2*self.width+self.width+u];
                self.branch_sums[(b*2)*self.width+u]+=square as f64;
                self.branch_sums[(b*2+1)*self.width+u]+=normal as f64;
                self.upper[b*self.width+u]+=usize::from(square>=1.0 && normal>=1.0);
                self.lower[b*self.width+u]+=usize::from(square<=0.0 && normal<=0.0);
            }
        }
        Ok(())
    }
    pub fn candidates_for(&self, upper: bool, zero: bool) -> Vec<usize> {
        if !self.contribution.relative.is_empty() {
            return if upper || zero {self.contribution.selected(&self.counts,self.width)} else {Vec::new()};
        }
        (0..self.upper.len()).filter(|&i| {
            self.is_upper(i,upper) || self.is_zero(i,zero)
        }).collect()
    }
}

fn reset(s: &mut RangerParamStateReadback, i: usize, slow: f32) {
    s.momentum[i]=0.0; s.velocity[i]=0.0; s.slow_params[i]=slow;
}

impl SfnnTrainStepRunner {
    pub fn l1_revival_done(&self) -> bool { self.l1_revival_flags & 1 != 0 }
    pub fn l1_zero_revival_done(&self) -> bool { self.l1_revival_flags & 2 != 0 }

    pub fn revive_l1_selected(&mut self, ctx: &Context, c: &Calibration, upper: bool, zero: bool) -> Result<Vec<usize>> {
        let upper=upper && !self.l1_revival_done();
        let zero=zero && !self.l1_zero_revival_done();
        if !upper && !zero { return Ok(Vec::new()); }
        if self.pending_gradient_batches!=0 || self.batch_norm.is_some() || self.bn_qat.is_some()
            || self.factorizer.any_axis() || self.residual_count_gates_enabled || self.shape.has_compact_l1()
            || self.weights.l2fw.is_some() || self.weights.l3fw.is_some() {
            return Err(CudaCppError::message("L1 revival requires non-BN dense L1 (none/shared), no axis/count gates or legacy L2/L3 factorizers, and no pending gradients"));
        }
        let s=self.shape;
        if c.input!=s.ft_size || c.width!=s.l1_hidden || c.counts.len()!=s.num_stacks
            || c.counts.iter().sum::<usize>()==0 {
            return Err(CudaCppError::message("empty or incompatible L1 calibration"));
        }
        let ids=c.candidates_for(upper,zero);
        let mut w1=self.weights.l1w.download(ctx)?;
        let mut b1=self.weights.l1b.download(ctx)?;
        let mut w2=self.weights.l2w.download(ctx)?;
        let mut b2=self.weights.l2b.download(ctx)?;
        let mut ws=self.optimizer_states.l1w.download(ctx)?;
        let mut bs=self.optimizer_states.l1b.download(ctx)?;
        let mut os=self.optimizer_states.l2w.download(ctx)?;
        let mut obs=self.optimizer_states.l2b.download(ctx)?;
        let (shared_w,shared_b,slow_w,slow_b)=if self.factorizer.shared {
            (self.weights.l1fw.as_ref().unwrap().download(ctx)?,self.weights.l1fb.as_ref().unwrap().download(ctx)?,
             self.optimizer_states.l1fw.as_ref().unwrap().slow_params.download(ctx)?,
             self.optimizer_states.l1fb.as_ref().unwrap().slow_params.download(ctx)?)
        } else { (vec![0.0;s.ft_size*s.l1_out()],vec![0.0;s.l1_out()],
                  vec![0.0;s.ft_size*s.l1_out()],vec![0.0;s.l1_out()]) };
        let a=self.factorizer_alpha.shared;
        let bound=(6.0/(s.ft_size+s.l1_hidden) as f32).sqrt();
        for &id in &ids {
            let bucket=id/s.l1_hidden; let u=id%s.l1_hidden;
            let out=bucket*s.l1_out()+u;
            let mut rng=0x9e3779b97f4a7c15u64 ^ id as u64;
            let mut mean=0.0f64;
            for j in 0..s.ft_size {
                rng^=rng<<13; rng^=rng>>7; rng^=rng<<17;
                let value=(2.0*((rng>>40) as f32/16777216.0)-1.0)*bound;
                mean+=value as f64*c.sums[bucket*s.ft_size+j]/c.counts[bucket] as f64;
                let i=out*s.ft_size+j; let shared=j*s.l1_out()+u;
                w1[i]=value-a*shared_w[shared];
                reset(&mut ws,i,value-a*slow_w[shared]);
            }
            let bias=0.5-mean as f32;
            b1[out]=bias-a*shared_b[u];
            reset(&mut bs,out,bias-a*slow_b[u]);
            // Quantization-visible connections allow immediate upstream gradients.
            // The caller measures the new branches and compensates their means
            // before training or saving. Treat Lookahead independently.
            for j in 0..s.l2_size {
                let row=bucket*s.l2_size+j;
                for col in [u,s.l1_hidden+u] {
                    let constant=(c.branch_sums[bucket*2*s.l1_hidden+col]/c.counts[bucket] as f64) as f32;
                    let i=row*s.l2_in()+col;
                    b2[row]+=constant*w2[i];
                    obs.slow_params[row]+=constant*os.slow_params[i];
                    let v=if w2[i]<0.0 {-1.0/64.0} else {1.0/64.0};
                    w2[i]=v; reset(&mut os,i,v);
                }
                obs.momentum[row]=0.0; obs.velocity[row]=0.0;
            }
        }
        if w1.iter().chain(&b1).chain(&w2).chain(&b2)
            .chain(&ws.slow_params).chain(&bs.slow_params).chain(&obs.slow_params).any(|v|!v.is_finite()) {
            return Err(CudaCppError::message("non-finite L1 revival result"));
        }
        for (buffer,state,values,host) in [
            (&self.weights.l1w,&self.optimizer_states.l1w,&w1,&ws),
            (&self.weights.l1b,&self.optimizer_states.l1b,&b1,&bs),
            (&self.weights.l2w,&self.optimizer_states.l2w,&w2,&os),
            (&self.weights.l2b,&self.optimizer_states.l2b,&b2,&obs)] {
            buffer.upload(ctx,values)?;
            state.upload(ctx,values.len(),RangerParamHostState {
                momentum:&host.momentum,velocity:&host.velocity,slow_params:&host.slow_params })?;
        }
        self.forward_workspace.invalidate_l1_qat();
        self.layer_qat=Default::default();
        self.l1_revival_flags |= u8::from(upper) | (u8::from(zero)<<1);
        Ok(ids)
    }

    /// Subtract the measured new branch contributions at each L2 preactivation.
    /// Call once after revival, using the same masked teacher sample and quantized proxy.
    pub fn compensate_l1_revival_mean(&mut self, ctx:&Context, ids:&[usize], c:&Calibration) -> Result<()> {
        if ids.is_empty() { return Ok(()); }
        let s=self.shape;
        if c.width!=s.l1_hidden || c.counts.len()!=s.num_stacks
            || ids.iter().any(|&i|i>=s.num_stacks*s.l1_hidden || c.counts[i/s.l1_hidden]==0) {
            return Err(CudaCppError::message("invalid L1 revival mean calibration"));
        }
        let w=self.weights.l2w.download(ctx)?;
        let slow_w=self.optimizer_states.l2w.slow_params.download(ctx)?;
        let mut b=self.weights.l2b.download(ctx)?;
        let mut slow_b=self.optimizer_states.l2b.slow_params.download(ctx)?;
        for &id in ids {
            let bucket=id/s.l1_hidden; let u=id%s.l1_hidden;
            for col in [u,s.l1_hidden+u] {
                let mean=(c.branch_sums[bucket*2*s.l1_hidden+col]/c.counts[bucket] as f64) as f32;
                for j in 0..s.l2_size {
                    let row=bucket*s.l2_size+j;
                    b[row]-=w[row*s.l2_in()+col]*mean;
                    slow_b[row]-=slow_w[row*s.l2_in()+col]*mean;
                }
            }
        }
        if b.iter().chain(&slow_b).any(|v|!v.is_finite()) {
            return Err(CudaCppError::message("non-finite L1 revival mean compensation"));
        }
        self.weights.l2b.upload(ctx,&b)?;
        self.optimizer_states.l2b.slow_params.upload(ctx,&slow_b)?;
        self.layer_qat=Default::default();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn contribution_mad_normalization_and_strict_boundary() {
        let mut c=Contribution::new(4,2);
        for i in 0..1024 {
            // Both branches of unit 0 are constant, but one is 0 and one is 1.
            c.add(0,&[1.0,(i%2) as f32,0.0,(i%2) as f32]);
            c.add(1,&[0.0,1.0,0.0,1.0]);
        }
        c.finish(&[1.0;8],1,2,0.01).unwrap();
        assert_eq!(c.utility,vec![0.0,1.0,0.0,0.0]);
        assert_eq!(c.relative,vec![0.0,1.0,0.0,0.0]);
        assert_eq!(c.selected(&[1024,1024],2),vec![0,2,3]);
        assert_eq!(c.selected(&[1024,1023],2),vec![0]);
        c.relative=vec![0.009999,0.01,0.010001,0.0];
        assert_eq!(c.selected(&[1024,1024],2),vec![0,3]);
    }
    #[test]
    fn contribution_uses_outgoing_weights_and_rejects_invalid_values() {
        let mut c=Contribution::new(3,1);
        c.add(0,&[0.0,0.0,0.25]);c.add(0,&[1.0,1.0,0.75]);
        c.finish(&[0.0,-2.0,1.0],1,1,0.01).unwrap();
        assert_eq!(c.utility,vec![0.0,1.0,0.25]);
        assert_eq!(c.means,vec![0.5;3]);
        for bad in [0.0,-0.1,1.01,f64::NAN,f64::INFINITY] {
            assert!(Contribution::new(1,1).finish(&[1.0],1,1,bad).is_err());
        }
    }
    #[test]
    fn gpu_contribution_mixed_constant_transfers_only_square_mean() {
        let ctx=Context::new(0).unwrap();
        let s=crate::tests::tiny_sfnn_shape();
        let mut host=crate::tests::tiny_sfnn_weights(s);
        host.l1fw=None;host.l1fb=None;host.l2fw=None;host.l2fb=None;host.l3fw=None;host.l3fb=None;
        let mut r=SfnnTrainStepRunner::new(&ctx,host,1024,1).unwrap();
        let mut c=Calibration::new(s.ft_size,s.l1_hidden,s.num_stacks);
        assert_eq!(s.l1_hidden,2);
        for i in 0..1024 {c.add(&[0],&vec![0.0;s.ft_size],&[1.0,(i%2) as f32,0.0,(i%2) as f32]).unwrap();}
        let old=r.read_weights(&ctx).unwrap();
        c.contribution.finish(&old.l2w,s.l2_size,2,0.01).unwrap();
        assert_eq!(c.candidates_for(true,false),vec![0]);
        assert_eq!(r.revive_l1_selected(&ctx,&c,true,false).unwrap(),vec![0]);
        let new=r.read_weights(&ctx).unwrap();
        for j in 0..s.l2_size {
            assert!((new.l2b[j]-old.l2b[j]-old.l2w[j*s.l2_in()]).abs()<1e-6);
        }
        assert_eq!(old.l0w,new.l0w);assert_eq!(old.l3w,new.l3w);
        assert!(r.revive_l1_selected(&ctx,&c,true,false).unwrap().is_empty());
    }
    #[test]
    fn configurable_thresholds_and_zero_classification() {
        let mut c=Calibration::new(1,4,1);
        c.counts[0]=10000;
        c.upper=vec![9900,9899,0,0];
        c.lower=vec![0,0,9900,9899];
        assert_eq!(c.candidates_for(true,true),vec![0,2]);
        assert!(!c.is_upper(2,true));
        assert!(c.candidates_for(false,false).is_empty());
        c.set_thresholds(1.0,1.0).unwrap();
        assert!(c.candidates_for(true,true).is_empty());
        c.set_thresholds(0.98,0.995).unwrap();
        assert_eq!(c.candidates_for(true,true),vec![0,1]);
        for bad in [0.0,-0.1,1.01,f64::NAN,f64::INFINITY] {
            assert!(c.set_thresholds(bad,0.99).is_err());
            assert!(c.set_thresholds(0.99,bad).is_err());
        }
        c.set_thresholds(0.99,0.99).unwrap();
        c.counts[0]=5331; c.upper[0]=5323; c.upper[1]=0; c.lower.fill(0);
        assert_eq!(c.candidates_for(true,false),vec![0]); // reported epoch9 case
    }
    #[test]
    fn gpu_revival_mean_compensation_and_immediate_qat_gradient() {
        let ctx=Context::new(0).unwrap();
        let shape=crate::tests::tiny_sfnn_shape();
        for shared in [false,true] { for qat in [false,true] {
            let mut host=crate::tests::tiny_sfnn_weights(shape);
            host.l2fw=None; host.l2fb=None; host.l3fw=None; host.l3fb=None;
            if !shared { host.l1fw=None; host.l1fb=None; }
            let mut w=host.l1w.to_vec(); let mut b=host.l1b.to_vec();
            let alpha=0.5;
            for k in 0..2 {
                for j in 0..4 { w[k*12+j]=-alpha*host.l1fw.map_or(0.0,|v|v[j*3]); }
                b[k*3]=if k==0 {2.0} else {0.0} - alpha*host.l1fb.map_or(0.0,|v|v[0]);
            }
            host.l1w=&w; host.l1b=&b;
            let mut r=SfnnTrainStepRunner::new(&ctx,host,2048,1).unwrap();
            r.factorizer_alpha.shared=alpha;
            let stm:Vec<i32>=(0..2048).map(|i|(i/2)%4).collect();
            let nstm:Vec<i32>=stm.iter().map(|i|3-i).collect();
            let buckets:Vec<i32>=(0..2048).map(|i|i%2).collect();
            r.device_batch.stm_indices.upload(&ctx,&stm).unwrap();
            r.device_batch.nstm_indices.upload(&ctx,&nstm).unwrap();
            r.device_batch.buckets.upload(&ctx,&buckets).unwrap();
            r.prepare_l1_qat(&ctx,qat).unwrap();
            let forward=|r:&SfnnTrainStepRunner| {
                sfnn_forward_train_device_with_factorizer(&ctx,&r.device_batch,&r.weights,&r.forward_workspace,
                    r.factorizer,r.factorizer_alpha,None,None,None).unwrap();
                r.forward_workspace.output.download(&ctx).unwrap()
            };
            forward(&r);
            let mut c=Calibration::new(4,2,2);
            c.add(&buckets,&r.forward_workspace.combined.download(&ctx).unwrap(),
                &r.forward_workspace.l2_input.download(&ctx).unwrap()).unwrap();
            assert_eq!(c.candidates_for(true,true),vec![0,2]);
            let old=r.read_weights(&ctx).unwrap();
            r.optimizer_states.l1w.momentum.fill(&ctx,0.25).unwrap();
            r.optimizer_states.l2w.slow_params.fill(&ctx,0.75).unwrap();
            r.optimizer_states.l2b.slow_params.fill(&ctx,0.1).unwrap();
            assert_eq!(r.revive_l1_selected(&ctx,&c,true,false).unwrap(),vec![0]);
            assert!(r.l1_revival_done() && !r.l1_zero_revival_done());
            c.lower[2]-=1; // near-zero selection must still compensate as zero, not one
            assert_eq!(r.revive_l1_selected(&ctx,&c,true,true).unwrap(),vec![2]);
            r.prepare_l1_qat(&ctx,qat).unwrap();
            forward(&r);
            let mut means=Calibration::new(4,2,2);
            means.add(&buckets,&r.forward_workspace.combined.download(&ctx).unwrap(),
                &r.forward_workspace.l2_input.download(&ctx).unwrap()).unwrap();
            r.compensate_l1_revival_mean(&ctx,&[0,2],&means).unwrap();
            let new=r.read_weights(&ctx).unwrap();
            assert_eq!(old.l0w,new.l0w); assert_eq!(old.l3w,new.l3w);
            assert_eq!(old.l1fw,new.l1fw); assert_eq!(old.l1fb,new.l1fb);
            for k in 0..2 {
                assert_eq!(&old.l1w[k*12+4..k*12+12],&new.l1w[k*12+4..k*12+12]);
                assert_eq!(&old.l1b[k*3+1..k*3+3],&new.l1b[k*3+1..k*3+3]);
            }
            let slow=r.optimizer_states.l2b.slow_params.download(&ctx).unwrap();
            for bucket in 0..2 {
                for j in 0..shape.l2_size {
                    let row=bucket*shape.l2_size+j;
                    let mut correction=0.0;
                    let mut old_constant=0.0;
                    for col in [0,2] {
                        let idx=row*shape.l2_in()+col;
                        let v=if old.l2w[idx]<0.0 {-1.0/64.0} else {1.0/64.0};
                        assert_eq!(new.l2w[idx],v);
                        correction+=v*(means.branch_sums[bucket*4+col]/means.counts[bucket] as f64) as f32;
                        if bucket==0 { old_constant+=old.l2w[idx]; }
                    }
                    assert!((new.l2b[row]-(old.l2b[row]+old_constant-correction)).abs()<1e-6);
                    let expected_slow=if bucket==0 {1.6} else {0.1};
                    assert!((slow[row]-(expected_slow-correction)).abs()<1e-6);
                }
            }
            let m=r.optimizer_states.l1w.momentum.download(&ctx).unwrap();
            for k in 0..2 { assert_eq!(&m[k*12..k*12+4],&[0.0;4]); assert_eq!(&m[k*12+4..k*12+12],&[0.25;8]); }
            assert_eq!(new.l1_revival_flags,3);
            let snapshot=r.snapshot_device(&ctx).unwrap();
            r.l1_revival_flags=0;
            r.copy_state_from_device(&ctx,&snapshot).unwrap();
            assert_eq!(r.l1_revival_flags,3);
            assert!(r.revive_l1_selected(&ctx,&c,true,true).unwrap().is_empty());
            assert_eq!(new.l1w,r.read_weights(&ctx).unwrap().l1w);
            let targets=vec![0.1;2048]; let entry_weights=vec![1.0;2048];
            for step in 1..=3 {
                r.step_no_readback_with_loss_finalize_update_lr_multipliers_and_dirty_buckets(
                    &ctx,RangerUpdateParams { radam:RAdamUpdateParams {step,learning_rate:0.01,..Default::default()},
                        ..Default::default() },ScalarLossKind::SigmoidPow {pow_exp:2.0},1.0,
                    SfnnTrainStepHostBatch {stm_indices:&stm,nstm_indices:&nstm,buckets:&buckets,
                        targets:&targets,entry_weights:&entry_weights,batch_size:2048,max_active:1},
                    true,true,SfnnLayerLrMultipliers {qat_l1:qat,qat_l2:qat,qat_l3:qat,..Default::default()},None).unwrap();
                if step==1 {
                    let first=r.read_weights(&ctx).unwrap();
                    assert!(first.l1w[..4].iter().zip(&new.l1w[..4]).any(|(a,b)|a!=b),
                        "revived L1 must update on the first step, including with L2 QAT");
                }
            }
            let trained=r.read_weights(&ctx).unwrap();
            assert!(trained.l2w.chunks_exact(4).any(|row|row[0]!=0.0 || row[2]!=0.0));
            assert!(trained.l1w[..4].iter().zip(&new.l1w[..4]).any(|(a,b)|a!=b));
        }}
    }
    #[test]
    fn both_branches_and_coverage_required() {
        let mut c=Calibration::new(1,4,2);
        c.set_thresholds(1.0,1.0).unwrap();
        let y=[1.0,1.0,0.0,0.0, 1.0,0.5,0.0,0.3];
        c.add(&vec![0;1024],&vec![0.2;1024],&y.repeat(1024)).unwrap();
        assert_eq!(c.candidates_for(true,true),vec![0,2]);
        c.add_masked(&[0],&[0.0],&[0.5;8],Some(&[0.0])).unwrap();
        assert_eq!(c.counts[0],1024);
        assert_eq!(c.candidates_for(true,true),vec![0,2]);
        c.add(&[0],&[0.2],&[0.9,1.0,0.0,0.0, 1.0,0.5,0.0,0.3]).unwrap();
        assert_eq!(c.candidates_for(true,false),Vec::<usize>::new());
        assert!(c.add(&[-1],&[0.0],&y).is_err());
        assert!(c.add(&[0],&[f32::NAN],&y).is_err());
        let mut short=Calibration::new(1,4,1);
        short.add(&[0],&[0.0],&y).unwrap();
        assert!(short.candidates_for(true,true).is_empty());
    }
}
