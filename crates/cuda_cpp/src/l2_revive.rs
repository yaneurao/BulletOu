//! Explicit one-shot revival on restore. Calibration is inference-only.
use super::*;

pub struct Calibration {
    pub input: usize,
    pub width: usize,
    pub counts: Vec<usize>,
    pub upper: Vec<usize>,
    pub lower: Vec<usize>,
    inputs: Vec<Vec<f32>>,
}
impl Calibration {
    pub fn new(input: usize, width: usize, groups: usize) -> Self {
        Self { input, width, counts: vec![0; groups], upper: vec![0; groups*width],
            lower: vec![0; groups*width], inputs: vec![Vec::new(); groups] }
    }
    pub fn add(&mut self, buckets: &[i32], x: &[f32], y: &[f32]) -> Result<()> {
        expect_len("revive inputs",buckets.len()*self.input,x.len())?;
        expect_len("revive outputs",buckets.len()*self.width,y.len())?;
        if x.iter().chain(y).any(|v| !v.is_finite()) || buckets.iter().any(|&b| b<0 || b as usize>=self.counts.len()) {
            return Err(CudaCppError::message("invalid L2 revival calibration"));
        }
        for (row,&b) in buckets.iter().enumerate() {
            let b=b as usize; self.counts[b]+=1;
            self.inputs[b].extend_from_slice(&x[row*self.input..(row+1)*self.input]);
            for u in 0..self.width {
                self.upper[b*self.width+u]+=usize::from(y[row*self.width+u]>=1.0);
                self.lower[b*self.width+u]+=usize::from(y[row*self.width+u]<=0.0);
            }
        }
        Ok(())
    }
    pub fn candidates(&self) -> Vec<usize> {
        self.candidates_for(true, false)
    }
    pub fn candidates_for(&self, upper: bool, zero: bool) -> Vec<usize> {
        self.upper.iter().enumerate().filter_map(|(i,&hits)| {
            let n=self.counts[i/self.width];
            (n>=1024 && ((upper && hits==n) || (zero && self.lower[i]==n))).then_some(i)
        }).collect()
    }
}

impl SfnnTrainStepRunner {
    /// Epoch boundary only. Checkpoint flags prevent duplicate work within an epoch,
    /// but must not suppress a newly requested calibration in a later epoch.
    pub fn begin_revival_epoch(&mut self) {
        self.l1_revival_flags = 0;
        self.l2_revival_flags = 0;
        if let Some((layer, _)) = self.batch_norm.as_mut().and_then(|b| b.layers[2].as_mut()) {
            layer.revival_done = false;
            layer.zero_revival_done = false;
        }
    }
    pub fn l2_revival_done(&self) -> bool {
        self.l2_revival_flags & 1 != 0 || self.batch_norm.as_ref().and_then(|b| b.layers[2].as_ref()).is_some_and(|(l,_)|l.revival_done)
    }
    pub fn l2_zero_revival_done(&self) -> bool {
        self.l2_revival_flags & 2 != 0 || self.batch_norm.as_ref().and_then(|b| b.layers[2].as_ref()).is_some_and(|(l,_)|l.zero_revival_done)
    }
    pub fn revive_l2(&mut self, ctx:&Context, c:&Calibration) -> Result<Vec<usize>> {
        self.revive_l2_selected(ctx,c,true,false)
    }
    pub fn revive_l2_selected(&mut self, ctx:&Context, c:&Calibration, upper:bool, zero:bool) -> Result<Vec<usize>> {
        let upper=upper && !self.l2_revival_done();
        let zero=zero && !self.l2_zero_revival_done();
        if !upper && !zero { return Ok(Vec::new()); }
        if self.pending_gradient_batches!=0
            || self.factorizer.any_axis() || self.residual_count_gates_enabled
            || self.shape.has_compact_l1() || self.weights.l2fw.is_some() || self.weights.l3fw.is_some() {
            return Err(CudaCppError::message("L2 revival requires dense L1, no axes/L2/L3 factorizer or residual gates, and no pending gradients"));
        }
        if self.batch_norm.is_some() && (!self.bn_qat.as_ref().is_some_and(|q|q.freeze_stats)
            || self.batch_norm.as_ref().unwrap().layers[2].is_none()) {
            return Err(CudaCppError::message("with BN enabled, L2 revival requires saved L2 BN and frozen BN QAT"));
        }
        let s=self.shape;
        if c.input!=s.l2_in() || c.width!=s.l2_size || c.counts.len()!=s.num_stacks || c.counts.iter().sum::<usize>()==0 {
            return Err(CudaCppError::message("empty or incompatible L2 calibration"));
        }
        let ids=c.candidates_for(upper,zero); let n=s.l2_size*s.num_stacks;
        let mut state=self.batch_norm.as_ref().and_then(|b|b.layers[2].as_ref())
            .map(|bn|bn.0.read_state(ctx)).transpose()?;
        if state.as_ref().is_some_and(|s|s.config.epsilon>=1.0) {return Err(CudaCppError::message("L2 revival requires BN epsilon < 1"));}
        let mut w=self.weights.l2w.download(ctx)?;let mut b=self.weights.l2b.download(ctx)?;
        let mut out=self.weights.l3w.download(ctx)?;let mut ob=self.weights.l3b.download(ctx)?;
        let mut ws=self.optimizer_states.l2w.download(ctx)?;let mut bs=self.optimizer_states.l2b.download(ctx)?;
        let mut os=self.optimizer_states.l3w.download(ctx)?;let mut obs=self.optimizer_states.l3b.download(ctx)?;
        let mut rng=20260926u64;
        let bound=(6.0/(s.l2_in()+s.l2_size) as f32).sqrt();
        for &i in &ids {
            let g=i/s.l2_size;let start=i*s.l2_in();
            for k in 0..s.l2_in() {
                rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;
                w[start+k]=(2.0*((rng>>40) as f32/16777216.0)-1.0)*bound;
                ws.momentum[start+k]=0.0;ws.velocity[start+k]=0.0;ws.slow_params[start+k]=w[start+k];
            }
            let rows=&c.inputs[g];let count=c.counts[g];
            let mean_dot=rows.chunks_exact(s.l2_in()).map(|x|x.iter().zip(&w[start..start+s.l2_in()])
                .map(|(&x,&w)|x as f64*w as f64).sum::<f64>()).sum::<f64>()/count as f64;
            let beta=0.5-mean_dot as f32;
            let mean_y=rows.chunks_exact(s.l2_in()).map(|x| {
                let v=x.iter().zip(&w[start..start+s.l2_in()]).map(|(&x,&w)|x as f64*((w*64.0).round()/64.0) as f64).sum::<f64>()
                    +((beta*8128.0).round()/8128.0) as f64; v.clamp(0.0,1.0)
            }).sum::<f64>()/count as f64;
            let v=if out[i]<0.0 {-1.0/64.0} else {1.0/64.0};
            let old_activation=if c.lower[i]==c.counts[g] {0.0} else {1.0};
            ob[g]+=old_activation*out[i]-v*mean_y as f32;
            obs.slow_params[g]+=old_activation*os.slow_params[i]-v*mean_y as f32;
            out[i]=v;os.slow_params[i]=v;os.momentum[i]=0.0;os.velocity[i]=0.0;
            // Without BN, the ordinary L2 bias supplies the same effective shift.
            b[i]=if state.is_some() {0.0} else {beta};
            bs.slow_params[i]=b[i];bs.momentum[i]=0.0;bs.velocity[i]=0.0;
            if let Some(state)=state.as_mut() {
                state.affine[i]=1.0;state.affine[n+i]=beta;
                state.running[i]=0.0;state.running[n+i]=1.0-state.config.epsilon;state.running[2*n+i]=1.0;
                for j in [i,n+i] {state.optimizer.momentum[j]=0.0;state.optimizer.velocity[j]=0.0;state.optimizer.slow_params[j]=state.affine[j];}
            }
        }
        if let Some(state)=state.as_mut() {
            state.revival_done|=upper;state.zero_revival_done|=zero;state.validate()?;
        }
        // Validate all host results before modifying device state.
        if w.iter().chain(&b).chain(&out).chain(&ob).any(|x|!x.is_finite()) {return Err(CudaCppError::message("non-finite L2 revival result"));}
        self.weights.l2w.upload(ctx,&w)?;self.weights.l2b.upload(ctx,&b)?;
        self.weights.l3w.upload(ctx,&out)?;self.weights.l3b.upload(ctx,&ob)?;
        for (dst,src) in [(&self.optimizer_states.l2w,&ws),(&self.optimizer_states.l2b,&bs),(&self.optimizer_states.l3w,&os),(&self.optimizer_states.l3b,&obs)] {
            dst.momentum.upload(ctx,&src.momentum)?;dst.velocity.upload(ctx,&src.velocity)?;dst.slow_params.upload(ctx,&src.slow_params)?;
        }
        if let Some(state)=state {
            let layer=&mut self.batch_norm.as_mut().unwrap().layers[2].as_mut().unwrap().0;
            *layer=batch_norm::Layer::from_state(ctx,&state)?;
        }
        self.l2_revival_flags |= u8::from(upper) | (u8::from(zero)<<1);
        self.layer_qat=Default::default();
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn non_bn_upper_zero_and_l1_combination_roundtrip_and_train() {
        let ctx=Context::new(0).unwrap(); let shape=crate::tests::tiny_sfnn_shape();
        for qat in [false,true] {
            let mut host=crate::tests::tiny_sfnn_weights(shape);
            host.l2fw=None;host.l2fb=None;host.l3fw=None;host.l3fb=None;
            let mut w1=host.l1w.to_vec(); let mut b1=host.l1b.to_vec();
            for bucket in 0..2 {
                for j in 0..shape.ft_size { w1[bucket*12+j]=-host.l1fw.unwrap()[j*3]; }
                b1[bucket*3]=(if bucket==0 {2.0} else {0.0})-host.l1fb.unwrap()[0];
            }
            let w2=vec![0.0;16]; let b2=vec![2.0,-1.0,0.3,0.4];
            host.l1w=&w1;host.l1b=&b1;host.l2w=&w2;host.l2b=&b2;
            let mut r=SfnnTrainStepRunner::new(&ctx,host,2048,1).unwrap();
            let stm:Vec<i32>=(0..2048).map(|i|(i/2)%4).collect();
            let nstm:Vec<i32>=stm.iter().map(|i|3-i).collect();
            let buckets:Vec<i32>=(0..2048).map(|i|i%2).collect();
            r.device_batch.stm_indices.upload(&ctx,&stm).unwrap();
            r.device_batch.nstm_indices.upload(&ctx,&nstm).unwrap();
            r.device_batch.buckets.upload(&ctx,&buckets).unwrap();
            let proxy=SfnnForwardDeviceWeights::new_dense(&ctx,shape).unwrap();
            let workspace=SfnnForwardWorkspace::new(&ctx,SfnnForwardWorkspaceLayout::new(shape,2048)).unwrap();
            let forward=|r:&SfnnTrainStepRunner| {
                r.build_quantized_proxy(&ctx,shape.input_size,0,&proxy).unwrap();
                sfnn_forward_device(&ctx,&r.device_batch,&proxy,&workspace).unwrap();
            };
            forward(&r);
            let mut l1=l1_revive::Calibration::new(shape.ft_size,shape.l1_hidden,shape.num_stacks);
            l1.add(&buckets,&workspace.combined.download(&ctx).unwrap(),&workspace.l2_input.download(&ctx).unwrap()).unwrap();
            assert_eq!(r.revive_l1_selected(&ctx,&l1,true,true).unwrap(),vec![0,2]);
            forward(&r);
            let mut c=Calibration::new(shape.l2_in(),shape.l2_size,shape.num_stacks);
            c.add(&buckets,&workspace.l2_input.download(&ctx).unwrap(),&workspace.l2.download(&ctx).unwrap()).unwrap();
            assert_eq!(c.candidates_for(true,true),vec![0,1]);
            let old=r.read_weights(&ctx).unwrap();
            r.optimizer_states.l2w.momentum.fill(&ctx,0.25).unwrap();
            let old_slow_bias=r.optimizer_states.l3b.slow_params.download(&ctx).unwrap();
            r.optimizer_states.l3w.slow_params.fill(&ctx,0.75).unwrap();
            assert_eq!(r.revive_l2_selected(&ctx,&c,true,false).unwrap(),vec![0]);
            assert!(r.l2_revival_done() && !r.l2_zero_revival_done());
            assert_eq!(r.revive_l2_selected(&ctx,&c,true,true).unwrap(),vec![1]);
            let new=r.read_weights(&ctx).unwrap();
            assert_eq!(new.l2_revival_flags,3); assert_eq!(new.l1_revival_flags,3);
            assert_eq!(old.l0w,new.l0w);assert_eq!(old.l1w,new.l1w);assert_eq!(old.l1fw,new.l1fw);
            assert_eq!(&old.l2w[8..],&new.l2w[8..]);assert_eq!(&old.l2b[2..],&new.l2b[2..]);
            assert_eq!(&old.l3w[2..],&new.l3w[2..]);assert_eq!(old.l3b[1],new.l3b[1]);
            assert!(r.batch_norm.is_none());
            let mut correction=0.0f64;
            for i in 0..2 {
                let mean=c.inputs[0].chunks_exact(4).map(|x| {
                    (x.iter().zip(&new.l2w[i*4..i*4+4]).map(|(&x,&w)|x as f64*((w*64.0).round()/64.0) as f64).sum::<f64>()
                        +((new.l2b[i]*8128.0).round()/8128.0) as f64).clamp(0.0,1.0)
                }).sum::<f64>()/c.counts[0] as f64;
                assert_eq!(new.l3w[i].abs(),1.0/64.0);
                correction+=new.l3w[i] as f64*mean;
            }
            assert!((new.l3b[0] as f64-old.l3b[0] as f64-old.l3w[0] as f64+correction).abs()<1e-6);
            let slow_bias=r.optimizer_states.l3b.slow_params.download(&ctx).unwrap();
            assert!((slow_bias[0] as f64-old_slow_bias[0] as f64-0.75+correction).abs()<1e-6);
            let m=r.optimizer_states.l2w.momentum.download(&ctx).unwrap();
            assert_eq!(&m[..8],&[0.0;8]);assert_eq!(&m[8..],&[0.25;8]);
            let snapshot=r.snapshot_device(&ctx).unwrap();r.l2_revival_flags=0;
            r.copy_state_from_device(&ctx,&snapshot).unwrap();
            assert!(r.revive_l2_selected(&ctx,&c,true,true).unwrap().is_empty());
            let targets=vec![0.1;2048];let weights=vec![1.0;2048];
            for step in 1..=3 {
                r.step_no_readback_with_loss_finalize_update_lr_multipliers_and_dirty_buckets(
                    &ctx,RangerUpdateParams {radam:RAdamUpdateParams {step,learning_rate:0.01,..Default::default()},..Default::default()},
                    ScalarLossKind::SigmoidPow {pow_exp:2.0},1.0,
                    SfnnTrainStepHostBatch {stm_indices:&stm,nstm_indices:&nstm,buckets:&buckets,
                        targets:&targets,entry_weights:&weights,batch_size:2048,max_active:1},true,true,
                    SfnnLayerLrMultipliers {qat_ft:qat,qat_l1:qat,qat_l2:qat,qat_l3:qat,..Default::default()},None).unwrap();
            }
            let trained=r.read_weights(&ctx).unwrap();
            assert!(trained.l2w[..8].iter().zip(&new.l2w[..8]).any(|(a,b)|a!=b));
            assert!(trained.l3b.iter().all(|v|v.is_finite()));
        }
    }
    #[test]
    fn candidates_require_coverage_and_all_upper() {
        let mut c=Calibration::new(1,3,2);
        c.add(&vec![0;1024],&vec![0.5;1024],&[1.0,0.0,0.5].repeat(1024)).unwrap();
        assert_eq!(c.candidates(),vec![0]);assert_eq!(c.lower[1],1024);
        assert_eq!(c.candidates_for(false,true),vec![1]);
        assert_eq!(c.candidates_for(true,true),vec![0,1]);
        let mut short=Calibration::new(1,1,1);
        short.add(&[0],&[0.5],&[0.0]).unwrap();
        assert!(short.candidates_for(false,true).is_empty());
        c.add(&[0],&[0.5],&[0.99,0.0,0.5]).unwrap();assert!(c.candidates().is_empty());
        assert!(c.add(&[-1],&[0.0],&[0.0;3]).is_err());
    }
    #[test]
    fn revive_preserves_unselected_state_and_roundtrips_marker() {
        let ctx=Context::new(0).unwrap();let shape=crate::tests::tiny_sfnn_shape();
        let mut r=SfnnTrainStepRunner::new(&ctx,crate::tests::tiny_sfnn_weights(shape),4,1).unwrap();
        r.factorizer=SfnnFactorizerActive::NONE;
        r.weights.l2fw=None;r.weights.l2fb=None;r.weights.l3fw=None;r.weights.l3fb=None;
        r.configure_batch_norm(&ctx,[false,false,true],Default::default(),&Default::default()).unwrap();
        let l=&r.batch_norm.as_ref().unwrap().layers[2].as_ref().unwrap().0;
        let c=shape.num_stacks*shape.l2_size;let mut stat=vec![0.0;3*c];stat[c..].fill(1.0);l.running.upload(&ctx,&stat).unwrap();
        r.configure_bn_qat_mode(&ctx,true,shape.input_size,0,true).unwrap();
        let old=r.weights.l2w.download(&ctx).unwrap();let ft=r.weights.l0w.download(&ctx).unwrap();
        let mut cal=Calibration::new(shape.l2_in(),shape.l2_size,shape.num_stacks);
        let mut y=vec![0.5;1024*shape.l2_size];for row in 0..1024 {y[row*shape.l2_size]=1.0;y[row*shape.l2_size+1]=0.0;}
        cal.add(&vec![0;1024],&vec![0.2;1024*shape.l2_in()],&y).unwrap();
        assert_eq!(r.revive_l2(&ctx,&cal).unwrap(),vec![0]);
        assert_eq!(r.weights.l0w.download(&ctx).unwrap(),ft);
        let new=r.weights.l2w.download(&ctx).unwrap();assert_eq!(&new[shape.l2_in()..],&old[shape.l2_in()..]);
        assert_eq!(r.weights.l3w.download(&ctx).unwrap()[0].abs(),1.0/64.0);
        let state=r.read_batch_norm_state(&ctx).unwrap().0[2].clone().unwrap();
        assert!(state.revival_done);assert_eq!(batch_norm::State::decode(&state.encode().unwrap()).unwrap(),state);
        assert!(r.revive_l2(&ctx,&cal).unwrap().is_empty());assert_eq!(r.weights.l2w.download(&ctx).unwrap(),new);
        assert!(!r.l2_zero_revival_done());
        let bias_before=r.weights.l3b.download(&ctx).unwrap();
        assert_eq!(r.revive_l2_selected(&ctx,&cal,true,true).unwrap(),vec![1]);
        let after=r.weights.l2w.download(&ctx).unwrap();
        assert_eq!(&after[..shape.l2_in()],&new[..shape.l2_in()]);
        assert_eq!(&after[2*shape.l2_in()..],&new[2*shape.l2_in()..]);
        let out=r.weights.l3w.download(&ctx).unwrap();assert_eq!(out[1].abs(),1.0/64.0);
        let zero_state=r.read_batch_norm_state(&ctx).unwrap().0[2].clone().unwrap();
        let beta=zero_state.affine[c+1];
        let activation=(after[shape.l2_in()..2*shape.l2_in()].iter()
            .map(|w|0.2f64*((w*64.0).round()/64.0) as f64).sum::<f64>()
            +((beta*8128.0).round()/8128.0) as f64).clamp(0.0,1.0) as f32;
        let bias_after=r.weights.l3b.download(&ctx).unwrap();
        assert!((bias_after[0]-bias_before[0]+out[1]*activation).abs()<1e-6);
        assert_eq!(&bias_after[1..],&bias_before[1..]);
        assert!(r.l2_revival_done() && r.l2_zero_revival_done());
        assert!(r.revive_l2_selected(&ctx,&cal,true,true).unwrap().is_empty());
        for upper in [false,true] {for zero in [false,true] {
            let mut s=zero_state.clone();s.revival_done=upper;s.zero_revival_done=zero;
            let decoded=batch_norm::State::decode(&s.encode().unwrap()).unwrap();
            assert_eq!(s,decoded);
            let layer=batch_norm::Layer::from_state(&ctx,&decoded).unwrap();
            assert_eq!(layer.read_state(&ctx).unwrap(),decoded);
        }}
    }
}
