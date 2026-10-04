//! Shared FT product-pair revival. Exact two-pass empirical MAD; bounded CPU storage.
use super::*;

pub struct Calibration {
    pub width: usize,
    pub counts: Vec<usize>,
    pub means: Vec<f64>,
    pub utility: Vec<f64>,
    pub relative: Vec<f64>,
    sums: Vec<f64>,
    deviations: Vec<f64>,
    replay_counts: Vec<usize>,
}
impl Calibration {
    pub fn new(width:usize,groups:usize)->Self {
        Self {width,counts:vec![0;groups],means:vec![0.0;width*groups],
            utility:vec![0.0;width/2*groups],relative:vec![0.0;width/2*groups],
            sums:vec![0.0;width*groups],deviations:vec![0.0;width*groups],replay_counts:vec![0;groups]}
    }
    pub fn add(&mut self,buckets:&[i32],x:&[f32],weights:&[f32],mad:bool)->Result<()> {
        expect_len("FT calibration",buckets.len()*self.width,x.len())?;
        expect_len("FT calibration mask",buckets.len(),weights.len())?;
        if self.width==0 || self.width%2!=0 || x.iter().any(|v|!v.is_finite())
            || weights.iter().any(|v|!v.is_finite() || *v<0.0)
            || buckets.iter().any(|&b|b<0 || b as usize>=self.counts.len()) {
            return Err(CudaCppError::message("invalid FT revival calibration"));
        }
        for (row,&bucket) in buckets.iter().enumerate() {
            if weights[row]==0.0 {continue;}
            let b=bucket as usize;
            if mad {self.replay_counts[b]+=1;} else {self.counts[b]+=1;}
            for j in 0..self.width {
                let i=b*self.width+j;let v=x[row*self.width+j] as f64;
                if mad {self.deviations[i]+=(v-self.means[i]).abs();} else {self.sums[i]+=v;}
            }
        }
        Ok(())
    }
    pub fn finish_means(&mut self) {
        for b in 0..self.counts.len() {for j in 0..self.width {
            let i=b*self.width+j;
            self.means[i]=if self.counts[b]>0 {self.sums[i]/self.counts[b] as f64} else {0.0};
        }}
    }
    pub fn finish(&mut self,w:&[f32],out:usize)->Result<()> {
        expect_len("FT outgoing weights",self.counts.len()*out*self.width,w.len())?;
        if self.replay_counts!=self.counts || w.iter().any(|v|!v.is_finite()) {
            return Err(CudaCppError::message("FT calibration replay changed or non-finite weights"));
        }
        let pairs=self.width/2;
        self.utility.fill(0.0);self.relative.fill(0.0);
        for b in 0..self.counts.len() {
            if self.counts[b]==0 {continue;}
            for j in 0..self.width {
                let norm=(0..out).map(|u|w[(b*out+u)*self.width+j].abs() as f64).sum::<f64>();
                self.utility[b*pairs+j%pairs]+=self.deviations[b*self.width+j]/self.counts[b] as f64*norm;
            }
            let max=self.utility[b*pairs..(b+1)*pairs].iter().copied().fold(0.0,f64::max);
            if max>0.0 {for i in b*pairs..(b+1)*pairs {self.relative[i]=self.utility[i]/max;}}
        }
        if self.utility.iter().chain(&self.relative).any(|v|!v.is_finite()) {
            return Err(CudaCppError::message("non-finite FT contribution"));
        }
        Ok(())
    }
    pub fn selected(&self,threshold:f64)->Result<Vec<usize>> {
        if !threshold.is_finite() || threshold<=0.0 || threshold>1.0 {
            return Err(CudaCppError::message("FT contribution threshold must be in (0,1]"));
        }
        // FT is shared: never infer irrelevance from an unmeasured/under-sampled bucket.
        if self.counts.is_empty() || self.counts.iter().any(|&n|n<1024) {return Ok(vec![]);}
        let pairs=self.width/2;
        Ok((0..pairs).filter(|&i|(0..self.counts.len()).all(|b|self.relative[b*pairs+i]<threshold)).collect())
    }
}
fn reset(s:&mut RangerParamStateReadback,i:usize,v:f32) {
    s.momentum[i]=0.0;s.velocity[i]=0.0;s.slow_params[i]=v;
}
impl SfnnTrainStepRunner {
    pub fn revive_ft(&mut self,ctx:&Context,c:&Calibration,ids:&[usize],base_inputs:usize)->Result<()> {
        let s=self.shape;let pairs=s.ft_size/2;
        if self.pending_gradient_batches!=0 || self.batch_norm.is_some() || self.bn_qat.is_some()
            || self.residual_count_gates_enabled || s.has_compact_l1()
            || self.weights.l2fw.is_some() || self.weights.l3fw.is_some()
            || c.width!=s.ft_size || c.counts.len()!=s.num_stacks || base_inputs==0 || base_inputs>s.input_size
            || ids.iter().any(|&i|i>=pairs) || c.counts.iter().any(|&n|n<1024) {
            return Err(CudaCppError::message("FT revival requires calibrated non-BN dense SFNN, L1 none/shared/axis, no residual count gates or pending gradients"));
        }
        if ids.is_empty() {return Ok(());}
        let common=revive_common::Common::read(self,ctx)?;
        // Reinitialize real feature rows. Reset virtual factorizer columns to zero.
        let mut w=self.weights.l0w.download(ctx)?;
        let mut state=self.optimizer_states.l0w.download(ctx)?;
        let bound=(6.0/(base_inputs+s.ft_size) as f32).sqrt().max(1.0/127.0);
        let mut rng=self.revival_rng;
        for &id in ids {for col in [id,id+pairs] {
            for row in 0..s.input_size {
                let v=if row<base_inputs {(rng.signed_uniform()*bound*127.0).round()/127.0} else {0.0};
                let i=row*s.ft_size+col;w[i]=v;reset(&mut state,i,v);
            }
        }}
        self.weights.l0w.upload(ctx,&w)?;
        self.optimizer_states.l0w.upload(ctx,w.len(),RangerParamHostState{momentum:&state.momentum,velocity:&state.velocity,slow_params:&state.slow_params})?;
        drop(w);drop(state);
        let mut b=self.weights.l0b.download(ctx)?;let mut state=self.optimizer_states.l0b.download(ctx)?;
        for &id in ids {for col in [id,id+pairs] {b[col]=0.5;reset(&mut state,col,0.5);}}
        self.weights.l0b.upload(ctx,&b)?;
        self.optimizer_states.l0b.upload(ctx,b.len(),RangerParamHostState{momentum:&state.momentum,velocity:&state.velocity,slow_params:&state.slow_params})?;
        let mut w=self.weights.l1w.download(ctx)?;let mut b=self.weights.l1b.download(ctx)?;
        let mut ws=self.optimizer_states.l1w.download(ctx)?;let mut bs=self.optimizer_states.l1b.download(ctx)?;
        for bucket in 0..s.num_stacks {for u in 0..s.l1_out() {
            let row=bucket*s.l1_out()+u;
            for &id in ids {for col in [id,id+pairs] {
                let i=row*s.ft_size+col;
                let old=w[i]+common.weight(bucket,u,col,false);let old_slow=ws.slow_params[i]+common.weight(bucket,u,col,true);
                let mean=c.means[bucket*s.ft_size+col] as f32;
                b[row]+=old*mean;bs.slow_params[row]+=old_slow*mean;
                let v=if old<0.0 {-1.0/64.0} else {1.0/64.0};
                w[i]=v-common.weight(bucket,u,col,false);reset(&mut ws,i,v-common.weight(bucket,u,col,true));
            }}
            bs.momentum[row]=0.0;bs.velocity[row]=0.0;
        }}
        for (buf,st,v,h) in [(&self.weights.l1w,&self.optimizer_states.l1w,&w,&ws),(&self.weights.l1b,&self.optimizer_states.l1b,&b,&bs)] {
            buf.upload(ctx,v)?;st.upload(ctx,v.len(),RangerParamHostState{momentum:&h.momentum,velocity:&h.velocity,slow_params:&h.slow_params})?;
        }
        self.revival_rng=rng;
        self.forward_workspace.invalidate_l1_qat();self.layer_qat=Default::default();
        Ok(())
    }
    pub fn compensate_ft_revival_mean(&mut self,ctx:&Context,ids:&[usize],c:&Calibration)->Result<()> {
        let s=self.shape;
        if c.width!=s.ft_size || c.counts.len()!=s.num_stacks || c.counts.iter().any(|&n|n==0)
            || ids.iter().any(|&i|i>=s.ft_size/2) || c.means.iter().any(|v|!v.is_finite()) {
            return Err(CudaCppError::message("invalid FT mean compensation"));
        }
        let w=self.weights.l1w.download(ctx)?;let sw=self.optimizer_states.l1w.slow_params.download(ctx)?;
        let common=revive_common::Common::read(self,ctx)?;
        let mut b=self.weights.l1b.download(ctx)?;let mut sb=self.optimizer_states.l1b.slow_params.download(ctx)?;
        for bucket in 0..s.num_stacks {for u in 0..s.l1_out() {
            let row=bucket*s.l1_out()+u;
            for &id in ids {for col in [id,id+s.ft_size/2] {
                let i=row*s.ft_size+col;let m=c.means[bucket*s.ft_size+col] as f32;
                b[row]-=(w[i]+common.weight(bucket,u,col,false))*m;
                sb[row]-=(sw[i]+common.weight(bucket,u,col,true))*m;
            }}
        }}
        self.weights.l1b.upload(ctx,&b)?;self.optimizer_states.l1b.slow_params.upload(ctx,&sb)?;
        self.forward_workspace.invalidate_l1_qat();self.layer_qat=Default::default();
        Ok(())
    }
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn gpu_reset_and_mean_compensation_preserve_other_columns() {
        let ctx=Context::new(0).unwrap();let s=crate::tests::tiny_sfnn_shape();
        let mut host=crate::tests::tiny_sfnn_weights(s);
        host.l1fw=None;host.l1fb=None;host.l2fw=None;host.l2fb=None;host.l3fw=None;host.l3fb=None;
        let mut r=SfnnTrainStepRunner::new(&ctx,host,1024,1).unwrap();
        let mut c=Calibration::new(s.ft_size,s.num_stacks);
        c.counts.fill(1024);c.means.fill(0.25);
        let old=r.read_weights(&ctx).unwrap();
        r.revive_ft(&ctx,&c,&[0],s.input_size).unwrap();
        let changed=r.read_weights(&ctx).unwrap();
        for row in 0..s.input_size {for j in 0..s.ft_size {
            if j!=0 && j!=s.ft_size/2 {assert_eq!(changed.l0w[row*s.ft_size+j],old.l0w[row*s.ft_size+j]);}
        }}
        assert_eq!(changed.l0b[0],0.5);assert_eq!(changed.l0b[s.ft_size/2],0.5);
        for row in 0..s.num_stacks*s.l1_out() {
            let expected=old.l1b[row]+0.25*(old.l1w[row*s.ft_size]+old.l1w[row*s.ft_size+s.ft_size/2]);
            assert!((changed.l1b[row]-expected).abs()<1e-6);
        }
        c.means.fill(0.5);r.compensate_ft_revival_mean(&ctx,&[0],&c).unwrap();
        let after=r.read_weights(&ctx).unwrap();
        for row in 0..s.num_stacks*s.l1_out() {
            let expected=changed.l1b[row]-0.5*(changed.l1w[row*s.ft_size]+changed.l1w[row*s.ft_size+s.ft_size/2]);
            assert!((after.l1b[row]-expected).abs()<1e-6);
        }
        assert_eq!(old.l2w,after.l2w);assert_eq!(old.l3w,after.l3w);
        let snapshot=r.snapshot_device(&ctx).unwrap();
        r.revive_ft(&ctx,&c,&[],s.input_size).unwrap();
        assert_eq!(r.revival_rng,after.revival_rng);
        r.begin_revival_epoch();
        r.revive_ft(&ctx,&c,&[0],s.input_size).unwrap();
        let second=r.read_weights(&ctx).unwrap();
        assert_ne!(after.l0w,second.l0w);
        r.copy_state_from_device(&ctx,&snapshot).unwrap();
        r.begin_revival_epoch();
        r.revive_ft(&ctx,&c,&[0],s.input_size).unwrap();
        assert_eq!(second,r.read_weights(&ctx).unwrap());
    }
    #[test] fn pair_mad_and_all_bucket_guard() {
        let mut c=Calibration::new(4,2);
        for mad in [false,true] {
            for b in 0..2 {for n in 0..1024 {c.add(&[b],&[0.0,(n%2) as f32,0.25,(n%2) as f32],&[1.0],mad).unwrap();}}
            if !mad {c.finish_means();}
        }
        c.finish(&[1.0;8],1).unwrap();
        assert_eq!(c.selected(0.01).unwrap(),vec![0]);
        c.relative[2]=0.01;assert!(c.selected(0.01).unwrap().is_empty());
        c.relative[2]=0.0;c.counts[1]=1023;assert!(c.selected(0.01).unwrap().is_empty());
        assert!(c.selected(0.0).is_err());
    }
    #[test] fn gpu_shared_and_virtual_columns() {
        let ctx=Context::new(0).unwrap();let s=crate::tests::tiny_sfnn_shape();
        let mut host=crate::tests::tiny_sfnn_weights(s);
        host.l2fw=None;host.l2fb=None;host.l3fw=None;host.l3fb=None;
        assert!(host.l1fw.is_some());
        let mut r=SfnnTrainStepRunner::new(&ctx,host,1024,1).unwrap();
        r.factorizer_alpha.shared=0.7;
        let mut c=Calibration::new(s.ft_size,s.num_stacks);c.counts.fill(1024);c.means.fill(0.25);
        let old=r.read_weights(&ctx).unwrap();
        r.revive_ft(&ctx,&c,&[0],s.input_size-1).unwrap();
        let new=r.read_weights(&ctx).unwrap();
        assert_eq!(old.l1fw,new.l1fw);assert_eq!(old.l1fb,new.l1fb);
        for col in [0,s.ft_size/2] {
            assert_eq!(new.l0w[(s.input_size-1)*s.ft_size+col],0.0);
            for bucket in 0..s.num_stacks {for u in 0..s.l1_out() {
                let eff=new.l1w[(bucket*s.l1_out()+u)*s.ft_size+col]+0.7*new.l1fw.as_ref().unwrap()[col*s.l1_out()+u];
                assert!((eff.abs()-1.0/64.0).abs()<1e-6);
            }}
        }
        let state=r.read_optimizer_states(&ctx).unwrap();
        assert_eq!(state.l0w.slow_params,new.l0w);
        assert!(state.l0w.momentum.iter().all(|&v|v==0.0));
    }
}
