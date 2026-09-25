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
        self.upper.iter().enumerate().filter_map(|(i,&hits)| {
            let n=self.counts[i/self.width]; (n>=1024 && hits==n).then_some(i)
        }).collect()
    }
}

impl SfnnTrainStepRunner {
    pub fn l2_revival_done(&self) -> bool {
        self.batch_norm.as_ref().and_then(|b| b.layers[2].as_ref()).is_some_and(|(l,_)|l.revival_done)
    }
    pub fn revive_l2(&mut self, ctx:&Context, c:&Calibration) -> Result<Vec<usize>> {
        if self.pending_gradient_batches!=0 || !self.bn_qat.as_ref().is_some_and(|q|q.freeze_stats)
            || self.factorizer.any_axis() || self.residual_count_gates_enabled
            || self.weights.l2fw.is_some() || self.weights.l3fw.is_some() {
            return Err(CudaCppError::message("L2 revival requires frozen BN QAT, no L2/L3 factorizer or residual gates, and no pending gradients"));
        }
        if self.l2_revival_done() { return Ok(Vec::new()); }
        let s=self.shape;
        if c.input!=s.l2_in() || c.width!=s.l2_size || c.counts.len()!=s.num_stacks || c.counts.iter().sum::<usize>()==0 {
            return Err(CudaCppError::message("empty or incompatible L2 calibration"));
        }
        let ids=c.candidates(); let n=s.l2_size*s.num_stacks;
        let bn=self.batch_norm.as_ref().and_then(|b|b.layers[2].as_ref())
            .ok_or_else(||CudaCppError::message("L2 revival requires saved L2 BN"))?;
        let mut state=bn.0.read_state(ctx)?;
        if state.config.epsilon>=1.0 {return Err(CudaCppError::message("L2 revival requires BN epsilon < 1"));}
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
            ob[g]+=out[i]-v*mean_y as f32; obs.slow_params[g]+=os.slow_params[i]-v*mean_y as f32;
            out[i]=v;os.slow_params[i]=v;os.momentum[i]=0.0;os.velocity[i]=0.0;
            b[i]=0.0;bs.slow_params[i]=0.0;bs.momentum[i]=0.0;bs.velocity[i]=0.0;
            state.affine[i]=1.0;state.affine[n+i]=beta;
            state.running[i]=0.0;state.running[n+i]=1.0-state.config.epsilon;state.running[2*n+i]=1.0;
            for j in [i,n+i] {state.optimizer.momentum[j]=0.0;state.optimizer.velocity[j]=0.0;state.optimizer.slow_params[j]=state.affine[j];}
        }
        state.revival_done=true;state.validate()?;
        // Validate all host results before modifying device state.
        if w.iter().chain(&b).chain(&out).chain(&ob).any(|x|!x.is_finite()) {return Err(CudaCppError::message("non-finite L2 revival result"));}
        self.weights.l2w.upload(ctx,&w)?;self.weights.l2b.upload(ctx,&b)?;
        self.weights.l3w.upload(ctx,&out)?;self.weights.l3b.upload(ctx,&ob)?;
        for (dst,src) in [(&self.optimizer_states.l2w,&ws),(&self.optimizer_states.l2b,&bs),(&self.optimizer_states.l3w,&os),(&self.optimizer_states.l3b,&obs)] {
            dst.momentum.upload(ctx,&src.momentum)?;dst.velocity.upload(ctx,&src.velocity)?;dst.slow_params.upload(ctx,&src.slow_params)?;
        }
        let layer=&mut self.batch_norm.as_mut().unwrap().layers[2].as_mut().unwrap().0;
        *layer=batch_norm::Layer::from_state(ctx,&state)?;
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn candidates_require_coverage_and_all_upper() {
        let mut c=Calibration::new(1,3,2);
        c.add(&vec![0;1024],&vec![0.5;1024],&[1.0,0.0,0.5].repeat(1024)).unwrap();
        assert_eq!(c.candidates(),vec![0]);assert_eq!(c.lower[1],1024);
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
        let mut y=vec![0.5;1024*shape.l2_size];for row in 0..1024 {y[row*shape.l2_size]=1.0;}
        cal.add(&vec![0;1024],&vec![0.2;1024*shape.l2_in()],&y).unwrap();
        assert_eq!(r.revive_l2(&ctx,&cal).unwrap(),vec![0]);
        assert_eq!(r.weights.l0w.download(&ctx).unwrap(),ft);
        let new=r.weights.l2w.download(&ctx).unwrap();assert_eq!(&new[shape.l2_in()..],&old[shape.l2_in()..]);
        assert_eq!(r.weights.l3w.download(&ctx).unwrap()[0].abs(),1.0/64.0);
        let state=r.read_batch_norm_state(&ctx).unwrap().0[2].clone().unwrap();
        assert!(state.revival_done);assert_eq!(batch_norm::State::decode(&state.encode().unwrap()).unwrap(),state);
        assert!(r.revive_l2(&ctx,&cal).unwrap().is_empty());assert_eq!(r.weights.l2w.download(&ctx).unwrap(),new);
    }
}
