//! Read-only L1 shared/axis terms for revival. Never reset shared optimizer state.
use super::*;

pub(crate) struct Common {
    shape: SfnnForwardShape,
    shared_alpha: f32,
    w: Vec<f32>, b: Vec<f32>, sw: Vec<f32>, sb: Vec<f32>,
    aw: Vec<f32>, ab: Vec<f32>, saw: Vec<f32>, sab: Vec<f32>,
    axes: Vec<Vec<(usize,f32)>>,
}

#[cfg(test)] mod tests {
    use super::*;
    #[test]
    fn gpu_k3k3_terms_match_quantized_proxy() {
        let ctx=Context::new(0).unwrap();
        let mut s=crate::tests::tiny_sfnn_shape();s.num_stacks=9;s.factorizer_king_axis_dim=3;
        let mut host=crate::tests::tiny_sfnn_weights(crate::tests::tiny_sfnn_shape());
        let w1=vec![0.0;s.num_stacks*s.l1_out()*s.ft_size];let b1=vec![0.0;s.num_stacks*s.l1_out()];
        let w2=vec![0.0;s.num_stacks*s.l2_size*s.l2_in()];let b2=vec![0.0;s.num_stacks*s.l2_size];
        let w3=vec![0.0;s.num_stacks*s.l2_size];let b3=vec![0.0;s.num_stacks];
        host.shape=s;host.l1w=&w1;host.l1b=&b1;host.l2w=&w2;host.l2b=&b2;host.l3w=&w3;host.l3b=&b3;
        host.l2fw=None;host.l2fb=None;host.l3fw=None;host.l3fb=None;
        let mut r=SfnnTrainStepRunner::new(&ctx,host,4,1).unwrap();
        r.enable_l1_axes_zero(&ctx,s,SfnnFactorizerActive{king_axis:true,..SfnnFactorizerActive::SHARED},SfnnFactorizerAlpha::ONE).unwrap();
        let aw:Vec<f32>=(0..s.factorizer_axis_count()*s.ft_size*s.l1_out()).map(|i|((i%11) as f32-5.0)/64.0).collect();
        let ab:Vec<f32>=(0..s.factorizer_axis_count()*s.l1_out()).map(|i|i as f32/8128.0).collect();
        r.weights.l1axw.as_ref().unwrap().upload(&ctx,&aw).unwrap();r.weights.l1axb.as_ref().unwrap().upload(&ctx,&ab).unwrap();
        let common=Common::read(&r,&ctx).unwrap();
        let proxy=SfnnForwardDeviceWeights::new_dense(&ctx,SfnnForwardShape{factorizer_king_axis_dim:0,..s}).unwrap();
        r.build_quantized_proxy(&ctx,s.input_size,0,&proxy).unwrap();
        let pw=proxy.l1w.download(&ctx).unwrap();let pb=proxy.l1b.download(&ctx).unwrap();
        for b in 0..9 {for u in 0..s.l1_out() {for j in 0..s.ft_size {
            let x=common.weight(b,u,j,false);
            assert!((pw[(b*s.l1_out()+u)*s.ft_size+j]-(x*64.0).round()/64.0).abs()<1e-6);
        }
        assert!((pb[b*s.l1_out()+u]-(common.bias(b,u,false)*8128.0).round()/8128.0).abs()<1e-6);
        }}
    }
    #[test]
    fn gpu_axis_revival_preserves_shared_axes_and_other_bucket() {
        let ctx=Context::new(0).unwrap();
        let mut host=crate::tests::tiny_sfnn_weights(crate::tests::tiny_sfnn_shape());
        host.l2fw=None;host.l2fb=None;host.l3fw=None;host.l3fb=None;
        let mut r=SfnnTrainStepRunner::new(&ctx,host,1024,1).unwrap();
        let mut s=r.shape;s.factorizer_progress_axis=true;
        let active=SfnnFactorizerActive{progress_axis:true,..SfnnFactorizerActive::SHARED};
        r.enable_l1_axes_zero(&ctx,s,active,SfnnFactorizerAlpha{shared:0.6,progress_axis:0.7,..SfnnFactorizerAlpha::ONE}).unwrap();
        let aw=vec![0.2;s.factorizer_axis_count()*s.ft_size*s.l1_out()];
        let ab=vec![0.1;s.factorizer_axis_count()*s.l1_out()];
        r.weights.l1axw.as_ref().unwrap().upload(&ctx,&aw).unwrap();
        r.weights.l1axb.as_ref().unwrap().upload(&ctx,&ab).unwrap();
        r.optimizer_states.l1axw.as_ref().unwrap().slow_params.fill(&ctx,0.3).unwrap();
        r.optimizer_states.l1axb.as_ref().unwrap().slow_params.fill(&ctx,0.4).unwrap();
        r.optimizer_states.l1axw.as_ref().unwrap().momentum.fill(&ctx,0.123).unwrap();
        r.set_factorizer_axis_confidences(&ctx,Some(&[0.8,0.6])).unwrap();
        let common=Common::read(&r,&ctx).unwrap();
        let old=r.read_weights(&ctx).unwrap();
        assert!((common.weight(0,0,0,false)-(0.6*old.l1fw.as_ref().unwrap()[0]+0.7*0.8*0.2)).abs()<1e-6);
        assert!((common.weight(1,0,0,true)-(0.6*r.optimizer_states.l1fw.as_ref().unwrap().slow_params.download(&ctx).unwrap()[0]+0.7*0.6*0.3)).abs()<1e-6);
        // Reset only bucket 0/unit 0. Effective initialized master/slow must match.
        let mut c=l1_revive::Calibration::new(s.ft_size,s.l1_hidden,s.num_stacks);
        c.counts.fill(1024);c.contribution.relative=vec![1.0;s.num_stacks*s.l1_hidden];
        c.contribution.relative[0]=0.0;
        assert_eq!(r.revive_l1_selected(&ctx,&c,true,false).unwrap(),vec![0]);
        let new=r.read_weights(&ctx).unwrap();let os=r.read_optimizer_states(&ctx).unwrap();
        assert_eq!(&old.l1w[s.l1_out()*s.ft_size..],&new.l1w[s.l1_out()*s.ft_size..]);
        assert_eq!(old.l1axw,new.l1axw);assert_eq!(old.l1axb,new.l1axb);assert_eq!(old.l1fw,new.l1fw);
        for j in 0..s.ft_size {assert!((new.l1w[j]+common.weight(0,0,j,false)-os.l1w.slow_params[j]-common.weight(0,0,j,true)).abs()<1e-6);}
        assert!((new.l1b[0]+common.bias(0,0,false)-0.5).abs()<1e-6);
        assert!((os.l1b.slow_params[0]+common.bias(0,0,true)-0.5).abs()<1e-6);
        // FT selected columns must cancel both shared and axis terms.
        let mut ft=ft_revive::Calibration::new(s.ft_size,s.num_stacks);ft.counts.fill(1024);ft.means.fill(0.25);
        r.revive_ft(&ctx,&ft,&[0],s.input_size).unwrap();
        ft.means.fill(0.4);r.compensate_ft_revival_mean(&ctx,&[0],&ft).unwrap();
        let new=r.read_weights(&ctx).unwrap();let os=r.read_optimizer_states(&ctx).unwrap();
        for b in 0..s.num_stacks {for u in 0..s.l1_out() {for j in [0,s.ft_size/2] {
            let i=(b*s.l1_out()+u)*s.ft_size+j;
            let m=new.l1w[i]+common.weight(b,u,j,false);
            let slow=os.l1w.slow_params[i]+common.weight(b,u,j,true);
            assert!((m.abs()-1.0/64.0).abs()<1e-6);assert!((m-slow).abs()<1e-6);
        }}}
        assert_eq!(old.l1axw,new.l1axw);
        assert!(os.l1axw.as_ref().unwrap().momentum.iter().all(|&x|x==0.123));
        // L2 does not alter upstream axis tensors.
        let mut c=l2_revive::Calibration::new(s.l2_in(),s.l2_size,s.num_stacks);
        for _ in 0..1024 {c.add(&[0],&vec![0.0;s.l2_in()],&vec![0.0;s.l2_size]).unwrap();}
        c.contribution.finish(&new.l3w,1,1,0.01).unwrap();
        r.revive_l2_selected(&ctx,&c,true,false).unwrap();
        assert_eq!(r.read_weights(&ctx).unwrap().l1axw,old.l1axw);
    }
}
impl Common {
    pub fn read(r:&SfnnTrainStepRunner,ctx:&Context)->Result<Self> {
        let s=r.shape;let f=r.factorizer;let a=r.factorizer_alpha;
        if f.king_hand_pair || f.king_progress_pair || f.hand_progress_pair {
            return Err(CudaCppError::message("revival supports L1 none/shared/axis, not pair"));
        }
        let (w,b,sw,sb)=if f.shared {
            (r.weights.l1fw.as_ref().unwrap().download(ctx)?,r.weights.l1fb.as_ref().unwrap().download(ctx)?,
             r.optimizer_states.l1fw.as_ref().unwrap().slow_params.download(ctx)?,r.optimizer_states.l1fb.as_ref().unwrap().slow_params.download(ctx)?)
        } else {(vec![0.0;s.ft_size*s.l1_out()],vec![0.0;s.l1_out()],vec![0.0;s.ft_size*s.l1_out()],vec![0.0;s.l1_out()])};
        let (aw,ab,saw,sab)=if f.any_axis() {
            (r.weights.l1axw.as_ref().ok_or_else(||CudaCppError::message("missing L1 axis weights"))?.download(ctx)?,
             r.weights.l1axb.as_ref().ok_or_else(||CudaCppError::message("missing L1 axis bias"))?.download(ctx)?,
             r.optimizer_states.l1axw.as_ref().ok_or_else(||CudaCppError::message("missing L1 axis optimizer"))?.slow_params.download(ctx)?,
             r.optimizer_states.l1axb.as_ref().ok_or_else(||CudaCppError::message("missing L1 axis bias optimizer"))?.slow_params.download(ctx)?)
        } else {(vec![],vec![],vec![],vec![])};
        let confidence=if r.factorizer_axis_confidences_enabled {r.factorizer_axis_confidences.download(ctx)?}
            else {vec![1.0;s.factorizer_axis_count()]};
        let mut axes=vec![vec![];s.num_stacks];
        let progress=s.factorizer_progress_multiplier().max(1);
        for (bucket,terms) in axes.iter_mut().enumerate() {
            let king=(bucket/progress)%s.factorizer_king_bucket_count();
            let hand=(bucket/progress/s.factorizer_king_bucket_count())%s.factorizer_hand_bucket_count();
            let mut add=|id:usize,alpha:f32| {terms.push((id,alpha*confidence[id]));};
            let k=s.factorizer_king_axis_dim;let h=s.factorizer_hand_axis_dim;
            if f.king_axis && k>0 {add(king/k,a.king_axis);add(k+king%k,a.king_axis);}
            if f.hand_axis && h>0 {add(2*k+hand/h,a.hand_axis);add(2*k+h+hand%h,a.hand_axis);}
            if f.progress_axis && s.factorizer_progress_axis && progress>1 {
                add(s.factorizer_base_axis_count()+s.factorizer_pair_count()+bucket%progress,a.progress_axis);
            }
        }
        Ok(Self{shape:s,shared_alpha:if f.shared {a.shared} else {0.0},w,b,sw,sb,aw,ab,saw,sab,axes})
    }
    pub fn weight(&self,bucket:usize,u:usize,j:usize,slow:bool)->f32 {
        let n=self.shape.l1_out();let i=j*n+u;
        let (w,aw)=if slow {(&self.sw,&self.saw)} else {(&self.w,&self.aw)};
        self.shared_alpha*w[i]+self.axes[bucket].iter().map(|&(axis,a)|a*aw[axis*self.shape.ft_size*n+i]).sum::<f32>()
    }
    pub fn bias(&self,bucket:usize,u:usize,slow:bool)->f32 {
        let (b,ab)=if slow {(&self.sb,&self.sab)} else {(&self.b,&self.ab)};
        self.shared_alpha*b[u]+self.axes[bucket].iter().map(|&(axis,a)|a*ab[axis*self.shape.l1_out()+u]).sum::<f32>()
    }
}
