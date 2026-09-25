// Project using the same explicitly rounded scale as inference folding.
__device__ float bn_project_weight(float w, float r) {
    const float folded=__fmul_rn(r,w);
    if(folded>=-2.0f && folded<=127.0f/64.0f)return w;
    float v=__fdiv_rn(fminf(fmaxf(folded,-2.0f),127.0f/64.0f),r);
    // Division rounding must not leave the projected value outside the bound.
    if(__fmul_rn(r,v) < -2.0f || __fmul_rn(r,v) > 127.0f/64.0f)v=nextafterf(v,0.0f);
    return v;
}
__global__ void bn_project_l2(float* w,float* slow,size_t input,size_t channels,BnConfig bn) {
    size_t j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=input*channels)return;
    size_t ch=j/input;
    float r=__fdiv_rn(bn.params[ch],__fsqrt_rn(__fadd_rn(bn.running[channels+ch],bn.epsilon)));
    // Zero gamma means zero effective weights; no division or reset is needed.
    if(r==0.0f || !isfinite(r))return;
    w[j]=bn_project_weight(w[j],r);
    slow[j]=bn_project_weight(slow[j],r);
}
extern "C" int bulletou_bn_qat_project_l2(BulletOuCudaCppContext* ctx,
    BulletOuCudaCppF32Buffer* w,BulletOuCudaCppF32Buffer* slow,size_t input,size_t channels) {
    if(!ctx || !input || !channels || !ctx->bn[2].params ||
        ctx->bn[2].width*ctx->bn[2].groups!=channels ||
        validate_buffer(ctx,w,input*channels,"BN L2 projection weights") ||
        validate_buffer(ctx,slow,input*channels,"BN L2 projection slow"))return -1;
    bn_project_l2<<<static_cast<unsigned>((input*channels+255)/256),256,0,ctx->stream>>>(
        w->ptr,slow->ptr,input,channels,ctx->bn[2]);
    return check_kernel_launch("BN L2 effective weight projection");
}

// Fold QAT. Identity STE through both rounding and clipping.
// The raw parameter tensors are never quantized in place.
__global__ void bn_qat_ft(const float* src,float* dst,size_t base,size_t vr,size_t width,float alpha,BnConfig bn,
    bool training=false,const float* bias=nullptr,float* out_bias=nullptr) {
    size_t j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=(base+vr)*width)return;
    size_t f=j/width,u=j%width;
    if(f>=base) {dst[j]=0;return;}
    float v=bn_fold_weight(src[j],bn,0,u);
    if(vr) v=__fadd_rn(v,__fmul_rn(alpha,bn_fold_weight(src[(base+f%vr)*width+u],bn,0,u)));
    v=sfnn_quantize_dequant_clamped(v,127.0f,-32768.0f,32767.0f,true);
    if(training && bn.params) {
        float r=bn.params[u]/sqrtf(bn.running[width+u]+bn.epsilon);
        if(bn.running[2*width+u]!=0 && r!=0) v=v/r;
        else {
            v=src[j];
            if(vr)v+=alpha*src[(base+f%vr)*width+u];
        }
        if(f==0)out_bias[u]=bias[u];
    }
    dst[j]=v;
}
// Return quantized folded tensors to pre-BN coordinates. Running scale is a
// detached quantizer calibration, NOT another differentiable BN operation.
// Unknown channels calibrate on their first batch. Gamma==0 uses raw weights
// to avoid division by zero and permit gamma to learn away from zero.
__global__ void bn_qat_unfold_training(const float* w,const float* b,const float* sw,const float* sb,
    float* qw,float* qb,size_t input,size_t output,size_t groups,int layer,size_t vr,float alpha,BnConfig bn) {
    size_t j=blockIdx.x*blockDim.x+threadIdx.x,n=input*output*groups;
    if(j>=n)return;
    size_t ch=layer==0?j%output:j/input,u=ch%output,group=ch/output;
    if(!bn.params || u>=bn.width)return;
    size_t bc=group*bn.width+u,c=bn.width*bn.groups;
    float r=bn.params[bc]/sqrtf(bn.running[c+bc]+bn.epsilon);
    size_t k=layer==0?j/output:j%input;
    bool calibrated=bn.running[2*c+bc]!=0 && r!=0;
    float raw=w[j];
    if(layer==0 && vr)raw+=alpha*w[(input+k%vr)*output+u];
    if(sw)raw+=alpha*sw[k*output+u];
    qw[j]=calibrated?qw[j]/r:raw;
    if(k==0) {
        // Batch centering cancels pre-BN bias. Keep it raw: unscaling rounded
        // folded bias would amplify beta rounding by 1/gamma near gamma==0,
        // destroying variance precision and polluting the running mean.
        qb[ch]=b[ch]+(sb?alpha*sb[u]:0);
    }
}
extern "C" int bulletou_bn_qat_unfold_training(BulletOuCudaCppContext* ctx,
    BulletOuCudaCppF32Buffer* w,BulletOuCudaCppF32Buffer* b,BulletOuCudaCppF32Buffer* sw,BulletOuCudaCppF32Buffer* sb,
    BulletOuCudaCppF32Buffer* qw,BulletOuCudaCppF32Buffer* qb,size_t input,size_t output,size_t groups,int layer,size_t vr,float alpha) {
    if(!ctx || !input || !output || !groups || layer<0 || layer>2 || (vr && layer!=0))return -1;
    size_t n=input*output*groups;
    if(validate_buffer(ctx,w,n,"BN QAT unfold raw") || validate_buffer(ctx,qw,n,"BN QAT unfold proxy") ||
        validate_buffer(ctx,b,output*groups,"BN QAT unfold bias") || validate_buffer(ctx,qb,output*groups,"BN QAT unfold proxy bias"))return -1;
    bn_qat_unfold_training<<<static_cast<unsigned>((n+255)/256),256,0,ctx->stream>>>(w->ptr,b->ptr,sw?sw->ptr:nullptr,sb?sb->ptr:nullptr,
        qw->ptr,qb->ptr,input,output,groups,layer,vr,alpha,ctx->bn[layer]);
    return check_kernel_launch("BN QAT train-mode unfold");
}
// Must run before replacing base gradients with raw-coordinate gradients.
__global__ void bn_qat_ft_virtual(float* g,size_t base,size_t vr,size_t width,float alpha,BnConfig bn) {
    size_t j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=vr*width)return;
    size_t f=j/width,u=j%width; double sum=0;
    for(size_t k=f;k<base;k+=vr)sum+=g[k*width+u];
    float scale=bn.params?bn.params[u]/sqrtf(bn.running[width+u]+bn.epsilon):1.0f;
    g[base*width+j]=alpha*scale*float(sum);
}
// 32 channels x 8 input lanes: coalesced FT reads, modest stacked layers.
__global__ void bn_qat_pullback(const float* w,const float* b,const float* sw,const float* sb,
    float* gw,float* gb,float* gs,float* gsb,size_t input,size_t output,size_t groups,int layer,
    size_t vr,float alpha,BnConfig bn) {
    __shared__ double sums[256];
    size_t ch=blockIdx.x*32+threadIdx.x%32,lane=threadIdx.x/32;
    size_t group=ch/output,u=ch%output;
    bool valid=ch<groups*output, normalized=valid && bn.params && u<bn.width;
    size_t bc=group*bn.width+u,c=bn.width*bn.groups;
    float inv=normalized?1.0f/sqrtf(bn.running[c+bc]+bn.epsilon):1.0f;
    float scale=normalized?bn.params[bc]*inv:1.0f;
    double sum=0;
    if(valid)for(size_t k=lane;k<input;k+=8) {
        size_t j=layer==0?k*output+u:ch*input+k;
        size_t sj=k*output+u;
        float effective=w[j];
        if(layer==0 && vr)effective+=alpha*w[(input+k%vr)*output+u];
        if(sw)effective+=alpha*sw[sj];
        float d=gw[j];sum+=double(d)*effective;
        gw[j]=scale*d;
        if(gs)atomicAdd(gs+sj,alpha*scale*d);
    }
    sums[threadIdx.x]=sum;__syncthreads();
    for(int d=4;d;d/=2) {
        if(lane<size_t(d))sums[threadIdx.x]+=sums[threadIdx.x+d*32];
        __syncthreads();
    }
    if(valid && lane==0) {
        float d=gb[ch],bias=b[ch]+(sb?alpha*sb[u]:0.0f);
        if(normalized) {
            bn.grads[bc]+=float((sums[threadIdx.x]+double(d)*(bias-bn.running[bc]))*inv);
            bn.grads[c+bc]+=d;
        }
        gb[ch]=scale*d;
        if(gsb)atomicAdd(gsb+u,alpha*scale*d);
    }
}
// Split the large FT reduction across input tiles. The old kernel launched only
// width/32 blocks (32 for FT1024), leaving most SMs idle for the whole matrix.
__global__ void bn_qat_ft_pullback_tiles(const float* w,float* gw,double* partial,
    size_t input,size_t output,size_t vr,float alpha,BnConfig bn) {
    __shared__ double sums[256];
    size_t u=blockIdx.x*32+threadIdx.x%32,lane=threadIdx.x/32;
    double sum=0;
    if(u<output) {
        float inv=1.0f/sqrtf(bn.running[output+u]+bn.epsilon);
        float scale=bn.params[u]*inv;
        for(size_t k=blockIdx.y*1024+lane;k<input && k<(blockIdx.y+1)*1024;k+=8) {
            size_t j=k*output+u;float effective=w[j];
            if(vr)effective+=alpha*w[(input+k%vr)*output+u];
            float d=gw[j];sum+=double(d)*effective;gw[j]=scale*d;
        }
    }
    sums[threadIdx.x]=sum;__syncthreads();
    for(int d=4;d;d/=2) {if(lane<size_t(d))sums[threadIdx.x]+=sums[threadIdx.x+d*32];__syncthreads();}
    if(u<output && lane==0)partial[blockIdx.y*output+u]=sums[threadIdx.x];
}
__global__ void bn_qat_ft_pullback_finish(const double* partial,const float* b,float* gb,
    size_t tiles,size_t output,BnConfig bn) {
    size_t u=blockIdx.x*blockDim.x+threadIdx.x;if(u>=output)return;
    double sum=0;for(size_t p=0;p<tiles;++p)sum+=partial[p*output+u];
    float inv=1.0f/sqrtf(bn.running[output+u]+bn.epsilon),d=gb[u];
    bn.grads[u]+=float((sum+double(d)*(b[u]-bn.running[u]))*inv);
    bn.grads[output+u]+=d;gb[u]=bn.params[u]*inv*d;
}
extern "C" int bulletou_bn_qat_ft(BulletOuCudaCppContext* ctx,BulletOuCudaCppF32Buffer* src,
    BulletOuCudaCppF32Buffer* dst,size_t base,size_t vr,size_t width,float alpha) {
    if(!ctx || !base || !width || validate_buffer(ctx,src,(base+vr)*width,"BN QAT FT source") ||
        validate_buffer(ctx,dst,(base+vr)*width,"BN QAT FT proxy"))return -1;
    bn_qat_ft<<<static_cast<unsigned>(((base+vr)*width+255)/256),256,0,ctx->stream>>>(src->ptr,dst->ptr,base,vr,width,alpha,ctx->bn[0]);
    return check_kernel_launch("BN QAT FT quantization");
}
extern "C" int bulletou_bn_qat_ft_training(BulletOuCudaCppContext* ctx,BulletOuCudaCppF32Buffer* src,
    BulletOuCudaCppF32Buffer* dst,BulletOuCudaCppF32Buffer* bias,BulletOuCudaCppF32Buffer* out_bias,
    size_t base,size_t vr,size_t width,float alpha) {
    if(!ctx || !base || !width || validate_buffer(ctx,src,(base+vr)*width,"BN QAT FT source") ||
        validate_buffer(ctx,dst,(base+vr)*width,"BN QAT FT proxy") ||
        validate_buffer(ctx,bias,width,"BN QAT FT bias") || validate_buffer(ctx,out_bias,width,"BN QAT FT proxy bias"))return -1;
    bn_qat_ft<<<static_cast<unsigned>(((base+vr)*width+255)/256),256,0,ctx->stream>>>(src->ptr,dst->ptr,base,vr,width,alpha,ctx->bn[0],true,bias->ptr,out_bias->ptr);
    return check_kernel_launch("BN QAT fused FT quantization/unfold");
}
extern "C" int bulletou_bn_qat_pullback(BulletOuCudaCppContext* ctx,
    BulletOuCudaCppF32Buffer* w,BulletOuCudaCppF32Buffer* b,BulletOuCudaCppF32Buffer* sw,BulletOuCudaCppF32Buffer* sb,
    BulletOuCudaCppF32Buffer* gw,BulletOuCudaCppF32Buffer* gb,BulletOuCudaCppF32Buffer* gs,BulletOuCudaCppF32Buffer* gsb,
    size_t input,size_t output,size_t groups,int layer,size_t vr,float alpha) {
    if(!ctx || !input || !output || !groups || layer<0 || layer>3 || (vr && layer!=0))return -1;
    size_t n=(input+vr)*output*groups;
    if(validate_buffer(ctx,w,n,"BN QAT weights") || validate_buffer(ctx,gw,n,"BN QAT weight gradients") ||
        validate_buffer(ctx,b,output*groups,"BN QAT bias") || validate_buffer(ctx,gb,output*groups,"BN QAT bias gradients"))return -1;
    if(sw && (validate_buffer(ctx,sw,input*output,"BN QAT shared") || validate_buffer(ctx,sb,output,"BN QAT shared bias") ||
        validate_buffer(ctx,gs,input*output,"BN QAT shared gradient") || validate_buffer(ctx,gsb,output,"BN QAT shared bias gradient")))return -1;
    BnConfig bn=layer<3?ctx->bn[layer]:BnConfig{};
    if(vr)bn_qat_ft_virtual<<<static_cast<unsigned>((vr*output+255)/256),256,0,ctx->stream>>>(gw->ptr,input,vr,output,alpha,bn);
    // Train-mode BN has already produced raw-coordinate gradients. With no
    // shared factorizer, the remaining pullback is exactly the identity;
    // scanning the entire FT matrix only rewrites each gradient unchanged.
    // Keep the virtual-row reduction above: it is still required for FT.
    if(!bn.params && !gs && !gsb)return check_kernel_launch("BN QAT identity pullback");
    if(layer==0 && groups==1 && input>1024 && bn.params && !sw && !sb && !gs && !gsb) {
        size_t tiles=(input+1023)/1024;
        if(ensure_f32_scratch(&ctx->bn_qat_partials,&ctx->bn_qat_partials_len,2*tiles*output,"BN QAT partial sums"))return -1;
        auto partial=reinterpret_cast<double*>(ctx->bn_qat_partials);
        bn_qat_ft_pullback_tiles<<<dim3(static_cast<unsigned>((output+31)/32),static_cast<unsigned>(tiles)),256,0,ctx->stream>>>(
            w->ptr,gw->ptr,partial,input,output,vr,alpha,bn);
        bn_qat_ft_pullback_finish<<<static_cast<unsigned>((output+255)/256),256,0,ctx->stream>>>(partial,b->ptr,gb->ptr,tiles,output,bn);
        return check_kernel_launch("BN QAT tiled FT pullback");
    }
    if(gs)cudaMemsetAsync(gs->ptr,0,input*output*sizeof(float),ctx->stream);
    if(gsb)cudaMemsetAsync(gsb->ptr,0,output*sizeof(float),ctx->stream);
    bn_qat_pullback<<<static_cast<unsigned>((groups*output+31)/32),256,0,ctx->stream>>>(w->ptr,b->ptr,sw?sw->ptr:nullptr,sb?sb->ptr:nullptr,
        gw->ptr,gb->ptr,gs?gs->ptr:nullptr,gsb?gsb->ptr:nullptr,input,output,groups,layer,vr,alpha,bn);
    return check_kernel_launch("BN QAT fold STE pullback");
}
