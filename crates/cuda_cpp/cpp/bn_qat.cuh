// Fold QAT. Identity STE through both rounding and clipping.
// The raw parameter tensors are never quantized in place.
__global__ void bn_qat_ft(const float* src,float* dst,size_t base,size_t vr,size_t width,float alpha,BnConfig bn) {
    size_t j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=(base+vr)*width)return;
    size_t f=j/width,u=j%width;
    if(f>=base) {dst[j]=0;return;}
    float v=bn_fold_weight(src[j],bn,0,u);
    if(vr) v=__fadd_rn(v,__fmul_rn(alpha,bn_fold_weight(src[(base+f%vr)*width+u],bn,0,u)));
    dst[j]=sfnn_quantize_dequant_clamped(v,127.0f,-32768.0f,32767.0f,true);
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
extern "C" int bulletou_bn_qat_ft(BulletOuCudaCppContext* ctx,BulletOuCudaCppF32Buffer* src,
    BulletOuCudaCppF32Buffer* dst,size_t base,size_t vr,size_t width,float alpha) {
    if(!ctx || !base || !width || validate_buffer(ctx,src,(base+vr)*width,"BN QAT FT source") ||
        validate_buffer(ctx,dst,(base+vr)*width,"BN QAT FT proxy"))return -1;
    bn_qat_ft<<<static_cast<unsigned>(((base+vr)*width+255)/256),256,0,ctx->stream>>>(src->ptr,dst->ptr,base,vr,width,alpha,ctx->bn[0]);
    return check_kernel_launch("BN QAT FT quantization");
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
    if(gs)cudaMemsetAsync(gs->ptr,0,input*output*sizeof(float),ctx->stream);
    if(gsb)cudaMemsetAsync(gsb->ptr,0,output*sizeof(float),ctx->stream);
    bn_qat_pullback<<<static_cast<unsigned>((groups*output+31)/32),256,0,ctx->stream>>>(w->ptr,b->ptr,sw?sw->ptr:nullptr,sb?sb->ptr:nullptr,
        gw->ptr,gb->ptr,gs?gs->ptr:nullptr,gsb?gsb->ptr:nullptr,input,output,groups,layer,vr,alpha,bn);
    return check_kernel_launch("BN QAT fold STE pullback");
}
