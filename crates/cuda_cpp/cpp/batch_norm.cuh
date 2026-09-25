// Included inside the backend namespace after buffer validation helpers.
// One channel/bucket per CTA. Reference-first implementation: no atomics,
// double-precision moments, and exact BN backward (not a straight-through op).
// Layout: params=[gamma,beta], running=[mean,variance,initialized],
// stats=[inverse_std,count_used_for_batch_statistics].
__global__ void bn_ft_activate(float* a,float* b,float* combined,size_t rows,size_t width) {
    size_t j=blockIdx.x*blockDim.x+threadIdx.x,half=width/2;
    if(j>=rows*half)return;
    size_t base=(j/half)*width,u=j%half;
    float a0=crelu(a[base+u]),a1=crelu(a[base+half+u]);
    float b0=crelu(b[base+u]),b1=crelu(b[base+half+u]);
    a[base+u]=a0;a[base+half+u]=a1;b[base+u]=b0;b[base+half+u]=b1;
    combined[base+u]=a0*a1*SFNN_PAIRWISE_SCALE;
    combined[base+half+u]=b0*b1*SFNN_PAIRWISE_SCALE;
}
__global__ void bn_activate(float* a,size_t n) {
    size_t j=blockIdx.x*blockDim.x+threadIdx.x;if(j<n)a[j]=crelu(a[j]);
}
__global__ void bn_bias_sum(const float* a,const float* b,float* gb,size_t rows,size_t width) {
    size_t u=blockIdx.x*blockDim.x+threadIdx.x;if(u>=width)return;
    float s=0;for(size_t i=0;i<rows;++i)s+=a[i*width+u]+b[i*width+u];gb[u]+=s;
}
__global__ void bn_forward_kernel(float* a, float* b, float* xa, float* xb,
    const int* buckets, const float* params, float* running, float* stats,
    size_t rows, size_t stride, size_t width, size_t groups,
    float epsilon, float momentum, bool training) {
    const size_t channel=blockIdx.x, unit=channel%width, group=channel/width;
    const size_t channels=groups*width;
    __shared__ double sums[256], counts[256];
    __shared__ double mean, variance, count;
    double s=0.0,n=0.0;
    for(size_t i=threadIdx.x;i<rows;i+=256) {
        if(groups>1 && buckets[i]!=static_cast<int>(group)) continue;
        s+=double(a[i*stride+unit]); n+=1.0;
        if(b) {s+=double(b[i*stride+unit]); n+=1.0;}
    }
    sums[threadIdx.x]=s; counts[threadIdx.x]=n;
    __syncthreads();
    for(int d=128;d;d/=2) {
        if(threadIdx.x<d) {sums[threadIdx.x]+=sums[threadIdx.x+d]; counts[threadIdx.x]+=counts[threadIdx.x+d];}
        __syncthreads();
    }
    if(threadIdx.x==0) {
        count=counts[0]; mean=count>0?sums[0]/count:0.0;
    }
    __syncthreads();
    s=0.0;
    if(training && count>=2.0) {
        for(size_t i=threadIdx.x;i<rows;i+=256) {
            if(groups>1 && buckets[i]!=static_cast<int>(group)) continue;
            double v=double(a[i*stride+unit])-mean; s+=v*v;
            if(b) {v=double(b[i*stride+unit])-mean; s+=v*v;}
        }
    }
    sums[threadIdx.x]=s;
    __syncthreads();
    for(int d=128;d;d/=2) {
        if(threadIdx.x<d) sums[threadIdx.x]+=sums[threadIdx.x+d];
        __syncthreads();
    }
    if(threadIdx.x==0) {
        if(training && count>=2.0) {
            variance=sums[0]/count;
            const float m=running[2*channels+channel]>0?momentum:1.0f;
            running[channel]=(1-m)*running[channel]+m*float(mean);
            // Unbiased running variance; biased batch variance in forward.
            running[channels+channel]=(1-m)*running[channels+channel]+m*float(sums[0]/(count-1.0));
            running[2*channels+channel]=1.0f;
            stats[channels+channel]=float(count);
        } else {
            mean=running[channel]; variance=running[channels+channel];
            stats[channels+channel]=0.0f;
        }
        stats[channel]=1.0f/sqrtf(float(variance)+epsilon);
    }
    __syncthreads();
    const float inv=stats[channel], gamma=params[channel], beta=params[channels+channel];
    for(size_t i=threadIdx.x;i<rows;i+=256) {
        if(groups>1 && buckets[i]!=static_cast<int>(group)) continue;
        size_t j=i*stride+unit;
        float x=float(double(a[j])-mean)*inv; xa[j]=x; a[j]=gamma*x+beta;
        if(b) {x=float(double(b[j])-mean)*inv; xb[j]=x; b[j]=gamma*x+beta;}
    }
}

__global__ void bn_backward_kernel(float* da,float* db,const float* xa,const float* xb,
    const int* buckets,const float* params,const float* stats,float* grads,
    size_t rows,size_t stride,size_t width,size_t groups) {
    const size_t ch=blockIdx.x,u=ch%width,g=ch/width,c=groups*width;
    __shared__ double sums[256],products[256];
    double s=0.0,p=0.0;
    for(size_t i=threadIdx.x;i<rows;i+=256) {
        if(groups>1 && buckets[i]!=static_cast<int>(g)) continue;
        const size_t j=i*stride+u;
        s+=da[j]; p+=double(da[j])*xa[j];
        if(db) {s+=db[j]; p+=double(db[j])*xb[j];}
    }
    sums[threadIdx.x]=s; products[threadIdx.x]=p;
    __syncthreads();
    for(int d=128;d;d/=2) {
        if(threadIdx.x<d) {sums[threadIdx.x]+=sums[threadIdx.x+d];products[threadIdx.x]+=products[threadIdx.x+d];}
        __syncthreads();
    }
    if(threadIdx.x==0) {grads[ch]+=float(products[0]);grads[c+ch]+=float(sums[0]);}
    const float n=stats[c+ch],scale=params[ch]*stats[ch];
    const float sm=n>0?float(sums[0])/n:0.0f,pm=n>0?float(products[0])/n:0.0f;
    for(size_t i=threadIdx.x;i<rows;i+=256) {
        if(groups>1 && buckets[i]!=static_cast<int>(g)) continue;
        const size_t j=i*stride+u;
        da[j]=scale*(da[j]-sm-xa[j]*pm);
        if(db) db[j]=scale*(db[j]-sm-xb[j]*pm);
    }
}


// Default reductions coalesce 32 units for widths divisible by 32, otherwise 8.
// Scratch after stats[2*C]: double mean[C], partial_sum[T*C], partial_aux[T*C].
// T = ceil(rows/1024). No atomics, no changes to BN/EMA definitions.
template<int Mode,int Units=8>
__global__ void bn_partial(const float* a,const float* b,const float* xa,const float* xb,
    const int* ids,float* stats,size_t rows,size_t stride,size_t width,size_t groups) {
    size_t lane=threadIdx.x%Units,ch=blockIdx.x*Units+lane,c=width*groups,u=ch%width,g=ch/width;
    size_t tiles=(rows+1023)/1024,begin=blockIdx.y*1024,end=min(rows,begin+1024);
    double* mean=reinterpret_cast<double*>(stats+2*c);
    double* sums=mean+c;double* aux=sums+tiles*c;
    __shared__ double s[256],t[256];
    double p=0,q=0;
    for(size_t i=begin+threadIdx.x/Units;i<end;i+=256/Units) {
        if(ch>=c || (groups>1 && ids[i]!=static_cast<int>(g)))continue;
        size_t j=i*stride+u;
        if constexpr(Mode==0) {p+=a[j];q+=1;if(b){p+=b[j];q+=1;}}
        if constexpr(Mode==1) {double d=double(a[j])-mean[ch];p+=d*d;q+=1;
            if(b){d=double(b[j])-mean[ch];p+=d*d;q+=1;}}
        if constexpr(Mode==2) {p+=a[j];q+=double(a[j])*xa[j];if(b){p+=b[j];q+=double(b[j])*xb[j];}}
    }
    s[threadIdx.x]=p;t[threadIdx.x]=q;__syncthreads();
    for(int d=128;d>=Units;d/=2){if(threadIdx.x<d){s[threadIdx.x]+=s[threadIdx.x+d];t[threadIdx.x]+=t[threadIdx.x+d];}__syncthreads();}
    if(threadIdx.x<Units && ch<c){sums[blockIdx.y*c+ch]=s[lane];aux[blockIdx.y*c+ch]=t[lane];}
}
template<int Mode>
void bn_launch_partial(BulletOuCudaCppContext* ctx,const float* a,const float* b,const float* xa,const float* xb,
    const int* ids,float* stats,size_t rows,size_t stride,size_t width,size_t groups) {
    size_t c=width*groups;
    if(width%32==0) {
        dim3 grid(static_cast<unsigned>((c+31)/32),static_cast<unsigned>((rows+1023)/1024));
        bn_partial<Mode,32><<<grid,256,0,ctx->stream>>>(a,b,xa,xb,ids,stats,rows,stride,width,groups);
    } else {
        dim3 grid(static_cast<unsigned>((c+7)/8),static_cast<unsigned>((rows+1023)/1024));
        bn_partial<Mode,8><<<grid,256,0,ctx->stream>>>(a,b,xa,xb,ids,stats,rows,stride,width,groups);
    }
}
template<int Mode>
__global__ void bn_finish(float* stats,float* running,float* grads,size_t rows,size_t c,float epsilon,float momentum) {
    size_t ch=blockIdx.x*blockDim.x+threadIdx.x;if(ch>=c)return;
    size_t tiles=(rows+1023)/1024;
    double* mean=reinterpret_cast<double*>(stats+2*c);
    double* sums=mean+c;double* aux=sums+tiles*c;
    double p=0,q=0;for(size_t t=0;t<tiles;++t){p+=sums[t*c+ch];q+=aux[t*c+ch];}
    if constexpr(Mode==0) {mean[ch]=q>0?p/q:0;}
    if constexpr(Mode==1) {
        double variance;
        if(q>=2) {
            variance=p/q;float m=running[2*c+ch]>0?momentum:1.0f;
            running[ch]=(1-m)*running[ch]+m*float(mean[ch]);
            running[c+ch]=(1-m)*running[c+ch]+m*float(p/(q-1));
            running[2*c+ch]=1;stats[c+ch]=float(q);
        } else {mean[ch]=running[ch];variance=running[c+ch];stats[c+ch]=0;}
        stats[ch]=1.0f/sqrtf(float(variance)+epsilon);
    }
    if constexpr(Mode==2) {
        grads[ch]+=float(q);grads[c+ch]+=float(p);
        float n=stats[c+ch];
        sums[ch]=n>0?float(p)/n:0;aux[ch]=n>0?float(q)/n:0;
    }
}
template<bool Backward>
__global__ void bn_apply(float* a,float* b,float* xa,float* xb,const int* ids,
    const float* params,const float* stats,size_t rows,size_t stride,size_t width,size_t groups) {
    size_t j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=rows*width)return;
    size_t row=j/width,u=j%width,g=groups>1?static_cast<size_t>(ids[row]):0,c=width*groups,ch=g*width+u,k=row*stride+u;
    const double* mean=reinterpret_cast<const double*>(stats+2*c);
    if constexpr(!Backward) {
        float x=float(double(a[k])-mean[ch])*stats[ch];xa[k]=x;a[k]=params[ch]*x+params[c+ch];
        if(b){x=float(double(b[k])-mean[ch])*stats[ch];xb[k]=x;b[k]=params[ch]*x+params[c+ch];}
    } else {
        const double* sums=mean+c;const double* aux=sums+((rows+1023)/1024)*c;
        float sm=float(sums[ch]),pm=float(aux[ch]),scale=params[ch]*stats[ch];
        a[k]=scale*(a[k]-sm-xa[k]*pm);
        if(b)b[k]=scale*(b[k]-sm-xb[k]*pm);
    }
}
__global__ void bn_inference_kernel(float* a,float* b,const int* ids,const float* params,
    const float* running,size_t rows,size_t stride,size_t width,size_t groups,float epsilon) {
    size_t j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=rows*width)return;
    size_t row=j/width,u=j%width,g=groups>1?static_cast<size_t>(ids[row]):0,c=width*groups,ch=g*width+u,k=row*stride+u;
    float inv=1.0f/sqrtf(running[c+ch]+epsilon);
    float x=float(double(a[k])-double(running[ch]))*inv;
    a[k]=params[ch]*x+params[c+ch];
    if(b) {x=float(double(b[k])-double(running[ch]))*inv;b[k]=params[ch]*x+params[c+ch];}
}
thread_local bool bn_reference_test=false;
// FT training: normalize, save x-hat for backward, clamp and multiply in one pass.
__global__ void bn_ft_apply_activate(float* a,float* b,float* xa,float* xb,float* combined,
    const float* params,const float* stats,size_t rows,size_t width) {
    size_t j=blockIdx.x*blockDim.x+threadIdx.x,half=width/2;
    if(j>=rows*half)return;
    size_t base=(j/half)*width,u=j%half;
    const double* mean=reinterpret_cast<const double*>(stats+2*width);
    float av[2],bv[2];
    for(int part=0;part<2;++part) {
        size_t ch=u+part*half,k=base+ch;
        float x=float(double(a[k])-mean[ch])*stats[ch];xa[k]=x;
        av[part]=crelu(params[ch]*x+params[width+ch]);a[k]=av[part];
        x=float(double(b[k])-mean[ch])*stats[ch];xb[k]=x;
        bv[part]=crelu(params[ch]*x+params[width+ch]);b[k]=bv[part];
    }
    combined[base+u]=av[0]*av[1]*SFNN_PAIRWISE_SCALE;
    combined[base+half+u]=bv[0]*bv[1]*SFNN_PAIRWISE_SCALE;
}
extern "C" void bulletou_bn_reference_mode(int enabled) {bn_reference_test=enabled!=0;}
bool bn_use_reference() {return bn_reference_test || std::getenv("BULLETOU_BN_REFERENCE")!=nullptr;}
void bn_launch_forward(BulletOuCudaCppContext* ctx,float* a,float* b,float* xa,float* xb,
    const int* ids,const float* params,float* running,float* stats,
    size_t rows,size_t stride,size_t width,size_t groups,float epsilon,float momentum,bool training,float* ft_combined=nullptr) {
    if(!training && !bn_use_reference()) {
        bn_inference_kernel<<<static_cast<unsigned>((rows*width+255)/256),256,0,ctx->stream>>>(
            a,b,ids,params,running,rows,stride,width,groups,epsilon);
        return;
    }
    if(bn_use_reference()) bn_forward_kernel<<<static_cast<unsigned>(width*groups),256,0,ctx->stream>>>(
        a,b,xa,xb,ids,params,running,stats,rows,stride,width,groups,epsilon,momentum,training);
    else {
        size_t c=width*groups;
        bn_launch_partial<0>(ctx,a,b,xa,xb,ids,stats,rows,stride,width,groups);
        bn_finish<0><<<static_cast<unsigned>((c+255)/256),256,0,ctx->stream>>>(stats,running,nullptr,rows,c,epsilon,momentum);
        bn_launch_partial<1>(ctx,a,b,xa,xb,ids,stats,rows,stride,width,groups);
        bn_finish<1><<<static_cast<unsigned>((c+255)/256),256,0,ctx->stream>>>(stats,running,nullptr,rows,c,epsilon,momentum);
        if(ft_combined) {
            bn_ft_apply_activate<<<static_cast<unsigned>((rows*(width/2)+255)/256),256,0,ctx->stream>>>(a,b,xa,xb,ft_combined,params,stats,rows,width);
        } else {
            bn_apply<false><<<static_cast<unsigned>((rows*width+255)/256),256,0,ctx->stream>>>(a,b,xa,xb,ids,params,stats,rows,stride,width,groups);
        }
    }
}
void bn_launch_backward(BulletOuCudaCppContext* ctx,float* da,float* db,const float* xa,const float* xb,
    const int* ids,const float* params,const float* stats,float* grads,
    size_t rows,size_t stride,size_t width,size_t groups) {
    if(bn_use_reference()) bn_backward_kernel<<<static_cast<unsigned>(width*groups),256,0,ctx->stream>>>(
        da,db,xa,xb,ids,params,stats,grads,rows,stride,width,groups);
    else {
        size_t c=width*groups;
        float* scratch=const_cast<float*>(stats);
        bn_launch_partial<2>(ctx,da,db,xa,xb,ids,scratch,rows,stride,width,groups);
        bn_finish<2><<<static_cast<unsigned>((c+255)/256),256,0,ctx->stream>>>(scratch,nullptr,grads,rows,c,0,0);
        bn_apply<true><<<static_cast<unsigned>((rows*width+255)/256),256,0,ctx->stream>>>(da,db,const_cast<float*>(xa),const_cast<float*>(xb),ids,params,stats,rows,stride,width,groups);
    }
}

extern "C" int bulletou_bn_forward(BulletOuCudaCppContext* ctx,
    BulletOuCudaCppF32Buffer* a,BulletOuCudaCppF32Buffer* b,
    BulletOuCudaCppF32Buffer* xa,BulletOuCudaCppF32Buffer* xb,
    BulletOuCudaCppI32Buffer* ids,BulletOuCudaCppF32Buffer* params,
    BulletOuCudaCppF32Buffer* running,BulletOuCudaCppF32Buffer* stats,
    size_t rows,size_t stride,size_t width,size_t groups,float epsilon,float momentum,int training) {
    if(!rows || !width || stride<width || !groups || groups>65536/width || rows>SIZE_MAX/stride ||
       !std::isfinite(epsilon) || epsilon<=0 || !std::isfinite(momentum) || momentum<=0 || momentum>1)
        return fail_message("invalid batch normalization shape/configuration");
    const size_t n=rows*stride,c=groups*width;
    if(validate_buffer(ctx,a,n,"BN input") || validate_buffer(ctx,xa,n,"BN normalized input") ||
       validate_buffer(ctx,params,2*c,"BN affine") || validate_buffer(ctx,running,3*c,"BN running stats") ||
       validate_buffer(ctx,stats,(4+4*((rows+1023)/1024))*c,"BN batch stats/scratch")) return -1;
    if((b==nullptr)!=(xb==nullptr)) return fail_message("BN second view requires matching workspace");
    if(b && (validate_buffer(ctx,b,n,"BN second view") || validate_buffer(ctx,xb,n,"BN second view workspace"))) return -1;
    if(groups>1 && validate_i32_buffer(ctx,ids,rows,"BN buckets")) return -1;
    if(set_context_device(ctx)) return -1;
    bn_launch_forward(ctx,a->ptr,b?b->ptr:nullptr,xa->ptr,xb?xb->ptr:nullptr,
        ids?ids->ptr:nullptr,params->ptr,running->ptr,stats->ptr,rows,stride,width,groups,epsilon,momentum,training!=0);
    return check_kernel_launch("BN forward");
}

extern "C" int bulletou_bn_backward(BulletOuCudaCppContext* ctx,
    BulletOuCudaCppF32Buffer* da,BulletOuCudaCppF32Buffer* db,
    BulletOuCudaCppF32Buffer* xa,BulletOuCudaCppF32Buffer* xb,
    BulletOuCudaCppI32Buffer* ids,BulletOuCudaCppF32Buffer* params,
    BulletOuCudaCppF32Buffer* stats,BulletOuCudaCppF32Buffer* grads,
    size_t rows,size_t stride,size_t width,size_t groups) {
    if(!rows || !width || stride<width || !groups || groups>65536/width || rows>SIZE_MAX/stride)
        return fail_message("invalid BN backward shape");
    const size_t n=rows*stride,c=groups*width;
    if(validate_buffer(ctx,da,n,"BN gradient") || validate_buffer(ctx,xa,n,"BN normalized input") ||
       validate_buffer(ctx,params,2*c,"BN affine") || validate_buffer(ctx,stats,(4+4*((rows+1023)/1024))*c,"BN batch stats/scratch") ||
       validate_buffer(ctx,grads,2*c,"BN affine gradients")) return -1;
    if((db==nullptr)!=(xb==nullptr)) return fail_message("BN backward second view requires matching workspace");
    if(db && (validate_buffer(ctx,db,n,"BN second gradient") || validate_buffer(ctx,xb,n,"BN second normalized input"))) return -1;
    if(groups>1 && validate_i32_buffer(ctx,ids,rows,"BN buckets")) return -1;
    if(set_context_device(ctx)) return -1;
    bn_launch_backward(ctx,da->ptr,db?db->ptr:nullptr,xa->ptr,xb?xb->ptr:nullptr,
        ids?ids->ptr:nullptr,params->ptr,stats->ptr,grads->ptr,rows,stride,width,groups);
    return check_kernel_launch("BN backward");
}

int bn_forward_bound(BulletOuCudaCppContext* ctx,int layer,float* a,float* b,
    const int* ids,size_t rows,size_t stride,float* ft_combined=nullptr) {
    const auto c=ctx->bn[layer];if(!c.params)return 0;
    bn_launch_forward(ctx,
        a,b,c.xa,c.xb,ids,c.params,c.running,c.stats,rows,stride,c.width,c.groups,c.epsilon,c.momentum,c.training,ft_combined);
    return check_kernel_launch("SFNN BN forward");
}
int bn_backward_bound(BulletOuCudaCppContext* ctx,int layer,float* a,float* b,
    const int* ids,size_t rows,size_t stride) {
    const auto c=ctx->bn[layer];if(!c.params)return 0;
    bn_launch_backward(ctx,
        a,b,c.xa,c.xb,ids,c.params,c.stats,c.grads,rows,stride,c.width,c.groups);
    return check_kernel_launch("SFNN BN backward");
}
__global__ void bn_bias_finish(const float* stats,float* gb,size_t rows,size_t width) {
    size_t u=blockIdx.x*blockDim.x+threadIdx.x;if(u>=width)return;
    const double* sums=reinterpret_cast<const double*>(stats+2*width)+width;
    double s=0;for(size_t t=0;t<(rows+1023)/1024;++t)s+=sums[t*width+u];
    gb[u]+=float(s);
}
void bn_bias_sum_bound(BulletOuCudaCppContext* ctx,const float* a,const float* b,float* gb,size_t rows,size_t width) {
    if(bn_use_reference()) {
        bn_bias_sum<<<static_cast<unsigned>((width+255)/256),256,0,ctx->stream>>>(a,b,gb,rows,width);
    } else {
        bn_launch_partial<0>(ctx,a,b,nullptr,nullptr,nullptr,ctx->bn[0].stats,rows,width,width,1);
        bn_bias_finish<<<static_cast<unsigned>((width+255)/256),256,0,ctx->stream>>>(ctx->bn[0].stats,gb,rows,width);
    }
}
extern "C" int bulletou_bn_bind(BulletOuCudaCppContext* ctx,int layer,
    BulletOuCudaCppF32Buffer* params,BulletOuCudaCppF32Buffer* running,BulletOuCudaCppF32Buffer* stats,
    BulletOuCudaCppF32Buffer* grads,BulletOuCudaCppF32Buffer* xa,BulletOuCudaCppF32Buffer* xb,
    size_t rows,size_t stride,size_t width,size_t groups,float epsilon,float momentum,int training) {
    if(layer<0 || layer>2)return fail_message("invalid BN layer");
    if(!params) {ctx->bn[layer]=BnConfig{};return 0;}
    if(!rows || !width || stride<width || !groups || groups>65536/width || rows>SIZE_MAX/stride ||
        !std::isfinite(epsilon) || epsilon<=0 || !std::isfinite(momentum) || momentum<=0 || momentum>1)
        return fail_message("invalid BN binding");
    size_t c=width*groups,n=rows*stride;
    if(validate_buffer(ctx,params,2*c,"BN params") || validate_buffer(ctx,running,3*c,"BN running") ||
       validate_buffer(ctx,stats,(4+4*((rows+1023)/1024))*c,"BN stats/scratch") || validate_buffer(ctx,grads,2*c,"BN gradients") ||
       validate_buffer(ctx,xa,n,"BN workspace") || (layer==0 && validate_buffer(ctx,xb,n,"BN second workspace")))return -1;
    ctx->bn[layer]=BnConfig{params->ptr,running->ptr,stats->ptr,grads->ptr,xa->ptr,xb?xb->ptr:nullptr,width,groups,epsilon,momentum,training!=0};
    return 0;
}
