// Independent test-only float64 CUDA oracle. No native conv/resize/GEMM calls.
#include "fullframe_compression_reference.cu"

__global__ static void ae_ref_expand(const double* x,double* y,int B,int H,int W,int OH,int OW,int C) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=B*OH*OW*C)return;
    int c=i%C,xx=i/C%OW,yy=i/(C*OW)%OH,b=i/(C*OW*OH);
    y[i]=x[((b*H+(long)yy*H/OH)*W+(long)xx*W/OW)*C+c];
}

__global__ static void ae_ref_expand_back(const double* g,double* dx,int B,int H,int W,int OH,int OW,int C) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=B*H*W*C)return;
    int c=i%C,x=i/C%W,y=i/(C*W)%H,b=i/(C*W*H);double sum=0;
    for(int yy=max(0,y*OH/H-1);yy<min(OH,(y+1)*OH/H+2);yy++) if(yy*H/OH==y)
        for(int xx=max(0,x*OW/W-1);xx<min(OW,(x+1)*OW/W+2);xx++) if(xx*W/OW==x)
            sum+=g[((b*OH+yy)*OW+xx)*C+c];
    dx[i]=sum;
}

__global__ static void ae_ref_convolution(const double* x,const double* w,double* y,Layer l,int B,bool relu) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=B*l.oh*l.ow*l.co)return;
    int c=i%l.co,xx=i/l.co%l.ow,yy=i/(l.co*l.ow)%l.oh,b=i/(l.co*l.ow*l.oh);
    double sum=w[l.offset+l.co*l.k*l.k*l.ci+c];
    for(int ky=0;ky<l.k;ky++)for(int kx=0;kx<l.k;kx++)for(int ci=0;ci<l.ci;ci++) {
        int sy=yy+ky-l.ph,sx=xx+kx-l.pw;
        if(sy>=0&&sy<l.h&&sx>=0&&sx<l.w)
            sum+=x[((b*l.h+sy)*l.w+sx)*l.ci+ci]*w[l.offset+((c*l.k+ky)*l.k+kx)*l.ci+ci];
    }
    y[i]=relu?fmax(0.,sum):sum;
}

struct AEReferenceDecoder {
    Storage storage;
    Layer layers[4];
    double *resized[4],*grad_resized[4],*output[4],*g[4],*delta[4],*grad_latent;
    int sh[4],sw[4],B,count=4,latent=128;
    AEReferenceDecoder(int H,int W,int batch,int stride=0,int channels=16):B(batch) {
        int hs[]={1,(H+15)/16,(H+3)/4,H},ws[]={1,(W+15)/16,(W+3)/4,W},cs[]={16,8,8,3};
        if(stride) {
            count=3; latent=((H+stride-1)/stride)*((W+stride-1)/stride)*channels;
            hs[0]=(H+3)/4;hs[1]=(H+1)/2;hs[2]=H;
            ws[0]=(W+3)/4;ws[1]=(W+1)/2;ws[2]=W;
            cs[0]=16;cs[1]=8;cs[2]=3;
        }
        int offset=0;
        for(int i=0;i<count;i++) {
            int ci=i?cs[i-1]:(stride?channels:128),co=i||stride?cs[i]:256,k=i||stride?3:1;
            layers[i]={hs[i],ws[i],ci,hs[i],ws[i],co,k,1,k==3?1:0,k==3?1:0,offset};
            offset+=co*(k*k*ci+1);
            int n=B*hs[i]*ws[i]*co;
            output[i]=storage.add(n);g[i]=storage.add(n);delta[i]=storage.add(n);
            resized[i]=storage.add(B*hs[i]*ws[i]*ci);grad_resized[i]=storage.add(B*hs[i]*ws[i]*ci);
            sh[i]=stride?(i?hs[i-1]:(H+stride-1)/stride):i==1?4:i?hs[i-1]:1;
            sw[i]=stride?(i?ws[i-1]:(W+stride-1)/stride):i==1?4:i?ws[i-1]:1;
        }
        grad_latent=storage.add(B*latent);
    }
    void run(const double* z,const double* w,const double* up,double* dw) {
        for(int i=0;i<count;i++) {
            Layer l=layers[i];int size=B*l.oh*l.ow*l.co;
            if(!i && count==4) checked(cudaMemcpy(resized[i],z,B*128*8,cudaMemcpyDeviceToDevice));
            else ae_ref_expand<<<(B*l.h*l.w*l.ci+127)/128,128>>>(i?output[i-1]:z,resized[i],B,sh[i],sw[i],l.h,l.w,l.ci);
            ae_ref_convolution<<<(size+127)/128,128>>>(resized[i],w,output[i],l,B,i!=count-1);
        }
        Layer last=layers[count-1];checked(cudaMemcpy(g[count-1],up,B*last.oh*last.ow*3*8,cudaMemcpyDeviceToDevice));
        for(int i=count-1;i>=0;i--) {
            Layer l=layers[i];int size=B*l.oh*l.ow*l.co;
            if(i==count-1) checked(cudaMemcpy(delta[i],g[i],size*8,cudaMemcpyDeviceToDevice));
            else relu_backward<<<(size+127)/128,128>>>(output[i],g[i],delta[i],size);
            int weights=l.co*l.k*l.k*l.ci;
            weight_backward<<<(weights+127)/128,128>>>(resized[i],delta[i],dw,l,B);
            bias_backward<<<(l.co+127)/128,128>>>(delta[i],dw,l,B);
            input_backward<<<(B*l.h*l.w*l.ci+127)/128,128>>>(w,delta[i],grad_resized[i],l,B);
            if(i || count==3) ae_ref_expand_back<<<(B*sh[i]*sw[i]*l.ci+127)/128,128>>>(grad_resized[i],i?g[i-1]:grad_latent,B,sh[i],sw[i],l.h,l.w,l.ci);
            else checked(cudaMemcpy(grad_latent,grad_resized[0],B*128*8,cudaMemcpyDeviceToDevice));
        }
        checked(cudaDeviceSynchronize());
    }
};

__global__ static void ae_native_probe(const float* y,const float* up,double* loss,int n) {
    if(threadIdx.x||blockIdx.x)return;double value=0;
    for(int i=0;i<n;i++)value+=(double)y[i]*up[i];*loss=value;
}

static double ae_compare(const std::vector<float>& actual,const std::vector<double>& expected,const char* name) {
    ffc_require(actual.size()==expected.size(),"reference size");double worst=0;long bad=0;
    for(size_t i=0;i<actual.size();i++) {
        double e=fabs(actual[i]-expected[i]);worst=fmax(worst,e);
        if(!isfinite(actual[i])||!isfinite(expected[i])||e>6e-5+8e-4*fabs(expected[i])) {
            if(bad++<3)fprintf(stderr,"%s[%zu] %.12g != %.12g\n",name,i,actual[i],expected[i]);
        }
    }
    ffc_require(!bad,name);return worst;
}

static int ae_math(int H,int W,int B,const char* folder,int stride=0,int channels=16) {
    ffc_require(H>=144&&W>=144&&H<=640&&W<=640&&(B==1||B==2),"math fixture bounds");
    ffc_require(!mkdir(folder,0700),"fresh math output");
    cublas_init_handle();cudaStream_t stream;checked(cudaStreamCreate(&stream));
    FFCWeights* e=ffc_graph(0,H,W);e->input_grad=true;FFCActivations ea={};AEDecoder d=ae_decoder(H,W);AEActivations da={};
#ifdef CNN_SPATIAL_RESEARCH
    if(stride) {free(e);e=sc_graph(H,W,stride,channels);e->input_grad=true;d=ae_spatial_decoder(H,W,stride,channels);}
#endif
    Allocator p={},a={},g={};ffc_params(e,&p);nature_reg_params(&d.net,&p);ffc_train(e,&ea,&a,&g,B);ae_register(&d,&da,&a,&g,B);
    alloc_create(&p);alloc_create(&a);alloc_create(&g);
    long N=(long)B*H*W*3,P=p.total_elems,E=0;
    for(int i=0;i<e->net.count;i++)E+=numel(e->net.weight[i].shape)+numel(e->net.bias[i].shape);
    std::vector<float> hw(P),hx(N),hu(N),y(N),dw(P),dx(N);
    std::vector<double> rw(P),rx(N),ru(N),ry(N),rdw(P),rdx(N);
    for(long i=0;i<P;i++)rw[i]=hw[i]=.005f+(i%17)*.0001f;
    for(int i=0;i<3;i++)rw[P-3+i]=hw[P-3+i]=(i==1?.1f:-.2f);
    for(long i=0;i<N;i++) {rx[i]=hx[i]=.1f+(i%257)/512.f;ru[i]=hu[i]=.01f*(1+i%7);}
    Prec input={.shape={B,H*W*3}};checked(cudaMalloc(&input.data,N*4));checked(cudaMemcpy(input.data,hx.data(),N*4,cudaMemcpyHostToDevice));
    checked(cudaMemcpy(p.mem,hw.data(),P*4,cudaMemcpyHostToDevice));
    float* up;checked(cudaMalloc(&up,N*4));checked(cudaMemcpy(up,hu.data(),N*4,cudaMemcpyHostToDevice));
    auto native_forward=[&](){Prec z=ffc_forward(e,&ea,input,stream);return ae_forward(&d,&da,z,stream);};
    auto native=[&](){Prec out=native_forward();checked(cudaMemcpyAsync(da.net.grad_output[d.net.count-1].data,up,N*4,cudaMemcpyDeviceToDevice,stream));
        Prec dz=ae_backward(&d,&da,stream);ffc_backward(e,&ea,dz,stream);return out;};
    // Synchronize the first eager pass at the boundary that failed, rather than
    // attributing an asynchronous device error to a later oracle allocation.
    Prec first_z=ffc_forward(e,&ea,input,stream);
    ffc_require(cudaStreamSynchronize(stream)==cudaSuccess,"AE math encoder forward device error");
    Prec output=ae_forward(&d,&da,first_z,stream);
    ffc_require(cudaStreamSynchronize(stream)==cudaSuccess,"AE math decoder forward device error");
    checked(cudaMemcpyAsync(da.net.grad_output[d.net.count-1].data,up,N*4,cudaMemcpyDeviceToDevice,stream));
    Prec first_dz=ae_backward(&d,&da,stream);
    ffc_require(cudaStreamSynchronize(stream)==cudaSuccess,"AE math decoder backward device error");
    ffc_backward(e,&ea,first_dz,stream);
    ffc_require(cudaStreamSynchronize(stream)==cudaSuccess,"AE math encoder backward device error");
    checked(cudaMemcpy(y.data(),output.data,N*4,cudaMemcpyDeviceToHost));
    checked(cudaMemcpy(dw.data(),g.mem,P*4,cudaMemcpyDeviceToHost));checked(cudaMemcpy(dx.data(),ea.grad_input.data,N*4,cudaMemcpyDeviceToHost));
    FFCReference er(0,H,W,B,stride,channels);AEReferenceDecoder dr(H,W,B,stride,channels);Storage memory;
    double *obs=memory.add(N),*weights=memory.add(P),*upstream=memory.add(N),*gradient=memory.add(P),*zero=memory.add(B*d.latent_values);
    checked(cudaMemcpy(obs,rx.data(),N*8,cudaMemcpyHostToDevice));checked(cudaMemcpy(weights,rw.data(),P*8,cudaMemcpyHostToDevice));
    checked(cudaMemcpy(upstream,ru.data(),N*8,cudaMemcpyHostToDevice));checked(cudaMemset(zero,0,B*d.latent_values*8));
    er.run(obs,weights,zero,gradient);dr.run(er.nodes.back().y,weights+E,upstream,gradient+E);
    std::vector<float> native_dz(B*d.latent_values);std::vector<double> reference_dz(B*d.latent_values);
    checked(cudaMemcpy(native_dz.data(),da.grad_latent.data,native_dz.size()*4,cudaMemcpyDeviceToHost));
    checked(cudaMemcpy(reference_dz.data(),dr.grad_latent,reference_dz.size()*8,cudaMemcpyDeviceToHost));
    ae_compare(native_dz,reference_dz,"AE latent derivative");
    er.run(obs,weights,dr.grad_latent,gradient);
    checked(cudaMemcpy(ry.data(),dr.output[dr.count-1],N*8,cudaMemcpyDeviceToHost));checked(cudaMemcpy(rdw.data(),gradient,P*8,cudaMemcpyDeviceToHost));
    checked(cudaMemcpy(rdx.data(),er.nodes[0].g,N*8,cudaMemcpyDeviceToHost));
    double ey=ae_compare(y,ry,"AE forward"),eg=ae_compare(dw,rdw,"AE gradient"),ex=ae_compare(dx,rdx,"AE input gradient");
    int probes=0;double fd_error=0;
    if(H==160&&W==192&&B==1) {
        double* loss;checked(cudaMalloc(&loss,8));long offset=0;
        auto value=[&](){Prec out=native_forward();ae_native_probe<<<1,1,0,stream>>>(out.data,up,loss,N);double v;
            checked(cudaMemcpyAsync(&v,loss,8,cudaMemcpyDeviceToHost,stream));checked(cudaStreamSynchronize(stream));return v;};
        for(int i=0;i<p.num_regs;i++) {
            long n=numel(p.regs[i].shape),at=offset+n/2;float lo=hw[at]-.0002f,hi=hw[at]+.0002f;
            checked(cudaMemcpy((float*)p.mem+at,&lo,4,cudaMemcpyHostToDevice));double lv=value();
            checked(cudaMemcpy((float*)p.mem+at,&hi,4,cudaMemcpyHostToDevice));double hv=value();
            checked(cudaMemcpy((float*)p.mem+at,&hw[at],4,cudaMemcpyHostToDevice));
            double error=fabs((hv-lv)/(hi-lo)-dw[at]);fd_error=fmax(fd_error,error);
            ffc_require(error<=3e-3+3e-2*fabs(dw[at]),"AE finite difference");offset+=n;probes++;
        }
        checked(cudaFree(loss));
    }
    cudaGraph_t graph;cudaGraphExec_t execution;
    checked(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));native();checked(cudaStreamEndCapture(stream,&graph));
    checked(cudaGraphInstantiate(&execution,graph,0));
    for(int repeat=0;repeat<3;repeat++) {
        if(repeat)checked(cudaGraphLaunch(execution,stream));else native();checked(cudaStreamSynchronize(stream));
        std::vector<float> yy(N),gg(P),xx(N),ww(P);
        checked(cudaMemcpy(yy.data(),output.data,N*4,cudaMemcpyDeviceToHost));checked(cudaMemcpy(gg.data(),g.mem,P*4,cudaMemcpyDeviceToHost));
        checked(cudaMemcpy(xx.data(),ea.grad_input.data,N*4,cudaMemcpyDeviceToHost));checked(cudaMemcpy(ww.data(),p.mem,P*4,cudaMemcpyDeviceToHost));
        ffc_require(!memcmp(yy.data(),y.data(),N*4)&&!memcmp(gg.data(),dw.data(),P*4)
            &&!memcmp(xx.data(),dx.data(),N*4)&&!memcmp(ww.data(),hw.data(),P*4),"AE eager/graph repeat or mutation");
    }
    char yh[65],gh[65],xh[65];ffc_sha256((unsigned char*)y.data(),N*4,yh);ffc_sha256((unsigned char*)dw.data(),P*4,gh);ffc_sha256((unsigned char*)dx.data(),N*4,xh);
    char result[2048];snprintf(result,sizeof(result),
        "{\"status\":\"passed\",\"height\":%d,\"width\":%d,\"batch\":%d,\"parameters\":%ld,"
        "\"forward_error\":%.17g,\"gradient_error\":%.17g,\"input_gradient_error\":%.17g,\"finite_difference_probes\":%d,"
        "\"finite_difference_error\":%.17g,\"eager_graph_repeats\":3,\"output_sha256\":\"%s\",\"gradient_sha256\":\"%s\",\"input_gradient_sha256\":\"%s\"}\n",
        H,W,B,P,ey,eg,ex,probes,fd_error,yh,gh,xh);
    char path[4096];snprintf(path,sizeof(path),"%s/result.json",folder);ae_write(path,result,strlen(result));puts(result);return 0;
}

__global__ static void ae_ref_loss(const float* y,const float* x,double* gradient,double* sums,
        int count,int active,int H,int W,double weight) {
    if(blockIdx.x||threadIdx.x)return;double raw=0,objective=0;
    for(int i=0;i<count;i++) {
        gradient[i]=0;if(i>=active)continue;
        int xx=i/3%W,yy=i/(3*W)%H;
        double factor=1+weight*((xx&&fabs((double)x[i]-x[i-3])>.05)||(yy&&fabs((double)x[i]-x[i-3*W])>.05));
        double delta=(double)y[i]-x[i];raw+=delta*delta;objective+=factor*delta*delta;
        gradient[i]=2*factor*delta/active;
    }
    sums[0]=raw/active;sums[1]=objective/active;
}

static int ae_auxiliary() {
    int H=5,W=7,n=2*H*W*3,active=H*W*3;
    std::vector<float> x(n),y(n),g(n);std::vector<double> expected(n);
    for(int i=0;i<n;i++){x[i]=(i%17)/16.f;y[i]=(i%13-6)/8.f;}
    float *dx,*dy,*dg,*parts,*sums;double *reference,*totals;
    checked(cudaMalloc(&dx,n*4));checked(cudaMalloc(&dy,n*4));checked(cudaMalloc(&dg,n*4));
    checked(cudaMalloc(&parts,512*4));checked(cudaMalloc(&sums,8));checked(cudaMalloc(&reference,n*8));checked(cudaMalloc(&totals,16));
    checked(cudaMemcpy(dx,x.data(),n*4,cudaMemcpyHostToDevice));checked(cudaMemcpy(dy,y.data(),n*4,cudaMemcpyHostToDevice));
    double worst=0;
    for(float weight:{0.f,2.f}) {
        ae_loss<<<256,256>>>(dy,dx,dg,parts,n,active,H,W,weight);
        muon_sum_sq_reduce<<<1,256>>>(sums,parts,256);muon_sum_sq_reduce<<<1,256>>>(sums+1,parts+256,256);
        ae_ref_loss<<<1,1>>>(dy,dx,reference,totals,n,active,H,W,weight);
        float result[2];double truth[2];
        checked(cudaMemcpy(result,sums,8,cudaMemcpyDeviceToHost));checked(cudaMemcpy(truth,totals,16,cudaMemcpyDeviceToHost));
        checked(cudaMemcpy(g.data(),dg,n*4,cudaMemcpyDeviceToHost));checked(cudaMemcpy(expected.data(),reference,n*8,cudaMemcpyDeviceToHost));
        for(int i=0;i<2;i++)ffc_require(isfinite(result[i])&&fabs(result[i]-truth[i])<2e-6,"weighted/raw MSE reduction");
        worst=fmax(worst,ae_compare(g,expected,"MSE derivative and inactive tail"));
    }
    float hw[4]={1,-2,.5f,0},hg[4]={3,-4,.001f,0},zero[4]={},norm=25.000001f;
    double ew[4]={1,-2,.5,0},em[4]={},ev[4]={};float *ww,*gg,*mm,*vv,*nn;
    checked(cudaMalloc(&ww,16));checked(cudaMalloc(&gg,16));checked(cudaMalloc(&mm,16));checked(cudaMalloc(&vv,16));checked(cudaMalloc(&nn,4));
    checked(cudaMemcpy(ww,hw,16,cudaMemcpyHostToDevice));checked(cudaMemcpy(gg,hg,16,cudaMemcpyHostToDevice));
    checked(cudaMemcpy(mm,zero,16,cudaMemcpyHostToDevice));checked(cudaMemcpy(vv,zero,16,cudaMemcpyHostToDevice));checked(cudaMemcpy(nn,&norm,4,cudaMemcpyHostToDevice));
    for(int step=1;step<=2;step++) {
        ae_adam<<<1,32>>>(ww,gg,mm,vv,nn,4,.0003f,.5f,1-pow(.9,step),1-pow(.999,step));
        checked(cudaMemcpy(hw,ww,16,cudaMemcpyDeviceToHost));
        for(int i=0;i<4;i++) {
            double grad=hg[i]*fmin(1.,.5/(sqrt(norm)+1e-6));
            em[i]=.9*em[i]+.1*grad;ev[i]=.999*ev[i]+.001*grad*grad;
            ew[i]-=.0003*(em[i]/(1-pow(.9,step)))/(sqrt(ev[i]/(1-pow(.999,step)))+1e-8);
            ffc_require(isfinite(hw[i])&&fabs(hw[i]-ew[i])<1e-6,"Adam scalar algebra");
        }
    }
    printf("{\"status\":\"passed\",\"loss_weights\":[0,2],\"inactive_tail_checked\":true,\"adam_steps\":2,\"loss_gradient_error\":%.17g}\n",worst);
    return 0;
}

static int ae_import_check(const char* recipe,const char* directory) {
    Ini ini={};puf_ini_load_file(&ini,recipe);Dict* policy=puf_ini_section(&ini,"policy",0);
    int sizes[]=ACT_SIZES;
    Arch arch=build_arch(OBS_SIZE,128,1,sizes[0],false,16,policy);
    Allocator a={},b={},enc={};Weights wa=weights_create(&arch,&a),wb=weights_create(&arch,&b);
    arch.encoder.reg_params(wa.encoder,&enc);alloc_create(&a);alloc_create(&b);cublas_init_handle();cudaStream_t stream;
    checked(cudaStreamCreate(&stream));uint64_t sa=7,sb=7;weights_init(&arch,wa,&sa,stream);
    ((FFCWeights*)wb.encoder)->initial=strdup(directory);weights_init(&arch,wb,&sb,stream);checked(cudaStreamSynchronize(stream));
    std::vector<float> scratch(a.total_elems),imported(b.total_elems);
    checked(cudaMemcpy(scratch.data(),a.mem,a.total_bytes,cudaMemcpyDeviceToHost));checked(cudaMemcpy(imported.data(),b.mem,b.total_bytes,cudaMemcpyDeviceToHost));
    ffc_require(sa==sb&&!memcmp(scratch.data()+enc.total_elems,imported.data()+enc.total_elems,(a.total_elems-enc.total_elems)*4),"import changed core/head/RNG");
    std::string file=std::string(directory)+"/encoder.bin";size_t bytes;char* raw=ffc_read_file(file.c_str(),&bytes);
    ffc_require(bytes==enc.total_elems*4&&!memcmp(imported.data(),raw,bytes),"autoencoder import differs");
    puts("{\"status\":\"passed\",\"core_head_unchanged\":true,\"rng_unchanged\":true,\"encoder_exact\":true}");return 0;
}
