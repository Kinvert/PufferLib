// Independent CUDA scalar references for the new loss/probe arithmetic.
__global__ static void fp_ref_class(const float* x,const float* weights,const float* cw,
        const unsigned char* labels,double* delta,double* loss,int B,int F,int cells,int C) {
    if(blockIdx.x||threadIdx.x)return;int O=cells*C;double total=0;
    for(int b=0;b<B;b++)for(int cell=0;cell<cells;cell++) {
        double logits[13],top=-INFINITY,sum=0;
        for(int c=0;c<C;c++) {
            int row=cell*C+c;double value=weights[O*F+row];
            for(int f=0;f<F;f++)value+=(double)x[b*F+f]*weights[row*F+f];
            logits[c]=value;top=fmax(top,value);
        }
        for(int c=0;c<C;c++)sum+=exp(logits[c]-top);
        int truth=labels[b*cells+cell];double factor=cw[truth]/(B*(double)cells);
        total+=factor*(log(sum)+top-logits[truth]);
        for(int c=0;c<C;c++)delta[b*O+cell*C+c]=factor*(exp(logits[c]-top)/sum-(c==truth));
    }
    *loss=total;
}

__global__ static void fp_ref_head_gradient(const float* x,const double* delta,double* grad,int B,int F,int O) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=O*(F+1))return;
    int row=i<O*F?i/F:i-O*F;double sum=0;
    for(int b=0;b<B;b++)sum+=delta[b*O+row]*(i<O*F?x[b*F+i%F]:1.);
    grad[i]=sum;
}

__global__ static void fp_ref_weighted(const float* prediction,const float* target,const float* weights,
        double* grad,double* result,int n,int active) {
    if(threadIdx.x||blockIdx.x)return;double sum=0;
    for(int i=0;i<n;i++) {
        double error=(double)prediction[i]-target[i];
        grad[i]=i<active?2*weights[i/3]*error/active:0;
        if(i<active)sum+=(double)weights[i/3]*error*error/active;
    }
    *result=sum;
}

static int forage_probe_checks() {
    ffc_require(!forage_annotation_checks(),"Forage mask contract");
    cublas_init_handle();cudaStream_t stream;ae_checked(cudaStreamCreate(&stream));
    const int B=3,F=4,cells=2,C=3,O=cells*C,P=O*(F+1);
    std::vector<float> hx(B*F),hw(P),native_grad(P),native_delta(B*O),loss(B*cells);
    for(int i=0;i<B*F;i++)hx[i]=(i%7-3)/8.f;
    for(int i=0;i<P;i++)hw[i]=(i%11-5)/16.f;
    float cw[]={.5f,1,2};unsigned char hy[]={0,2,1,0,2,1};
    Prec x={.shape={B,F}},w={.shape={O,F}},g={.shape={O,F}},logits={.shape={B,O}},delta={.shape={B,O}};
    float *device_w,*device_grad,*weights,*device_loss;unsigned char* labels;double *ref_delta,*ref_grad,*ref_loss;
    ae_checked(cudaMalloc(&x.data,B*F*4));ae_checked(cudaMalloc(&device_w,P*4));w.data=device_w;
    ae_checked(cudaMalloc(&device_grad,P*4));g.data=device_grad;
    ae_checked(cudaMalloc(&logits.data,B*O*4));ae_checked(cudaMalloc(&delta.data,B*O*4));
    ae_checked(cudaMalloc(&weights,C*4));ae_checked(cudaMalloc(&device_loss,B*cells*4));ae_checked(cudaMalloc(&labels,B*cells));
    ae_checked(cudaMalloc(&ref_delta,B*O*8));ae_checked(cudaMalloc(&ref_grad,P*8));ae_checked(cudaMalloc(&ref_loss,8));
    ae_checked(cudaMemcpy(x.data,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));ae_checked(cudaMemcpy(device_w,hw.data(),P*4,cudaMemcpyHostToDevice));
    ae_checked(cudaMemcpy(weights,cw,C*4,cudaMemcpyHostToDevice));ae_checked(cudaMemcpy(labels,hy,B*cells,cudaMemcpyHostToDevice));
    auto forward=[&](){puf_mm(&x,&w,&logits,stream);ae_checked(cudaStreamSynchronize(stream));
        fp_softmax<<<1,32>>>(logits.data,device_w+O*F,labels,weights,delta.data,device_loss,B,cells,C);
        ae_checked(cudaMemcpy(loss.data(),device_loss,loss.size()*4,cudaMemcpyDeviceToHost));double sum=0;for(float a:loss)sum+=a;return sum;};
    double native_loss=forward();puf_mm_tn(&delta,&x,&g,stream);ae_checked(cudaStreamSynchronize(stream));
    nature_bias_backward<<<O,256>>>(delta.data,device_grad+O*F,B,O);
    fp_ref_class<<<1,1>>>(x.data,device_w,weights,labels,ref_delta,ref_loss,B,F,cells,C);
    fp_ref_head_gradient<<<1,64>>>(x.data,ref_delta,ref_grad,B,F,O);
    std::vector<double> reference_grad(P),reference_delta(B*O);double reference_loss;
    ae_checked(cudaMemcpy(native_grad.data(),device_grad,P*4,cudaMemcpyDeviceToHost));
    ae_checked(cudaMemcpy(native_delta.data(),delta.data,B*O*4,cudaMemcpyDeviceToHost));
    ae_checked(cudaMemcpy(reference_grad.data(),ref_grad,P*8,cudaMemcpyDeviceToHost));
    ae_checked(cudaMemcpy(reference_delta.data(),ref_delta,B*O*8,cudaMemcpyDeviceToHost));
    ae_checked(cudaMemcpy(&reference_loss,ref_loss,8,cudaMemcpyDeviceToHost));
    ffc_require(fabs(native_loss-reference_loss)<2e-6,"probe weighted cross entropy");
    double worst=ae_compare(native_grad,reference_grad,"probe head gradients");ae_compare(native_delta,reference_delta,"probe logit gradients");
    int selected[]={0,3,7,13,20,24,27,29};double fd_worst=0;
    for(int i:selected) {
        float value=hw[i],plus=value+.001f,minus=value-.001f;
        ae_checked(cudaMemcpy(device_w+i,&plus,4,cudaMemcpyHostToDevice));double hi=forward();
        ae_checked(cudaMemcpy(device_w+i,&minus,4,cudaMemcpyHostToDevice));double lo=forward();
        ae_checked(cudaMemcpy(device_w+i,&value,4,cudaMemcpyHostToDevice));
        double finite=(hi-lo)/(plus-minus),error=fabs(finite-native_grad[i]);fd_worst=fmax(fd_worst,error);
        ffc_require(error<1e-4+.02*fabs(native_grad[i]),"probe selected finite difference");
    }
    // Normalization uses training IDs only, never the held-out outlier row.
    int hi[]={0,1};int* ids;float *mean,*scale;
    ae_checked(cudaMalloc(&ids,8));ae_checked(cudaMemcpy(ids,hi,8,cudaMemcpyHostToDevice));
    ae_checked(cudaMalloc(&mean,F*4));ae_checked(cudaMalloc(&scale,F*4));fp_normalization<<<1,32>>>(x.data,ids,2,mean,scale,F);
    float hm[F],hs[F];ae_checked(cudaMemcpy(hm,mean,F*4,cudaMemcpyDeviceToHost));ae_checked(cudaMemcpy(hs,scale,F*4,cudaMemcpyDeviceToHost));
    for(int f=0;f<F;f++){double expected=(hx[f]+hx[F+f])/2.,sigma=fabs(hx[f]-hx[F+f])/2.;
        ffc_require(fabs(hm[f]-expected)<1e-6 && fabs(hs[f]-1/fmax(sigma,1e-6))<1e-4,"train-only feature normalization");}

    // Full 640x480 foreground/background weights and inactive validation tail.
    AEStream sample;sample.H=OBS_HEIGHT;sample.W=OBS_WIDTH;sample.skins={0,4};sample.labels_host.resize(2*FP_CELLS);
    for(int b=0;b<2;b++)for(int c=0;c<FP_CELLS;c++)sample.labels_host[b*FP_CELLS+c]=c%FP_CLASSES;
    forage_weights(&sample,2,1,.1f);
    std::vector<float> pixel_weights(2*OBS_HEIGHT*OBS_WIDTH);
    ae_checked(cudaMemcpy(pixel_weights.data(),sample.pixel_weights,pixel_weights.size()*4,cudaMemcpyDeviceToHost));
    for(int b=0;b<2;b++) {
        double totals[FP_CLASSES+1]={};
        for(int y=0;y<OBS_HEIGHT;y++)for(int xx=0;xx<OBS_WIDTH;xx++) {
            int kind=sample.labels_host[b*FP_CELLS+y/FORAGE_CELL*FORAGE_WIDTH+xx/FORAGE_CELL];
            int p=y%FORAGE_CELL*FORAGE_CELL+xx%FORAGE_CELL;
            int region=forage_foreground(sample.skins[b],kind,p)?kind:FP_CLASSES;
            totals[region]+=pixel_weights[b*OBS_HEIGHT*OBS_WIDTH+y*OBS_WIDTH+xx]/(OBS_HEIGHT*(double)OBS_WIDTH);
        }
        for(int c=1;c<FP_CLASSES;c++)ffc_require(fabs(totals[c]-1./(FP_CLASSES-1))<1e-6,"per-class foreground normalization");
        ffc_require(fabs(totals[FP_CLASSES]-.1)<1e-6,"separate background normalization");
    }
    int n=2*OBS_SIZE,active=OBS_SIZE;float *target,*prediction,*dg,*parts,*sums;
    ae_checked(cudaMalloc(&target,(long)n*4));ae_checked(cudaMalloc(&prediction,(long)n*4));ae_checked(cudaMalloc(&dg,(long)n*4));
    ae_checked(cudaMalloc(&parts,512*4));ae_checked(cudaMalloc(&sums,8));
    std::vector<float> zero(n),one(n,1),weighted_grad(n);std::vector<double> weighted_truth(n);
    ae_checked(cudaMemcpy(target,zero.data(),(long)n*4,cudaMemcpyHostToDevice));ae_checked(cudaMemcpy(prediction,one.data(),(long)n*4,cudaMemcpyHostToDevice));
    double* expected;ae_checked(cudaMalloc(&expected,(long)n*8));
    ae_loss<<<256,256>>>(prediction,target,dg,parts,n,active,OBS_HEIGHT,OBS_WIDTH,0,sample.pixel_weights);
    muon_sum_sq_reduce<<<1,256>>>(sums,parts+256,256);
    fp_ref_weighted<<<1,1>>>(prediction,target,sample.pixel_weights,expected,ref_loss,n,active);
    float objective;double truth;ae_checked(cudaMemcpy(&objective,sums,4,cudaMemcpyDeviceToHost));ae_checked(cudaMemcpy(&truth,ref_loss,8,cudaMemcpyDeviceToHost));
    ffc_require(fabs(objective-truth)<2e-6 && fabs(truth-1.1)<2e-6,"object-balanced loss reduction");
    ae_checked(cudaMemcpy(weighted_grad.data(),dg,(long)n*4,cudaMemcpyDeviceToHost));
    ae_checked(cudaMemcpy(weighted_truth.data(),expected,(long)n*8,cudaMemcpyDeviceToHost));
    for(int i=0;i<n;i++) {
        double error=fabs(weighted_grad[i]-weighted_truth[i]);
        ffc_require(isfinite(weighted_grad[i]) && error<2e-11+5e-5*fabs(weighted_truth[i]),
            "small object-loss derivative/inactive tail");
        if(i>=active)ffc_require(weighted_grad[i]==0,"inactive tail must be exactly zero");
    }

    // Calibrate the actual region metric with intentionally bad reconstructions.
    int controls=0;
    std::vector<float> target_image(OBS_SIZE),region_scores(FP_CELLS*6);std::vector<unsigned char> tile_labels(FP_CELLS);
    float *atlas,*region;unsigned char *device_labels,*mask,*mark;
    ae_checked(cudaMalloc(&atlas,FP_CLASSES*FORAGE_CELL*FORAGE_CELL*3*4));ae_checked(cudaMalloc(&region,FP_CELLS*6*4));
    ae_checked(cudaMalloc(&device_labels,FP_CELLS));ae_checked(cudaMalloc(&mask,FP_CLASSES*FORAGE_CELL*FORAGE_CELL));ae_checked(cudaMalloc(&mark,FP_CLASSES*FORAGE_CELL*FORAGE_CELL));
    for(int skin=0;skin<FORAGE_REPRESENTATIONS;skin++) {
        forage_make_tiles(skin);auto masks=forage_mask_atlas(skin),marks=fp_marks(skin);
        for(int cell=0;cell<FP_CELLS;cell++) {
            tile_labels[cell]=cell%FP_CLASSES;
            for(int p=0;p<FORAGE_CELL*FORAGE_CELL;p++)for(int c=0;c<3;c++) {
                int at=((cell/FORAGE_WIDTH*FORAGE_CELL+p/FORAGE_CELL)*OBS_WIDTH+cell%FORAGE_WIDTH*FORAGE_CELL+p%FORAGE_CELL)*3+c;
                target_image[at]=forage_tiles[skin][tile_labels[cell]][p*3+c];
            }
        }
        ae_checked(cudaMemcpy(target,target_image.data(),(long)OBS_SIZE*4,cudaMemcpyHostToDevice));
        ae_checked(cudaMemcpy(atlas,forage_tiles[skin],FP_CLASSES*FORAGE_CELL*FORAGE_CELL*3*4,cudaMemcpyHostToDevice));
        ae_checked(cudaMemcpy(device_labels,tile_labels.data(),FP_CELLS,cudaMemcpyHostToDevice));
        ae_checked(cudaMemcpy(mask,masks.data(),masks.size(),cudaMemcpyHostToDevice));ae_checked(cudaMemcpy(mark,marks.data(),marks.size(),cudaMemcpyHostToDevice));
        for(int mode=0;mode<4;mode++) {
            ae_checked(cudaMemcpy(prediction,target,(long)OBS_SIZE*4,cudaMemcpyDeviceToDevice));
            fp_corrupt<<<grid_size(OBS_SIZE),BLOCK_SIZE>>>(prediction,atlas,device_labels,mode);
            fp_regions<<<FP_CELLS,256>>>(prediction,target,device_labels,mask,mark,region);
            ae_checked(cudaMemcpy(region_scores.data(),region,region_scores.size()*4,cudaMemcpyDeviceToHost));
            for(int cell=0;cell<FP_CELLS;cell++) {
                int kind=tile_labels[cell],replacement=kind;
                if(mode==1)replacement=FORAGE_EMPTY;
                if((mode==2||mode==3)&&(kind==FORAGE_APPLE||kind==FORAGE_SPOTTED_APPLE))
                    replacement=mode==2?FORAGE_EMPTY:kind==FORAGE_APPLE?FORAGE_SPOTTED_APPLE:FORAGE_APPLE;
                double expected[6]={};
                for(int p=0;p<FORAGE_CELL*FORAGE_CELL;p++) {
                    double error=0;
                    for(int c=0;c<3;c++){double d=(double)forage_tiles[skin][replacement][p*3+c]-forage_tiles[skin][kind][p*3+c];error+=d*d/3;}
                    int fg=masks[kind*FORAGE_CELL*FORAGE_CELL+p];expected[fg?0:2]+=error;expected[fg?1:3]++;
                    if(marks[kind*FORAGE_CELL*FORAGE_CELL+p]){expected[4]+=error;expected[5]++;}
                }
                for(int j=0;j<6;j++)ffc_require(fabs(region_scores[cell*6+j]-expected[j])<1e-6+2e-7*fabs(expected[j]),
                    "independent region-error/count calculation");
            }
            FPRegions metric;metric.add(region_scores.data(),tile_labels.data());
            if(mode==0)ffc_require(metric.global_sum==0,"identity reconstruction must be exact");
            else ffc_require(metric.object_mse[FORAGE_APPLE]>0 && metric.object_mse[FORAGE_SPOTTED_APPLE]>0,"deleted/swapped apple must fail");
            if(mode==3)ffc_require(metric.mark_mse[FORAGE_APPLE]>0 && metric.mark_mse[FORAGE_SPOTTED_APPLE]>0,"lookalike marks must fail");
            controls++;
        }
    }
    printf("{\"status\":\"passed\",\"head_gradient_error\":%.17g,\"finite_difference_error\":%.17g,\"finite_difference_probes\":8,\"region_controls\":%d,\"object_weight_sum\":1.1,\"inactive_tail_checked\":true,\"train_only_normalization\":true}\n",worst,fd_worst,controls);
    return 0;
}
