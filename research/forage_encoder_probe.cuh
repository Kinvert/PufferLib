// Full-frame frozen-feature diagnostic. No oracle crop enters the encoder.
// One linear head predicts 13 identities at every board location from the same z.
static constexpr int FP_FEATURES=128, FP_CLASSES=FORAGE_TILE_TYPES, FP_CELLS=FORAGE_AREA;

__global__ static void fp_normalization(const float* x,const int* ids,int N,float* mean,float* scale,int F) {
    int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=F)return;
    double m=0,v=0;
    for(int i=0;i<N;i++){double a=x[ids[i]*F+f];double d=a-m;m+=d/(i+1);v+=d*(a-m);}
    mean[f]=m;scale[f]=1/fmax(sqrt(v/N),1e-6);
}

__global__ static void fp_normalize(float* x,const float* mean,const float* scale,int N,int F) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<N*F)x[i]=(x[i]-mean[i%F])*scale[i%F];
}

__global__ static void fp_batch(const float* features,const unsigned char* labels,const int* ids,
        float* x,unsigned char* y,int B,int F,int cells) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<B*F)x[i]=features[ids[i/F]*F+i%F];
    if(i<B*cells)y[i]=labels[ids[i/cells]*cells+i%cells];
}

__global__ static void fp_softmax(float* logits,const float* bias,const unsigned char* target,
        const float* class_weights,float* gradient,float* loss,int B,int cells,int classes) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=B*cells)return;
    int b=i/cells,cell=i%cells,offset=i*classes;float top=-INFINITY,sum=0;
    for(int c=0;c<classes;c++){float v=logits[offset+c]+bias[cell*classes+c];logits[offset+c]=v;top=fmaxf(top,v);}
    for(int c=0;c<classes;c++)sum+=expf(logits[offset+c]-top);
    int truth=target[i];float weight=class_weights[truth];
    loss[i]=weight*(logf(sum)+top-logits[offset+truth])/(B*cells);
    for(int c=0;c<classes;c++)gradient[offset+c]=weight*(expf(logits[offset+c]-top)/sum-(c==truth))/(B*cells);
}

// Three disjoint pixel sets: visible foreground, background, and lookalike marks.
// Each block owns one cell and reduces in a fixed order, without atomics.
__global__ static void fp_regions(const float* prediction,const float* target,
        const unsigned char* labels,const unsigned char* masks,const unsigned char* marks,float* result) {
    __shared__ double values[6][256];
    int cell=blockIdx.x,kind=labels[cell],left=cell%FORAGE_WIDTH*FORAGE_CELL,top=cell/FORAGE_WIDTH*FORAGE_CELL;
    double sums[6]={};
    for(int p=threadIdx.x;p<FORAGE_CELL*FORAGE_CELL;p+=256) {
        int at=((top+p/FORAGE_CELL)*OBS_WIDTH+left+p%FORAGE_CELL)*3;
        double error=0;for(int c=0;c<3;c++){double d=(double)prediction[at+c]-target[at+c];error+=d*d/3;}
        int fg=masks[kind*FORAGE_CELL*FORAGE_CELL+p];
        sums[fg?0:2]+=error;sums[fg?1:3]++;
        if(marks[kind*FORAGE_CELL*FORAGE_CELL+p]){sums[4]+=error;sums[5]++;}
    }
    for(int i=0;i<6;i++)values[i][threadIdx.x]=sums[i];__syncthreads();
    for(int stride=128;stride;stride/=2){if(threadIdx.x<stride)for(int i=0;i<6;i++)values[i][threadIdx.x]+=values[i][threadIdx.x+stride];__syncthreads();}
    if(!threadIdx.x)for(int i=0;i<6;i++)result[cell*6+i]=values[i][0];
}

__global__ static void fp_corrupt(float* image,const float* atlas,const unsigned char* labels,int mode) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=OBS_SIZE)return;
    int c=i%3,x=i/3%OBS_WIDTH,y=i/(3*OBS_WIDTH);
    int cell=y/FORAGE_CELL*FORAGE_WIDTH+x/FORAGE_CELL,kind=labels[cell];
    int local=(y%FORAGE_CELL*FORAGE_CELL+x%FORAGE_CELL)*3+c;
    if(mode==1)image[i]=atlas[local];
    if((mode==2||mode==3)&&(kind==FORAGE_APPLE||kind==FORAGE_SPOTTED_APPLE)) {
        int other=mode==2?FORAGE_EMPTY:kind==FORAGE_APPLE?FORAGE_SPOTTED_APPLE:FORAGE_APPLE;
        image[i]=atlas[other*FORAGE_CELL*FORAGE_CELL*3+local];
    }
}

static std::vector<unsigned char> fp_marks(int skin) {
    std::vector<unsigned char> masks(FP_CLASSES*FORAGE_CELL*FORAGE_CELL);
    int pairs[][2]={{FORAGE_APPLE,FORAGE_SPOTTED_APPLE},{FORAGE_MUSHROOM,FORAGE_TOADSTOOL}};
    forage_make_tiles(skin);
    for(auto& pair:pairs)for(int p=0;p<FORAGE_CELL*FORAGE_CELL;p++) {
        bool different=memcmp(forage_tiles[skin][pair[0]]+3*p,forage_tiles[skin][pair[1]]+3*p,12)!=0;
        for(int kind:pair)masks[kind*FORAGE_CELL*FORAGE_CELL+p]=different;
    }
    return masks;
}

struct FPModel {
    FFCWeights* encoder; FFCActivations encoder_acts={}; AEDecoder decoder; AEActivations decoder_acts={};
    Allocator parameters={},gradients={},activations={}; Prec input; cudaStream_t stream;
    FPModel(const char* state_path) {
        encoder=ffc_graph(0,OBS_HEIGHT,OBS_WIDTH);decoder=ae_decoder(OBS_HEIGHT,OBS_WIDTH);
        ffc_params(encoder,&parameters);nature_reg_params(&decoder.net,&parameters);
        ffc_train(encoder,&encoder_acts,&activations,&gradients,1);ae_register(&decoder,&decoder_acts,&activations,&gradients,1);
        alloc_create(&parameters);alloc_create(&gradients);alloc_create(&activations);
        input={.shape={1,OBS_SIZE}};ae_checked(cudaMalloc(&input.data,(long)OBS_SIZE*4));
        ae_checked(cudaStreamCreate(&stream));ulong seed=7;
        ffc_init(encoder,&seed,stream);nature_init_weights(&decoder.net,&seed,stream);
        puf_kaiming_init(&decoder.net.weight[3],.1f,seed++,stream);ae_checked(cudaStreamSynchronize(stream));
        if(strcmp(state_path,"random")) {
            FILE* f=fopen(state_path,"rb");AEStateHeader h;
            ffc_require(f && fread(&h,sizeof(h),1,f)==1 && h.magic==UINT64_C(0x31454146465550)
                && h.version==1 && h.parameters==(uint64_t)parameters.total_elems && h.seed==7,"probe checkpoint header");
            std::vector<float> all(h.parameters*3);
            ffc_require(fread(all.data(),4,all.size(),f)==all.size() && fgetc(f)==EOF && !fclose(f),"probe checkpoint size");
            for(float v:all)ffc_require(isfinite(v),"probe finite checkpoint");
            ae_checked(cudaMemcpy(parameters.mem,all.data(),parameters.total_bytes,cudaMemcpyHostToDevice));
        }
    }
};

struct FPFrame {int split,skin;const AEFrame* frame;};
struct FPRegions {
    double global_sum=0,global_pixels=0,background_sum=0,background_pixels=0;
    double object_mse[FP_CLASSES]={},object_instances[FP_CLASSES]={},mark_mse[FP_CLASSES]={},mark_instances[FP_CLASSES]={};
    void add(const float* scores,const unsigned char* labels) {
        for(int cell=0;cell<FP_CELLS;cell++) {
            int kind=labels[cell];const float* v=scores+cell*6;
            for(int i=0;i<6;i++)ffc_require(isfinite(v[i])&&v[i]>=0,"finite/nonnegative region metric");
            global_sum+=v[0]+v[2];global_pixels+=v[1]+v[3];background_sum+=v[2];background_pixels+=v[3];
            if(v[1]){object_mse[kind]+=v[0]/v[1];object_instances[kind]++;}
            if(v[5]){mark_mse[kind]+=v[4]/v[5];mark_instances[kind]++;}
        }
    }
    cJSON* json() {
        cJSON* root=cJSON_CreateObject();cJSON_AddNumberToObject(root,"global_mse",global_sum/global_pixels);
        cJSON_AddNumberToObject(root,"background_mse",background_sum/background_pixels);
        cJSON* objects=cJSON_AddArrayToObject(root,"objects");
        for(int c=0;c<FP_CLASSES;c++) {
            cJSON* item=cJSON_CreateObject();cJSON_AddNumberToObject(item,"class",c);cJSON_AddStringToObject(item,"name",FORAGE_NAMES[c]);
            cJSON_AddNumberToObject(item,"instances",object_instances[c]);
            if(object_instances[c])cJSON_AddNumberToObject(item,"mse",object_mse[c]/object_instances[c]);
            cJSON_AddNumberToObject(item,"mark_instances",mark_instances[c]);
            if(mark_instances[c])cJSON_AddNumberToObject(item,"mark_mse",mark_mse[c]/mark_instances[c]);
            cJSON_AddItemToArray(objects,item);
        }
        return root;
    }
};

static void fp_json_write(const char* path,cJSON* value) {
    char* text=cJSON_PrintUnformatted(value);ae_write(path,text,strlen(text));free(text);cJSON_Delete(value);
}

static int forage_probe(const char* unused_config,const char* index,const char* checkpoint,
        const char* recipe,const char* folder) {
    auto streams=ae_index(index);ffc_require(streams.size()==1,"Forage probe requires one image stream");AEStream* s=streams[0];
    ffc_require(s->H==OBS_HEIGHT && s->W==OBS_WIDTH && !s->test.empty(),"probe requires full frames and a separate test split");
    Ini ini={};puf_ini_load_file(&ini,recipe);Dict* cfg=puf_ini_section(&ini,"probe",0);
    int B=dict_get(cfg,"batch"),steps=dict_get(cfg,"steps"),seconds=dict_get(cfg,"max_seconds");
    float lr=dict_get(cfg,"learning_rate");
    ffc_require(B>=1&&B<=64&&steps>=1&&steps<=100000&&seconds>=1&&seconds<=3600&&lr>0&&lr<=.1f,"probe budget");
    bool constant=!strcmp(checkpoint,"constant");
    double started=ae_now();cublas_init_handle();FPModel model(constant?"random":checkpoint);
    std::vector<FPFrame> frames;std::vector<int> train_ids;
    for(int split=0;split<3;split++) {
        auto& source=split==0?s->training:split==1?s->validation:s->test;
        for(auto& frame:source){if(!split)train_ids.push_back(frames.size());frames.push_back({split,frame.skin,&frame});}
    }
    int N=frames.size();ffc_require(N<=8192 && train_ids.size()>=2,"probe frame cap/train count");
    std::vector<float> features((long)N*FP_FEATURES),image(OBS_SIZE),scores(FP_CELLS*6);
    std::vector<unsigned char> targets((long)N*FP_CELLS);
    std::vector<float> before(model.parameters.total_elems),after(before.size());
    ae_checked(cudaMemcpy(before.data(),model.parameters.mem,model.parameters.total_bytes,cudaMemcpyDeviceToHost));
    unsigned char *labels,*masks,*marks;float* region;
    ae_checked(cudaMalloc(&labels,FP_CELLS));ae_checked(cudaMalloc(&masks,FP_CLASSES*FORAGE_CELL*FORAGE_CELL));
    ae_checked(cudaMalloc(&marks,FP_CLASSES*FORAGE_CELL*FORAGE_CELL));ae_checked(cudaMalloc(&region,FP_CELLS*6*4));
    FPRegions metrics[3][FORAGE_REPRESENTATIONS];
    std::string out=folder;
    FILE* mapping=fopen((out+"/frames.tsv").c_str(),"wx");ffc_require(mapping,"fresh frame mapping");
    for(int i=0;i<N;i++) {
        ffc_require(ae_now()-started<seconds,"feature extraction exceeded probe cap");
        const FPFrame& row=frames[i];const AEFrame& frame=*row.frame;
        FILE* f=fopen(frame.path.c_str(),"rb");
        ffc_require(f&&!fseeko(f,frame.frame*(long)OBS_SIZE*4,SEEK_SET)&&fread(image.data(),4,OBS_SIZE,f)==OBS_SIZE&&!fclose(f),"probe image read");
        for(float v:image)ffc_require(isfinite(v)&&v>=0&&v<=1,"probe input RGB");
        FILE* lab=fopen(frame.labels.c_str(),"rb");
        ffc_require(lab&&!fseeko(lab,frame.frame*FP_CELLS,SEEK_SET)&&fread(targets.data()+i*FP_CELLS,1,FP_CELLS,lab)==FP_CELLS&&!fclose(lab),"probe labels read");
        for(int c=0;c<FP_CELLS;c++)ffc_require(targets[i*FP_CELLS+c]<FP_CLASSES,"probe target identity");
        forage_make_tiles(row.skin);
        for(int cell=0;cell<FP_CELLS;cell++)for(int y=0;y<FORAGE_CELL;y++) {
            int at=((cell/FORAGE_WIDTH*FORAGE_CELL+y)*OBS_WIDTH+cell%FORAGE_WIDTH*FORAGE_CELL)*3;
            const float* tile=forage_tiles[row.skin][targets[i*FP_CELLS+cell]]+y*FORAGE_CELL*3;
            ffc_require(!memcmp(image.data()+at,tile,FORAGE_CELL*3*sizeof(float)),"label does not describe actual visible pixels");
        }
        auto mask=forage_mask_atlas(row.skin),mark=fp_marks(row.skin);
        ae_checked(cudaMemcpy(labels,targets.data()+i*FP_CELLS,FP_CELLS,cudaMemcpyHostToDevice));
        ae_checked(cudaMemcpy(masks,mask.data(),mask.size(),cudaMemcpyHostToDevice));
        ae_checked(cudaMemcpy(marks,mark.data(),mark.size(),cudaMemcpyHostToDevice));
        ae_checked(cudaMemcpy(model.input.data,image.data(),(long)OBS_SIZE*4,cudaMemcpyHostToDevice));
        Prec z=ffc_forward(model.encoder,&model.encoder_acts,model.input,model.stream);
        Prec reconstruction=ae_forward(&model.decoder,&model.decoder_acts,z,model.stream);
        ae_checked(cudaStreamSynchronize(model.stream));
        ae_checked(cudaMemcpy(features.data()+i*FP_FEATURES,z.data,FP_FEATURES*4,cudaMemcpyDeviceToHost));
        fp_regions<<<FP_CELLS,256>>>(reconstruction.data,model.input.data,labels,masks,marks,region);
        ae_checked(cudaMemcpy(scores.data(),region,scores.size()*4,cudaMemcpyDeviceToHost));
        metrics[row.split][row.skin].add(scores.data(),targets.data()+i*FP_CELLS);
        fprintf(mapping,"%d\t%d\t%d\t%s\t%ld\n",i,row.split,row.skin,frame.path.c_str(),frame.frame);
    }
    ffc_require(!fclose(mapping),"close frame mapping");
    for(float x:features)ffc_require(isfinite(x),"finite frozen features");
    if(constant)std::fill(features.begin(),features.end(),0);
    ae_write((out+"/features.f32").c_str(),features.data(),features.size()*4);
    ae_write((out+"/labels.u8").c_str(),targets.data(),targets.size());
    double feature_seconds=ae_now()-started;
    const int F=FP_FEATURES,O=FP_CELLS*FP_CLASSES,P=O*(F+1);
    float *cache,*normal_mean,*normal_scale,*cw,*weights,*gradient,*m,*v,*parts,*norm,*loss;
    int *ids,*training_ids;unsigned char *all_labels,*batch_labels;
    ae_checked(cudaMalloc(&cache,features.size()*4));ae_checked(cudaMemcpy(cache,features.data(),features.size()*4,cudaMemcpyHostToDevice));
    ae_checked(cudaMalloc(&all_labels,targets.size()));ae_checked(cudaMemcpy(all_labels,targets.data(),targets.size(),cudaMemcpyHostToDevice));
    ae_checked(cudaMalloc(&training_ids,train_ids.size()*4));ae_checked(cudaMemcpy(training_ids,train_ids.data(),train_ids.size()*4,cudaMemcpyHostToDevice));
    ae_checked(cudaMalloc(&normal_mean,F*4));ae_checked(cudaMalloc(&normal_scale,F*4));
    fp_normalization<<<1,128>>>(cache,training_ids,train_ids.size(),normal_mean,normal_scale,F);
    fp_normalize<<<grid_size(N*F),BLOCK_SIZE>>>(cache,normal_mean,normal_scale,N,F);
    double class_counts[FP_CLASSES]={};float class_weights[FP_CLASSES];
    for(int id:train_ids)for(int c=0;c<FP_CELLS;c++)class_counts[targets[id*FP_CELLS+c]]++;
    for(int c=0;c<FP_CLASSES;c++){ffc_require(class_counts[c]>0,"missing probe training class");class_weights[c]=(float)(train_ids.size()*FP_CELLS)/(FP_CLASSES*class_counts[c]);}
    ae_checked(cudaMalloc(&cw,FP_CLASSES*4));ae_checked(cudaMemcpy(cw,class_weights,FP_CLASSES*4,cudaMemcpyHostToDevice));
    ae_checked(cudaMalloc(&weights,P*4));ae_checked(cudaMalloc(&gradient,P*4));ae_checked(cudaMalloc(&m,P*4));ae_checked(cudaMalloc(&v,P*4));
    ae_checked(cudaMemset(weights,0,P*4));ae_checked(cudaMemset(m,0,P*4));ae_checked(cudaMemset(v,0,P*4));
    ae_checked(cudaMalloc(&parts,256*4));ae_checked(cudaMalloc(&norm,4));ae_checked(cudaMalloc(&loss,B*FP_CELLS*4));
    ae_checked(cudaMalloc(&ids,B*4));ae_checked(cudaMalloc(&batch_labels,B*FP_CELLS));
    Prec x={.shape={B,F}},w={.data=weights,.shape={O,F}},g_w={.data=gradient,.shape={O,F}};
    Prec logits={.shape={B,O}},g_logits={.shape={B,O}};
    ae_checked(cudaMalloc(&x.data,B*F*4));ae_checked(cudaMalloc(&logits.data,B*O*4));ae_checked(cudaMalloc(&g_logits.data,B*O*4));
    std::vector<int> order=train_ids,batch(B);long last_epoch=-1;int completed=0;
    for(int step=0;step<steps && ae_now()-started<seconds;step++) {
        for(int b=0;b<B;b++) {
            long position=(long)step*B+b,epoch=position/train_ids.size();
            if(epoch!=last_epoch){order=train_ids;uint64_t rng=7^epoch;for(long j=order.size()-1;j>0;j--)std::swap(order[j],order[ae_random(&rng)%(j+1)]);last_epoch=epoch;}
            batch[b]=order[position%order.size()];
        }
        ae_checked(cudaMemcpy(ids,batch.data(),B*4,cudaMemcpyHostToDevice));
        fp_batch<<<grid_size(B*max(F,FP_CELLS)),BLOCK_SIZE>>>(cache,all_labels,ids,x.data,batch_labels,B,F,FP_CELLS);
        puf_mm(&x,&w,&logits,model.stream);ae_checked(cudaStreamSynchronize(model.stream));
        fp_softmax<<<grid_size(B*FP_CELLS),BLOCK_SIZE>>>(logits.data,weights+O*F,batch_labels,cw,g_logits.data,loss,B,FP_CELLS,FP_CLASSES);
        ae_checked(cudaDeviceSynchronize());puf_mm_tn(&g_logits,&x,&g_w,model.stream);ae_checked(cudaStreamSynchronize(model.stream));
        nature_bias_backward<<<O,256>>>(g_logits.data,gradient+O*F,B,O);
        int blocks=min((int)grid_size(P),256);
        muon_sum_sq_partials<<<blocks,256>>>(parts,gradient,P);muon_sum_sq_reduce<<<1,256>>>(norm,parts,blocks);
        float sum_square;ae_checked(cudaMemcpy(&sum_square,norm,4,cudaMemcpyDeviceToHost));
        ffc_require(isfinite(sum_square),"nonfinite probe gradient; no optimizer update");
        ae_adam<<<grid_size(P),BLOCK_SIZE>>>(weights,gradient,m,v,norm,P,lr,1,1-pow(.9,step+1),1-pow(.999,step+1));
        ae_checked(cudaDeviceSynchronize());completed=step+1;
    }
    ffc_require(completed>0,"no probe updates before cap");
    std::vector<float> head(P),stats(2*F);
    ae_checked(cudaMemcpy(head.data(),weights,P*4,cudaMemcpyDeviceToHost));
    for(float value:head)ffc_require(isfinite(value),"finite probe head");
    ae_checked(cudaMemcpy(stats.data(),normal_mean,F*4,cudaMemcpyDeviceToHost));
    ae_checked(cudaMemcpy(stats.data()+F,normal_scale,F*4,cudaMemcpyDeviceToHost));
    ae_write((out+"/head.f32").c_str(),head.data(),head.size()*4);ae_write((out+"/normalization.f32").c_str(),stats.data(),stats.size()*4);
    // Use B=1 views for final inference; the same buffers have enough space.
    x.shape[0]=logits.shape[0]=1;
    std::vector<float> prediction(O);std::vector<int> majority(FP_CELLS),counts(FP_CELLS*FP_CLASSES);
    for(int id:train_ids)for(int cell=0;cell<FP_CELLS;cell++)counts[cell*FP_CLASSES+targets[id*FP_CELLS+cell]]++;
    for(int cell=0;cell<FP_CELLS;cell++)majority[cell]=std::max_element(counts.begin()+cell*FP_CLASSES,counts.begin()+(cell+1)*FP_CLASSES)-(counts.begin()+cell*FP_CLASSES);
    FILE* csv=fopen((out+"/predictions.csv").c_str(),"wx");ffc_require(csv,"fresh predictions");
    fprintf(csv,"frame,split,skin,cell,truth,prediction,majority,apple_pair_prediction,mushroom_pair_prediction\n");
    for(int i=0;i<N;i++) {
        ae_checked(cudaMemcpy(x.data,cache+i*F,F*4,cudaMemcpyDeviceToDevice));puf_mm(&x,&w,&logits,model.stream);
        ae_checked(cudaStreamSynchronize(model.stream));ae_checked(cudaMemcpy(prediction.data(),logits.data,O*4,cudaMemcpyDeviceToHost));
        for(int cell=0;cell<FP_CELLS;cell++) {
            int at=cell*FP_CLASSES,best=0;
            for(int c=0;c<FP_CLASSES;c++){prediction[at+c]+=head[O*F+at+c];ffc_require(isfinite(prediction[at+c]),"finite probe prediction");}
            for(int c=1;c<FP_CLASSES;c++)if(prediction[at+c]>prediction[at+best])best=c;
            int apple=prediction[at+FORAGE_APPLE]>=prediction[at+FORAGE_SPOTTED_APPLE]?FORAGE_APPLE:FORAGE_SPOTTED_APPLE;
            int mushroom=prediction[at+FORAGE_MUSHROOM]>=prediction[at+FORAGE_TOADSTOOL]?FORAGE_MUSHROOM:FORAGE_TOADSTOOL;
            fprintf(csv,"%d,%d,%d,%d,%d,%d,%d,%d,%d\n",i,frames[i].split,frames[i].skin,cell,targets[i*FP_CELLS+cell],best,majority[cell],apple,mushroom);
        }
    }
    ffc_require(!fclose(csv),"close predictions");
    ae_checked(cudaMemcpy(after.data(),model.parameters.mem,model.parameters.total_bytes,cudaMemcpyDeviceToHost));
    ffc_require(!memcmp(before.data(),after.data(),model.parameters.total_bytes),"frozen encoder/decoder mutated by probe");
    cJSON* result=cJSON_CreateObject();cJSON_AddStringToObject(result,"status",completed==steps?"complete":"budget_exhausted");
    char parameter_hash[65];ffc_sha256((const unsigned char*)before.data(),model.parameters.total_bytes,parameter_hash);
    cJSON_AddStringToObject(result,"parameter_sha256",parameter_hash);
    cJSON_AddNumberToObject(result,"frames",N);cJSON_AddNumberToObject(result,"probe_updates",completed);cJSON_AddNumberToObject(result,"requested_updates",steps);
    cJSON_AddNumberToObject(result,"probe_parameters",P);cJSON_AddNumberToObject(result,"feature_seconds",feature_seconds);
    cJSON_AddNumberToObject(result,"process_body_seconds",ae_now()-started);cJSON_AddBoolToObject(result,"encoder_decoder_unchanged",true);
    cJSON_AddBoolToObject(result,"constant_feature_control",constant);
    cJSON* regions=cJSON_AddArrayToObject(result,"reconstruction");
    for(int split=0;split<3;split++)for(int skin=0;skin<FORAGE_REPRESENTATIONS;skin++)if(metrics[split][skin].global_pixels) {
        cJSON* item=metrics[split][skin].json();cJSON_AddNumberToObject(item,"split",split);cJSON_AddNumberToObject(item,"skin",skin);cJSON_AddItemToArray(regions,item);
    }
    fp_json_write((out+"/native.json").c_str(),result);return 0;
}

#include "forage_probe_checks.cuh"
