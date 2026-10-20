// Optional native CUDA reconstruction trainer and headless frame collector.
#define main ae_unused_puffer_main
#include "src/pufferl.cu"
#undef main
#include "autoencoder.cuh"
#include <algorithm>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <sys/stat.h>

static void ae_checked(cudaError_t error) { ffc_require(error == cudaSuccess, cudaGetErrorString(error)); }

static double ae_now() {
    struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec+ts.tv_nsec*1e-9;
}

static uint64_t ae_random(uint64_t* state) {
    uint64_t x = (*state += UINT64_C(0x9e3779b97f4a7c15));
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static int ae_collect(int argc, char** argv) {
    ffc_require(argc == 10, "collect CONFIG FRESH_FILE SEED ACTION_SEED SKIN MAX_STEPS CADENCE MAX_FRAMES");
    Ini ini = {}; puf_ini_load_file(&ini, argv[2]);
    Dict* kwargs = puf_ini_section(&ini, "env", 0);
    dict_set(kwargs, "representation", atoi(argv[6]));
    dict_set(kwargs, "representation_mode", 0);
    Env* env = (Env*)calloc(1, sizeof(Env));
    std::vector<float> frame(OBS_SIZE);
    float action = 0, reward = 0, terminal = 0;
    env->rng = strtoul(argv[4], NULL, 10);
    env->agents[0].observations = frame.data(); env->agents[0].actions = &action;
    env->agents[0].rewards = &reward; env->agents[0].terminals = &terminal;
    puf_init(env, kwargs); puf_reset(env);
    uint64_t policy_rng = strtoull(argv[5], NULL, 10);
    int sizes[] = ACT_SIZES, steps = atoi(argv[7]), cadence = atoi(argv[8]), maximum = atoi(argv[9]);
    ffc_require(steps > 0 && steps <= 65536 && cadence > 0 && maximum > 0, "collection bounds");
    FILE* file = fopen(argv[3], "wx"); ffc_require(file, "fresh collection file");
    int frames = 0, decisions = 0;
    for (int step = 0; step < steps && frames < maximum; step++) {
        if (step % cadence == 0) {
            for (float v : frame) ffc_require(isfinite(v) && v >= 0 && v <= 1, "invalid frame RGB");
            ffc_require(fwrite(frame.data(), sizeof(float), frame.size(), file) == frame.size(), "write frame");
            frames++;
        }
        action = ae_random(&policy_rng) % sizes[0];
        puf_step(env); decisions++;
        if (terminal) break; // Do not record an automatic reset under this episode.
    }
    ffc_require(!fclose(file) && frames > 0, "close frame shard");
    printf("{\"frames\":%d,\"decisions\":%d,\"shape_hwc\":[%d,%d,3],\"terminal\":%s,\"cuda_execution\":false}\n",
        frames, decisions, OBS_HEIGHT, OBS_WIDTH, terminal ? "true" : "false");
    puf_close(env); free(env); puf_ini_free(&ini); return 0;
}

struct AEFrame { std::string path; long frame; };
struct AEStream {
    std::string name;
    int H, W;
    std::vector<AEFrame> training, validation;
    std::vector<long> order;
    long epoch = -1;
    FFCWeights* encoder;
    FFCActivations encoder_acts = {};
    AEDecoder decoder;
    AEActivations decoder_acts = {};
    Allocator parameters = {}, gradients = {}, activations = {};
    Prec input;
    float* host;
};

static std::vector<AEStream*> ae_index(const char* path) {
    std::ifstream file(path); ffc_require(file.good(), "open dataset index");
    std::string line; std::getline(file, line); ffc_require(line == "fullframe-autoencoder-index-v1", "dataset index schema");
    std::map<std::string, AEStream*> streams;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::vector<std::string> parts; size_t start = 0, end;
        while ((end = line.find('\t', start)) != std::string::npos) { parts.push_back(line.substr(start, end-start)); start=end+1; }
        parts.push_back(line.substr(start)); ffc_require(parts.size() == 6, "dataset index columns");
        int H = std::stoi(parts[2]), W = std::stoi(parts[3]), count = std::stoi(parts[4]);
        ffc_require(H >= 144 && W >= 144 && H <= 1024 && W <= 1024 && count > 0, "dataset shape/count");
        if (!streams.count(parts[1])) {
            AEStream* stream = new AEStream; stream->name=parts[1];stream->H=H;stream->W=W;
            streams[parts[1]]=stream;
        }
        AEStream* stream = streams[parts[1]];
        ffc_require(stream->H == H && stream->W == W, "stream raw shape changed");
        ffc_require(parts[0] == "train" || parts[0] == "validation", "dataset split");
        auto& target = parts[0] == "train" ? stream->training : stream->validation;
        for (int i=0;i<count;i++) target.push_back({parts[5],i});
    }
    std::vector<AEStream*> result;
    for (auto& pair : streams) {
        AEStream* stream=pair.second;
        ffc_require(!stream->training.empty() && !stream->validation.empty(), "every stream needs both splits");
        result.push_back(stream);
    }
    ffc_require(!result.empty() && result.size() <= 8, "dataset streams");return result;
}

static void ae_load(AEStream* s, int B, long cursor, bool validation, uint64_t seed, int stream_id) {
    auto& frames = validation ? s->validation : s->training;
    int active = validation ? min((long)B, (long)frames.size()-cursor) : B;
    ffc_require(active > 0, "nonempty batch");
    long pixels = (long)s->H*s->W*3;
    for (int b=0;b<B;b++) {
        long index;
        if (validation) index=cursor+min(b,active-1);
        else {
            long epoch=(cursor+b)/frames.size();
            if (s->epoch != epoch) {
                s->order.resize(frames.size());for (long i=0;i<(long)frames.size();i++) s->order[i]=i;
                uint64_t rng=seed ^ ((uint64_t)stream_id << 32) ^ epoch;
                for (long i=frames.size()-1;i>0;i--) std::swap(s->order[i],s->order[ae_random(&rng)%(i+1)]);
                s->epoch=epoch;
            }
            index=s->order[(cursor+b)%frames.size()];
        }
        const AEFrame& frame=frames[index];
        FILE* f=fopen(frame.path.c_str(),"rb");ffc_require(f,"open frame shard");
        ffc_require(!fseeko(f,frame.frame*pixels*4,SEEK_SET)
            && fread(s->host+b*pixels,4,pixels,f)==(size_t)pixels && !fclose(f),"read frame shard");
        for (long i=0;i<pixels;i++) ffc_require(isfinite(s->host[b*pixels+i])
            && s->host[b*pixels+i]>=0 && s->host[b*pixels+i]<=1,"invalid training pixels");
    }
    ae_checked(cudaMemcpy(s->input.data,s->host,pixels*B*4,cudaMemcpyHostToDevice));
}

static void ae_write(const char* path, const void* data, size_t bytes) {
    FILE* f=fopen(path,"wx");ffc_require(f && fwrite(data,1,bytes,f)==bytes && !fclose(f),"fresh output write");
}

struct AEStateHeader { uint64_t magic, version, parameters, step, seed; };

static void ae_checkpoint(const char* folder, long step, uint64_t seed, Allocator* params, float* m, float* v) {
    char path[4096],tmp[4096];snprintf(path,sizeof(path),"%s/step-%012ld.state",folder,step);
    snprintf(tmp,sizeof(tmp),"%s.tmp",path);
    AEStateHeader header={UINT64_C(0x31454146465550),1,(uint64_t)params->total_elems,(uint64_t)step,seed};
    size_t bytes=params->total_elems*4;
    std::vector<float> data(params->total_elems*3);
    ae_checked(cudaMemcpy(data.data(),params->mem,bytes,cudaMemcpyDeviceToHost));
    ae_checked(cudaMemcpy(data.data()+params->total_elems,m,bytes,cudaMemcpyDeviceToHost));
    ae_checked(cudaMemcpy(data.data()+2*params->total_elems,v,bytes,cudaMemcpyDeviceToHost));
    for (float x:data) ffc_require(isfinite(x),"nonfinite autoencoder state");
    FILE* f=fopen(tmp,"wx");ffc_require(f && fwrite(&header,sizeof(header),1,f)==1
        && fwrite(data.data(),4,data.size(),f)==data.size() && !fclose(f),"write autoencoder checkpoint");
    struct stat st;ffc_require(stat(path,&st)!=0 && !rename(tmp,path),"publish fresh autoencoder checkpoint");
    printf("{\"checkpoint\":\"step-%012ld.state\",\"step\":%ld,\"parameters\":%ld}\n",step,step,params->total_elems);fflush(stdout);
}

static int ae_train(const char* recipe, const char* index, const char* folder, const char* resume) {
    double start=ae_now();
    Ini ini={};puf_ini_load_file(&ini,recipe);Dict* cfg=puf_ini_section(&ini,"autoencoder",0);
    int B=dict_get(cfg,"batch"), target=dict_get(cfg,"steps"), cadence=dict_get(cfg,"checkpoint_interval");
    float lr=dict_get(cfg,"learning_rate"), edge=dict_get(cfg,"edge_weight"), clip=dict_get(cfg,"max_grad_norm");
    int seconds=dict_get(cfg,"max_seconds");uint64_t seed=dict_get(cfg,"seed");
    ffc_require(B>=1 && B<=16 && target>0 && target<=10000000 && cadence>0 && lr>0 && lr<=.1f
        && edge>=0 && edge<=100 && clip>0 && seconds>0 && seconds<=43200,"pretraining bounds");
    auto streams=ae_index(index);cublas_init_handle();cudaStream_t stream;ae_checked(cudaStreamCreate(&stream));
    Allocator *params=NULL,*grads=NULL;
    long allocated=0;
    for (AEStream* s:streams) {
        s->encoder=ffc_graph(0,s->H,s->W);s->decoder=ae_decoder(s->H,s->W);
        ffc_params(s->encoder,&s->parameters);nature_reg_params(&s->decoder.net,&s->parameters);
        ffc_train(s->encoder,&s->encoder_acts,&s->activations,&s->gradients,B);
        ae_register(&s->decoder,&s->decoder_acts,&s->activations,&s->gradients,B);
        ffc_require(s->parameters.total_bytes==s->parameters.total_elems*4
            && s->gradients.total_bytes==s->parameters.total_bytes,"flat state layout");
        allocated+=s->activations.total_bytes+(long)B*s->H*s->W*3*4;
        ffc_require(allocated<24L*1024*1024*1024,"activation storage budget");
        if (!params) {params=&s->parameters;grads=&s->gradients;alloc_create(params);alloc_create(grads);}
        else {ae_bind(&s->parameters,params);ae_bind(&s->gradients,grads);}
        alloc_create(&s->activations);
        s->input={.shape={B,s->H*s->W*3}};
        ae_checked(cudaMalloc(&s->input.data,numel(s->input.shape)*4));
        ae_checked(cudaMallocHost(&s->host,numel(s->input.shape)*4));
    }
    float *m,*v,*parts,*stats,*norm;
    ae_checked(cudaMalloc(&m,params->total_bytes));ae_checked(cudaMalloc(&v,params->total_bytes));
    ae_checked(cudaMemset(m,0,params->total_bytes));ae_checked(cudaMemset(v,0,params->total_bytes));
    ae_checked(cudaMalloc(&parts,512*4));ae_checked(cudaMalloc(&stats,2*4));ae_checked(cudaMalloc(&norm,4));
    ulong init_seed=seed;ffc_init(streams[0]->encoder,&init_seed,stream);nature_init_weights(&streams[0]->decoder.net,&init_seed,stream);
    // A linear RGB head uses a conservative initial gain; decoder is training-only.
    puf_kaiming_init(&streams[0]->decoder.net.weight[3],.1f,init_seed++,stream);
    ae_checked(cudaStreamSynchronize(stream));
    long step=0;
    if (strcmp(resume,"None")) {
        FILE* f=fopen(resume,"rb");AEStateHeader h;
        ffc_require(f && fread(&h,sizeof(h),1,f)==1 && h.magic==UINT64_C(0x31454146465550)
            && h.version==1 && h.parameters==(uint64_t)params->total_elems && h.seed==seed && h.step<(uint64_t)target,"resume state contract");
        std::vector<float> data(params->total_elems*3);
        ffc_require(fread(data.data(),4,data.size(),f)==data.size() && fgetc(f)==EOF && !fclose(f),"resume state length");
        for (float x:data) ffc_require(isfinite(x),"nonfinite resume state");
        ae_checked(cudaMemcpy(params->mem,data.data(),params->total_bytes,cudaMemcpyHostToDevice));
        ae_checked(cudaMemcpy(m,data.data()+params->total_elems,params->total_bytes,cudaMemcpyHostToDevice));
        ae_checked(cudaMemcpy(v,data.data()+2*params->total_elems,params->total_bytes,cudaMemcpyHostToDevice));step=h.step;
    }
    ae_checkpoint(folder,step,seed,params,m,v);
    auto forward=[&](AEStream* s,int active) {
        Prec z=ffc_forward(s->encoder,&s->encoder_acts,s->input,stream);
        Prec y=ae_forward(&s->decoder,&s->decoder_acts,z,stream);
        ae_loss<<<256,256,0,stream>>>(y.data,s->input.data,s->decoder_acts.net.grad_output[3].data,
            parts,B*s->H*s->W*3,active*s->H*s->W*3,s->H,s->W,edge);
        muon_sum_sq_reduce<<<1,256,0,stream>>>(stats,parts,256);
        muon_sum_sq_reduce<<<1,256,0,stream>>>(stats+1,parts+256,256);
        float loss[2];ae_checked(cudaMemcpyAsync(loss,stats,8,cudaMemcpyDeviceToHost,stream));
        ae_checked(cudaStreamSynchronize(stream));ffc_require(isfinite(loss[0]) && isfinite(loss[1]),"nonfinite reconstruction loss");
        return std::make_pair(loss[0],loss[1]);
    };
    auto validate=[&]() {
        for (size_t i=0;i<streams.size();i++) {
            AEStream* s=streams[i];double raw=0,weighted=0;
            for (long off=0;off<(long)s->validation.size();off+=B) {
                int active=min((long)B,(long)s->validation.size()-off);
                ae_load(s,B,off,true,seed,i);auto loss=forward(s,active);
                raw+=loss.first*active;weighted+=loss.second*active;
            }
            printf("{\"validation\":\"%s\",\"step\":%ld,\"frames\":%zu,\"mse\":%.12g,\"objective\":%.12g}\n",
                s->name.c_str(),step,s->validation.size(),raw/s->validation.size(),weighted/s->validation.size());
        }
        fflush(stdout);
    };
    validate();
    while (step<target && ae_now()-start<seconds) {
        int id=step%streams.size();AEStream* s=streams[id];long cursor=(step/streams.size())*B;
        ae_load(s,B,cursor,false,seed,id);auto loss=forward(s,B);
        Prec dz=ae_backward(&s->decoder,&s->decoder_acts,stream);
        ffc_backward(s->encoder,&s->encoder_acts,dz,stream);
        int blocks=min((int)grid_size(params->total_elems),256);
        muon_sum_sq_partials<<<blocks,256,0,stream>>>(parts,(float*)grads->mem,params->total_elems);
        muon_sum_sq_reduce<<<1,256,0,stream>>>(norm,parts,blocks);
        float sum;ae_checked(cudaMemcpyAsync(&sum,norm,4,cudaMemcpyDeviceToHost,stream));ae_checked(cudaStreamSynchronize(stream));
        ffc_require(isfinite(sum),"nonfinite gradient norm; no optimizer update");
        step++;
        ae_adam<<<grid_size(params->total_elems),BLOCK_SIZE,0,stream>>>((float*)params->mem,(float*)grads->mem,m,v,norm,
            params->total_elems,lr,clip,1-pow(.9,step),1-pow(.999,step));
        ae_checked(cudaStreamSynchronize(stream));
        if (step%cadence==0 || step==target || ae_now()-start>=seconds) {
            printf("{\"training_step\":%ld,\"stream\":\"%s\",\"mse\":%.12g,\"objective\":%.12g,\"seconds\":%.6f}\n",
                step,s->name.c_str(),loss.first,loss.second,ae_now()-start);
            ae_checkpoint(folder,step,seed,params,m,v);validate();
        }
    }
    printf("{\"status\":\"%s\",\"final_step\":%ld,\"target_steps\":%d,\"allocated_activation_bytes\":%ld}\n",
        step==target?"complete":"budget_exhausted",step,target,allocated);return 0;
}

static void ae_manifest(int H,int W) {
    ffc_require(H>=144 && W>=144 && H<=1024 && W<=1024,"manifest shape");
    FFCWeights* w=ffc_graph(0,H,W);Allocator p={};ffc_params(w,&p);
    cJSON* root=cJSON_CreateObject();
    cJSON_AddStringToObject(root,"schema","fullframe-compression-encoder-v1");
    cJSON_AddStringToObject(root,"dtype","float32");cJSON_AddStringToObject(root,"layout","NHWC");
    cJSON_AddStringToObject(root,"range","0,1");cJSON_AddNumberToObject(root,"model",0);
    cJSON_AddNumberToObject(root,"height",H);cJSON_AddNumberToObject(root,"width",W);cJSON_AddNumberToObject(root,"channels",3);
    cJSON_AddStringToObject(root,"activation","relu-zero-derivative-0");
    cJSON_AddStringToObject(root,"area","box4-rowmajor-partial-edge");
    cJSON_AddStringToObject(root,"adaptive","floor-start-ceil-end-4x4");
    cJSON_AddStringToObject(root,"padding","same-trailing-extra-or-nature-valid");
    cJSON_AddNumberToObject(root,"elements",p.total_elems);
    cJSON* layers=cJSON_AddArrayToObject(root,"layers");
    for(int i=0;i<4;i++) {
        NatureLayer l=w->net.layer[i];int values[]={l.ih,l.iw,l.ci,l.co,l.k,l.stride,l.oh,l.ow,w->ph[i],w->pw[i]};
        cJSON_AddItemToArray(layers,cJSON_CreateIntArray(values,10));
    }
    cJSON* tensors=cJSON_AddArrayToObject(root,"tensors");long offset=0;
    for(int i=0;i<p.num_regs;i++) {
        cJSON* t=cJSON_CreateObject();char name[32];snprintf(name,sizeof(name),"%s_%d",i%2?"bias":"weight",i/2);
        cJSON_AddStringToObject(t,"name",name);cJSON_AddNumberToObject(t,"offset",offset);
        long count=numel(p.regs[i].shape);cJSON_AddNumberToObject(t,"elements",count);
        cJSON* shape=cJSON_AddArrayToObject(t,"shape");
        for(int j=0;j<ndim(p.regs[i].shape);j++) cJSON_AddItemToArray(shape,cJSON_CreateNumber(p.regs[i].shape[j]));
        offset+=count;cJSON_AddItemToArray(tensors,t);
    }
    char* json=cJSON_PrintUnformatted(root);puts(json);free(json);cJSON_Delete(root);free(p.regs);free(w);
}

#include "autoencoder_checks.cuh"

int main(int argc,char** argv) {
    if(argc==2 && !strcmp(argv[1],"auxiliary")) return ae_auxiliary();
    if(argc==4 && !strcmp(argv[1],"import-check")) return ae_import_check(argv[2],argv[3]);
    if(argc==6 && !strcmp(argv[1],"math")) return ae_math(atoi(argv[2]),atoi(argv[3]),atoi(argv[4]),argv[5]);
    if(argc==4 && !strcmp(argv[1],"manifest")) {ae_manifest(atoi(argv[2]),atoi(argv[3]));return 0;}
    if (argc==2 && !strcmp(argv[1],"describe")) {
        AEDecoder d=ae_decoder(640,420);Allocator p={};nature_reg_params(&d.net,&p);
        printf("{\"decoder_parameters\":%ld,\"encoder_parameters\":36776,\"latent_values\":128,\"gpu_queried\":false}\n",p.total_elems);return 0;
    }
    if (argc>=2 && !strcmp(argv[1],"collect")) return ae_collect(argc,argv);
    ffc_require(argc==7 && !strcmp(argv[1],"train"),"train CONFIG INDEX FRESH_OUT RESUME RESERVED");
    ffc_require(!strcmp(argv[6],"v1"),"pretraining protocol");
    return ae_train(argv[2],argv[3],argv[4],argv[5]);
}
