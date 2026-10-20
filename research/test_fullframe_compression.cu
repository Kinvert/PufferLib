// Actual native callbacks versus independent direct-sum CUDA reference.
#define main ffc_training_main
#include "../src/pufferl.cu"
#undef main
#include "fullframe_compression_reference.cu"
#include <sys/stat.h>

static double ffc_compare(const float* a,const double* r,long n,const char* name,
        double atol=6e-5,double rtol=8e-4) {
    double worst=0;long bad=0;
    for(long i=0;i<n;i++) {
        double e=fabs(a[i]-r[i]);worst=fmax(worst,e);
        if(!isfinite(a[i])||!isfinite(r[i])||e>atol+rtol*fabs(r[i])) {
            if(bad<3) fprintf(stderr,"%s[%ld] actual=%.10g reference=%.10g error=%.10g\n",name,i,a[i],r[i],e);
            bad++;
        }
    }
    ffc_require(!bad,name);return worst;
}

static void ffc_write(const char* folder,const char* name,const void* data,size_t bytes) {
    char path[4096];snprintf(path,sizeof(path),"%s/%s",folder,name);
    FILE* f=fopen(path,"wx");ffc_require(f&&fwrite(data,1,bytes,f)==bytes&&!fclose(f),"write fixture");
}

__global__ static void ffc_test_loss(const float* y,const float* up,double* loss,int n) {
    if(threadIdx.x||blockIdx.x) return;
    double sum=0;
    for(int i=0;i<n;i++) sum+=(double)y[i]*up[i];
    *loss=sum;
}

static int ffc_host() {
    const char* inputs[]={"","abc","The quick brown fox jumps over the lazy dog"};
    const char* digests[]={"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592"};
    for(int i=0;i<3;i++) {
        char digest[65];ffc_sha256((const unsigned char*)inputs[i],strlen(inputs[i]),digest);
        ffc_require(!strcmp(digest,digests[i]),"SHA256 known-answer");
    }
    for(int model=0;model<3;model++) {
        FFCWeights* w=ffc_graph(model,640,420);Allocator p={};ffc_params(w,&p);
        long expected=model==0?36776:36608;
        if(model==2) expected=(32*(8*8*3+1)+64*(4*4*32+1)+64*(3*3*64+1)+128*(16*9*64+1));
        ffc_require(p.total_elems==expected&&p.total_bytes==expected*4,"host independent parameter count");
        free(p.regs);free(w);
    }
    puts("{\"status\":\"passed\",\"gpu_queried\":false,\"sha256_known_answers\":3,\"graphs\":3}");return 0;
}

static int ffc_import_check(const char* config,const char* directory,const char* folder) {
    Ini ini={};puf_ini_load_file(&ini,config);Dict* policy=puf_ini_section(&ini,"policy",0);
    Arch arch=build_arch(OBS_SIZE,128,1,2,false,16,policy);
    Allocator a={},b={};Weights wa=weights_create(&arch,&a),wb=weights_create(&arch,&b);
    Allocator enc={};arch.encoder.reg_params(wa.encoder,&enc);
    alloc_create(&a);alloc_create(&b);cublas_init_handle();cudaStream_t stream;
    checked(cudaStreamCreate(&stream));uint64_t seed_a=7,seed_b=7;
    weights_init(&arch,wa,&seed_a,stream);
    ((FFCWeights*)wb.encoder)->initial=strdup(directory);
    weights_init(&arch,wb,&seed_b,stream);checked(cudaStreamSynchronize(stream));
    std::vector<float> baseline(a.total_elems),imported(b.total_elems);
    checked(cudaMemcpy(baseline.data(),a.mem,a.total_bytes,cudaMemcpyDeviceToHost));
    checked(cudaMemcpy(imported.data(),b.mem,b.total_bytes,cudaMemcpyDeviceToHost));
    ffc_require(seed_a==seed_b&&!memcmp(baseline.data()+enc.total_elems,imported.data()+enc.total_elems,
        (a.total_elems-enc.total_elems)*4),"import changed core/head or initialization RNG");
    char path[4096];snprintf(path,sizeof(path),"%s/encoder.bin",directory);size_t bytes;
    char* raw=ffc_read_file(path,&bytes);
    ffc_require(bytes==enc.total_elems*4&&!memcmp(imported.data(),raw,bytes),"imported encoder differs");
    ffc_require(!mkdir(folder,0700),"fresh import-check output");
    ffc_write(folder,"scratch.f32",baseline.data(),a.total_bytes);
    ffc_write(folder,"imported.f32",imported.data(),b.total_bytes);
    puts("{\"status\":\"passed\",\"core_head_bytes_identical\":true,\"initialization_rng_identical\":true}");return 0;
}

// Synthetic RGB only: channel ramps, translated edges/dots and clipped values.
// These fixtures contain no game state, object labels or renderer dependencies.
static float ffc_fixture_pixel(int b,int y,int x,int c,int H,int W,int fixture) {
    if(fixture==1) return 0;
    uint32_t hash=((b*H+y)*W+x)*3+c+1;
    hash*=2246822519u;hash^=hash>>13;
    float value=(hash%1024)/1024.f;
    if(y<H/3) {
        value=c==0?x/(float)(W-1):c==1?y/(float)(H-1):value;
    } else if(y<2*H/3) {
        value=c==0?(x>=W/2+b*3):c==1?(y>=H/2+b*2):value;
    } else {
        int dx=x-(W/3+b*7),dy=y-(5*H/6-b*3);
        if(dx*dx+dy*dy<=16) value=c==0?1.25f:c==1?-.25f:.5f;
    }
    return fminf(1,fmaxf(0,value));
}

int main(int argc,char** argv) {
    if(argc==2&&!strcmp(argv[1],"--host")) return ffc_host();
    if(argc==5&&!strcmp(argv[1],"--import-check")) return ffc_import_check(argv[2],argv[3],argv[4]);
    ffc_require(argc==8,"checks HEIGHT WIDTH BATCH MODEL FIXTURE FRESH_OUT MODE(math|init)");
    int H=atoi(argv[1]),W=atoi(argv[2]),B=atoi(argv[3]),model=atoi(argv[4]),fixture=atoi(argv[5]);
    const char* folder=argv[6];
    bool diagnostic=!strcmp(argv[7],"diagnostic");
    ffc_require(diagnostic||!strcmp(argv[7],"math"),"unknown harness mode");
    ffc_require(H>=144&&W>=144&&B>=1&&B<=16&&model>=0&&model<=2&&fixture>=0&&fixture<=2,"fixture bounds");
    ffc_require(!mkdir(folder,0700),"fresh fixture folder");
    cublas_init_handle();cudaStream_t stream;checked(cudaStreamCreate(&stream));
    FFCWeights* w=ffc_graph(model,H,W);w->input_grad=true;
    FFCActivations acts={};Allocator parameters={},activations={},gradients={};
    ffc_params(w,&parameters);ffc_train(w,&acts,&activations,&gradients,B);
    alloc_create(&parameters);alloc_create(&activations);alloc_create(&gradients);
    long P=parameters.total_elems,N=(long)B*H*W*3;
    FFCReference ref(model,H,W,B);
    ffc_require(P==ref.parameters&&P==gradients.total_elems&&parameters.total_bytes==P*4,"reference registration");
    std::vector<float> hw(P),hx(N),hu(B*128),y(B*128),dw(P),dx(N);
    std::vector<double> rw(P),rx(N),ru(B*128),ry(B*128),rdw(P),rdx(N);
    for(long i=0;i<P;i++) {
        uint32_t h=(i+1)*2654435761u;h^=h>>16;
        hw[i]=fixture==1?0:fixture==0?.01f+(h%23)*.0001f:((int)(h%127)-63)*.002f;
        rw[i]=hw[i];
    }
    for(long i=0;i<N;i++)
        rx[i]=hx[i]=ffc_fixture_pixel(i/(H*W*3),i/(W*3)%H,i/3%W,i%3,H,W,fixture);
    for(int i=0;i<B*128;i++) hu[i]=(fixture==2?(i%9-4):1+i%7)*.01f,ru[i]=hu[i];
    Prec input={.shape={B,H*W*3}},up={.shape={B,128}};
    checked(cudaMalloc(&input.data,N*4));checked(cudaMalloc(&up.data,hu.size()*4));
    checked(cudaMemcpy(input.data,hx.data(),N*4,cudaMemcpyHostToDevice));
    checked(cudaMemcpy(up.data,hu.data(),hu.size()*4,cudaMemcpyHostToDevice));
    checked(cudaMemcpy(parameters.mem,hw.data(),P*4,cudaMemcpyHostToDevice));
    Prec output;
    auto native=[&](){output=ffc_forward(w,&acts,input,stream);ffc_backward(w,&acts,up,stream);};
    native();checked(cudaStreamSynchronize(stream));checked(cudaGetLastError());
    checked(cudaMemcpy(y.data(),output.data,y.size()*4,cudaMemcpyDeviceToHost));
    checked(cudaMemcpy(dw.data(),gradients.mem,P*4,cudaMemcpyDeviceToHost));
    checked(cudaMemcpy(dx.data(),acts.grad_input.data,N*4,cudaMemcpyDeviceToHost));
    FFCActivations actor={};Allocator actor_acts={};
    ffc_rollout(w,&actor,&actor_acts,B);alloc_create(&actor_acts);
    Prec actor_output=ffc_forward(w,&actor,input,stream);checked(cudaStreamSynchronize(stream));
    std::vector<float> actor_values(y.size());
    checked(cudaMemcpy(actor_values.data(),actor_output.data,y.size()*4,cudaMemcpyDeviceToHost));
    ffc_require(!memcmp(y.data(),actor_values.data(),y.size()*4),"actor/train forward disagreement");
    checked(cudaFree(actor_acts.mem));free(actor_acts.regs);
    double* obs=ref.storage.add(N),*weights=ref.storage.add(P),*g=ref.storage.add(hu.size()),*gradient=ref.storage.add(P);
    checked(cudaMemcpy(obs,rx.data(),N*8,cudaMemcpyHostToDevice));
    checked(cudaMemcpy(weights,rw.data(),P*8,cudaMemcpyHostToDevice));
    checked(cudaMemcpy(g,ru.data(),ru.size()*8,cudaMemcpyHostToDevice));
    ref.run(obs,weights,g,gradient);
    checked(cudaMemcpy(ry.data(),ref.nodes.back().y,ry.size()*8,cudaMemcpyDeviceToHost));
    checked(cudaMemcpy(rdw.data(),gradient,P*8,cudaMemcpyDeviceToHost));
    checked(cudaMemcpy(rdx.data(),ref.nodes[0].g,N*8,cudaMemcpyDeviceToHost));
    if(diagnostic) {
        ffc_write(folder,"weights.f32",hw.data(),P*4);ffc_write(folder,"input.f32",hx.data(),N*4);
        ffc_write(folder,"upstream.f32",hu.data(),hu.size()*4);
        ffc_write(folder,"gradient-native.f32",dw.data(),P*4);
        ffc_write(folder,"gradient-double.f64",rdw.data(),P*8);
        for(auto& n:ref.nodes) if(n.op==3) {
            long count=(long)B*n.h*n.w*n.c;
            std::vector<float> actual(count);std::vector<double> expected(count);
            checked(cudaMemcpy(actual.data(),acts.net.output[n.conv].data,count*4,cudaMemcpyDeviceToHost));
            checked(cudaMemcpy(expected.data(),n.y,count*8,cudaMemcpyDeviceToHost));
            long branches=0;double maximum=0;
            for(long i=0;i<count;i++) {branches+=(actual[i]>0)!=(expected[i]>0);maximum=fmax(maximum,fabs(actual[i]-expected[i]));}
            printf("layer=%d branch_disagreements=%ld max_output_error=%.17g\n",n.conv,branches,maximum);
            char file[64];snprintf(file,sizeof(file),"layer-%d-native.f32",n.conv);ffc_write(folder,file,actual.data(),count*4);
            snprintf(file,sizeof(file),"layer-%d-double.f64",n.conv);ffc_write(folder,file,expected.data(),count*8);
        }
        const float* masks[4];for(int i=0;i<4;i++) masks[i]=acts.net.output[i].data;
        ref.run(obs,weights,g,gradient,masks);
        checked(cudaMemcpy(rdw.data(),gradient,P*8,cudaMemcpyDeviceToHost));
        ffc_write(folder,"gradient-actual-branches.f64",rdw.data(),P*8);
        double error=ffc_compare(dw.data(),rdw.data(),P,"diagnostic actual-branch gradient");
        printf("DIAGNOSTIC ONLY actual_branch_max_error=%.17g\n",error);return 0;
    }
    double ey=ffc_compare(y.data(),ry.data(),y.size(),"forward");
    long branch_disagreements=0;double branch_distance=0,layer_error=0;
    for(auto& n:ref.nodes) if(n.op==3) {
        long count=(long)B*n.h*n.w*n.c;
        std::vector<float> actual(count);std::vector<double> expected(count);
        checked(cudaMemcpy(actual.data(),acts.net.output[n.conv].data,count*4,cudaMemcpyDeviceToHost));
        checked(cudaMemcpy(expected.data(),n.y,count*8,cudaMemcpyDeviceToHost));
        layer_error=fmax(layer_error,ffc_compare(actual.data(),expected.data(),count,"layer-forward"));
        for(long i=0;i<count;i++) if((actual[i]>0)!=(expected[i]>0)) {
            branch_disagreements++;branch_distance=fmax(branch_distance,fmax(actual[i],expected[i]));
        }
    }
    ffc_require(!branch_disagreements||fixture==2,"smooth/zero fixture changed ReLU branches");
    ffc_require(branch_distance<=2e-6,"ReLU disagreement beyond declared rounding neighborhood");
    // A derivative through a different ReLU branch is not the native derivative.
    // Preserve the full-double gradient as diagnostic evidence in those cases.
    ffc_write(folder,"gradient-double-branches.f64",rdw.data(),P*8);
    double ew=0,ex=0;
    for(long i=0;i<P;i++) ew=fmax(ew,fabs(dw[i]-rdw[i]));
    for(long i=0;i<N;i++) ex=fmax(ex,fabs(dx[i]-rdx[i]));
    if(!branch_disagreements) {
        ffc_compare(dw.data(),rdw.data(),P,"parameter-gradient");
        ffc_compare(dx.data(),rdx.data(),N,"input-gradient");
    }
    const float* branches[4];for(int i=0;i<4;i++) branches[i]=acts.net.output[i].data;
    ref.run(obs,weights,g,gradient,branches);
    checked(cudaMemcpy(rdw.data(),gradient,P*8,cudaMemcpyDeviceToHost));
    checked(cudaMemcpy(rdx.data(),ref.nodes[0].g,N*8,cudaMemcpyDeviceToHost));
    double eb=ffc_compare(dw.data(),rdw.data(),P,"actual-branch-parameter-gradient");
    double ebi=ffc_compare(dx.data(),rdx.data(),N,"actual-branch-input-gradient");
    double area_error=0;
    if(model) {
        auto& area=ref.nodes[1];long count=(long)B*area.h*area.w*3;
        std::vector<float> a(count);std::vector<double> r(count);
        checked(cudaMemcpy(a.data(),acts.reduced.data,count*4,cudaMemcpyDeviceToHost));
        checked(cudaMemcpy(r.data(),area.y,count*8,cudaMemcpyDeviceToHost));
        area_error=ffc_compare(a.data(),r.data(),count,"area",2e-6,2e-6);
    }
    int finite_difference_probes=0;double finite_difference_error=0;
    if(H==160&&W==192&&B==1&&fixture==0) {
        double* device_loss;checked(cudaMalloc(&device_loss,sizeof(double)));
        auto loss=[&](){Prec v=ffc_forward(w,&acts,input,stream);
            ffc_test_loss<<<1,1,0,stream>>>(v.data,up.data,device_loss,B*128);
            double value;checked(cudaMemcpyAsync(&value,device_loss,8,cudaMemcpyDeviceToHost,stream));
            checked(cudaStreamSynchronize(stream));return value;};
        long offset=0;
        for(int i=0;i<parameters.num_regs;i++) {
            long count=numel(parameters.regs[i].shape);
            for(long at:{0L,count/2,count-1}) {
                long index=offset+at;float minus=hw[index]-.0002f,plus=hw[index]+.0002f;
                checked(cudaMemcpyAsync((float*)parameters.mem+index,&minus,4,cudaMemcpyHostToDevice,stream));
                double low=loss();
                checked(cudaMemcpyAsync((float*)parameters.mem+index,&plus,4,cudaMemcpyHostToDevice,stream));
                double high=loss();
                checked(cudaMemcpyAsync((float*)parameters.mem+index,&hw[index],4,cudaMemcpyHostToDevice,stream));
                double fd=(high-low)/(plus-minus),error=fabs(fd-dw[index]);
                finite_difference_error=fmax(finite_difference_error,error);
                ffc_require(error<=3e-3+3e-2*fabs(dw[index]),"GPU finite difference parameter probe");
                finite_difference_probes++;
            }
            offset+=count;
        }
        checked(cudaFree(device_loss));native();checked(cudaStreamSynchronize(stream));
    }
    cudaGraph_t graph;cudaGraphExec_t execution;
    checked(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));native();
    checked(cudaStreamEndCapture(stream,&graph));checked(cudaGraphInstantiate(&execution,graph,0));
    for(int repeat=0;repeat<3;repeat++) {
        if(repeat) checked(cudaGraphLaunch(execution,stream));else native();
        checked(cudaStreamSynchronize(stream));
        std::vector<float> ay(y.size()),ag(P),ax(N),aw(P);
        checked(cudaMemcpy(ay.data(),output.data,ay.size()*4,cudaMemcpyDeviceToHost));
        checked(cudaMemcpy(ag.data(),gradients.mem,P*4,cudaMemcpyDeviceToHost));
        checked(cudaMemcpy(ax.data(),acts.grad_input.data,N*4,cudaMemcpyDeviceToHost));
        checked(cudaMemcpy(aw.data(),parameters.mem,P*4,cudaMemcpyDeviceToHost));
        ffc_require(!memcmp(ay.data(),y.data(),y.size()*4)&&!memcmp(ag.data(),dw.data(),P*4)
            &&!memcmp(ax.data(),dx.data(),N*4)&&!memcmp(aw.data(),hw.data(),P*4),"eager/graph repeat or mutation");
    }
    ffc_write(folder,"output.f32",y.data(),y.size()*4);
    ffc_write(folder,"gradient.f32",dw.data(),P*4);
    ffc_write(folder,"output-reference.f64",ry.data(),ry.size()*8);
    ffc_write(folder,"gradient-reference.f64",rdw.data(),P*8);
    if(H<200) {ffc_write(folder,"input-gradient.f32",dx.data(),N*4);ffc_write(folder,"input-gradient-reference.f64",rdx.data(),N*8);}
    char digest[65];ffc_sha256((unsigned char*)dx.data(),N*4,digest);
    char result[2048];snprintf(result,sizeof(result),
        "{\"status\":\"passed\",\"height\":%d,\"width\":%d,\"batch\":%d,\"model\":%d,\"fixture\":%d,"
        "\"numerical_protocol\":\"ffc-native-branches-v2\",\"input_fixture\":\"rgb-ramps-translated-edges-dots-clipped-v1\",\"parameters\":%ld,\"max_output_error\":%.17g,\"max_gradient_error\":%.17g,\"max_input_error\":%.17g,"
        "\"actual_branch_gradient_error\":%.17g,\"actual_branch_input_error\":%.17g,\"area_error\":%.17g,"
        "\"branch_disagreements\":%ld,\"branch_distance\":%.17g,\"max_layer_forward_error\":%.17g,"
        "\"input_gradient_sha256\":\"%s\",\"actor_train_equal\":true,\"eager_graph_repeats\":3,\"finite_difference_probes\":%d,"
        "\"finite_difference_max_error\":%.17g,\"allocated_native_bytes\":%ld}\n",
        H,W,B,model,fixture,P,ey,ew,ex,eb,ebi,area_error,branch_disagreements,branch_distance,layer_error,digest,finite_difference_probes,
        finite_difference_error,parameters.total_bytes+activations.total_bytes+gradients.total_bytes);
    ffc_write(folder,"result.json",result,strlen(result));puts(result);return 0;
}
