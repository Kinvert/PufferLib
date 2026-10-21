// Native spatial reconstruction report and encoder/CNN boundary checks.
#ifdef AE_FORAGE
static void sc_ppm(const std::string& path, const std::vector<float>& values, int H=OBS_HEIGHT, int W=OBS_WIDTH) {
    FILE* f=fopen(path.c_str(),"wx");ffc_require(f,"fresh reconstruction image");
    fprintf(f,"P6\n%d %d\n255\n",W,H);
    std::vector<unsigned char> rgb(values.size());
    for(size_t i=0;i<values.size();i++)rgb[i]=lroundf(fminf(1,fmaxf(0,values[i]))*255);
    ffc_require(fwrite(rgb.data(),1,rgb.size(),f)==rgb.size()&&!fclose(f),"write reconstruction image");
}

static int sc_reconstruct(const char* recipe,const char* index,const char* checkpoint,const char* folder,const char* artifact) {
    ffc_require(!mkdir(folder,0700),"fresh spatial reconstruction output");
    Ini ini={};puf_ini_load_file(&ini,recipe);Dict* cfg=puf_ini_section(&ini,"autoencoder",0);
    int stride=dict_get(cfg,"latent_stride"),channels=dict_get(cfg,"latent_channels");
    cublas_init_handle();cudaStream_t stream;ae_checked(cudaStreamCreate(&stream));
    auto streams=ae_index(index);ffc_require(streams.size()==1,"one Forage evaluation stream");
    AEStream* s=streams[0];ffc_require(s->H==OBS_HEIGHT&&s->W==OBS_WIDTH,"Forage evaluation raw shape");
    s->encoder=sc_graph(s->H,s->W,stride,channels);s->decoder=ae_spatial_decoder(s->H,s->W,stride,channels);
    ffc_params(s->encoder,&s->parameters);nature_reg_params(&s->decoder.net,&s->parameters);
    ffc_train(s->encoder,&s->encoder_acts,&s->activations,&s->gradients,1);
    ae_register(&s->decoder,&s->decoder_acts,&s->activations,&s->gradients,1);
    alloc_create(&s->parameters);alloc_create(&s->gradients);alloc_create(&s->activations);
    s->input={.shape={1,OBS_SIZE}};ae_checked(cudaMalloc(&s->input.data,(long)OBS_SIZE*4));
    ae_checked(cudaMallocHost(&s->host,(long)OBS_SIZE*4));
    ulong seed=7;ffc_init(s->encoder,&seed,stream);nature_init_weights(&s->decoder.net,&seed,stream);
    puf_kaiming_init(&s->decoder.net.weight[2],.1f,seed++,stream);ae_checked(cudaStreamSynchronize(stream));
    if(strcmp(checkpoint,"random")) {
        FILE* f=fopen(checkpoint,"rb");AEStateHeader h;
        ffc_require(f&&fread(&h,sizeof(h),1,f)==1&&h.magic==UINT64_C(0x31454146465550)
            &&h.version==2&&h.parameters==(uint64_t)s->parameters.total_elems&&h.seed==7,"spatial reconstruction checkpoint");
        std::vector<float> data(h.parameters*3);
        ffc_require(fread(data.data(),4,data.size(),f)==data.size()&&fgetc(f)==EOF&&!fclose(f),"spatial state length");
        for(float v:data)ffc_require(isfinite(v),"finite spatial checkpoint");
        ae_checked(cudaMemcpy(s->parameters.mem,data.data(),s->parameters.total_bytes,cudaMemcpyHostToDevice));
    }
    FFCWeights* imported=sc_graph(s->H,s->W,stride,channels);FFCActivations ia={};Allocator ip={},act={};
    ffc_params(imported,&ip);alloc_create(&ip);ffc_rollout(imported,&ia,&act,1);alloc_create(&act);
    bool verify=strcmp(artifact,"None");if(verify)sc_import(imported,artifact,stream);
    long Z=s->decoder.latent_values;std::vector<float> z(Z),zz(Z),prediction(OBS_SIZE),regions(FORAGE_AREA*6);
    std::vector<float> before(s->parameters.total_elems),after(before.size());
    ae_checked(cudaMemcpy(before.data(),s->parameters.mem,s->parameters.total_bytes,cudaMemcpyDeviceToHost));
    unsigned char *labels,*masks,*marks;float* sums;
    ae_checked(cudaMalloc(&labels,FORAGE_AREA));ae_checked(cudaMalloc(&masks,FORAGE_TILE_TYPES*FORAGE_CELL*FORAGE_CELL));
    ae_checked(cudaMalloc(&marks,FORAGE_TILE_TYPES*FORAGE_CELL*FORAGE_CELL));ae_checked(cudaMalloc(&sums,regions.size()*4));
    std::string base=folder;FILE* csv=fopen((base+"/regions.csv").c_str(),"wx");FILE* features=fopen((base+"/features.csv").c_str(),"wx");
    ffc_require(csv&&features,"fresh spatial metrics");
    fprintf(csv,"split,frame,skin,cell,kind,foreground_sum,foreground_count,background_sum,background_count,mark_sum,mark_count\n");
    fprintf(features,"split,frame,skin,sha256\n");
    int frames=0,previews[3][FORAGE_REPRESENTATIONS]={};
    for(int split=0;split<3;split++) {
        auto& data=split==0?s->training:split==1?s->validation:s->test;
        for(size_t i=0;i<data.size();i++) {
            // Evaluation always reads sequential frame IDs, including train; never shuffled.
            AEFrame frame=data[i];
            auto saved=s->validation;s->validation={frame};ae_load(s,1,0,1,7,0);s->validation=std::move(saved);
            int skin=s->skins[0];forage_make_tiles(skin);
            for(int cell=0;cell<FORAGE_AREA;cell++) {
                int kind=s->labels_host[cell],left=cell%FORAGE_WIDTH*FORAGE_CELL,top=cell/FORAGE_WIDTH*FORAGE_CELL;
                for(int y=0;y<FORAGE_CELL;y++)for(int x=0;x<FORAGE_CELL;x++)for(int c=0;c<3;c++)
                    ffc_require(s->host[((top+y)*OBS_WIDTH+left+x)*3+c]
                        ==forage_tiles[skin][kind][(y*FORAGE_CELL+x)*3+c],"evaluation label/image alignment");
            }
            auto mask=forage_mask_atlas(skin),mark=fp_marks(skin);
            ae_checked(cudaMemcpy(labels,s->labels_host.data(),FORAGE_AREA,cudaMemcpyHostToDevice));
            ae_checked(cudaMemcpy(masks,mask.data(),mask.size(),cudaMemcpyHostToDevice));
            ae_checked(cudaMemcpy(marks,mark.data(),mark.size(),cudaMemcpyHostToDevice));
            Prec latent=ffc_forward(s->encoder,&s->encoder_acts,s->input,stream);
            Prec y=ae_forward(&s->decoder,&s->decoder_acts,latent,stream);ae_checked(cudaStreamSynchronize(stream));
            ae_checked(cudaMemcpy(z.data(),latent.data,Z*4,cudaMemcpyDeviceToHost));
            for(float v:z)ffc_require(isfinite(v),"finite spatial features");
            if(verify) {
                Prec v=ffc_forward(imported,&ia,s->input,stream);ae_checked(cudaStreamSynchronize(stream));
                ae_checked(cudaMemcpy(zz.data(),v.data,Z*4,cudaMemcpyDeviceToHost));
                ffc_require(!memcmp(z.data(),zz.data(),Z*4),"encoder export/reload features differ");
            }
            fp_regions<<<FORAGE_AREA,256>>>(y.data,s->input.data,labels,masks,marks,sums);
            ae_checked(cudaMemcpy(regions.data(),sums,regions.size()*4,cudaMemcpyDeviceToHost));
            ae_checked(cudaMemcpy(prediction.data(),y.data,(long)OBS_SIZE*4,cudaMemcpyDeviceToHost));
            for(float v:prediction)ffc_require(isfinite(v),"finite reconstruction");
            for(int cell=0;cell<FORAGE_AREA;cell++) {
                fprintf(csv,"%d,%zu,%d,%d,%d",split,i,skin,cell,s->labels_host[cell]);
                for(int j=0;j<6;j++)fprintf(csv,",%.12g",regions[cell*6+j]);fputc('\n',csv);
            }
            char hash[65];ffc_sha256((unsigned char*)z.data(),Z*4,hash);fprintf(features,"%d,%zu,%d,%s\n",split,i,skin,hash);
            if(!previews[split][skin]++) {
                std::string name=base+"/split-"+std::to_string(split)+"-skin-"+std::to_string(skin);
                sc_ppm(name+"-original.ppm",std::vector<float>(s->host,s->host+OBS_SIZE));sc_ppm(name+"-reconstruction.ppm",prediction);
                if(split==2) {
                    // First visible instance of each class, chosen without its
                    // reconstruction score. Expose fine markings in the viewer.
                    int seen[FORAGE_TILE_TYPES]={};
                    for(int cell=0;cell<FORAGE_AREA;cell++) {
                        int kind=s->labels_host[cell];if(!kind||seen[kind]++)continue;
                        std::vector<float> original(FORAGE_CELL*FORAGE_CELL*3),recon(original.size());
                        int left=cell%FORAGE_WIDTH*FORAGE_CELL,top=cell/FORAGE_WIDTH*FORAGE_CELL;
                        for(int y=0;y<FORAGE_CELL;y++)for(int x=0;x<FORAGE_CELL;x++)for(int c=0;c<3;c++) {
                            int at=((top+y)*OBS_WIDTH+left+x)*3+c,local=(y*FORAGE_CELL+x)*3+c;
                            original[local]=s->host[at];recon[local]=prediction[at];
                        }
                        std::string crop=name+"-kind-"+std::to_string(kind)+"-cell-"+std::to_string(cell);
                        sc_ppm(crop+"-original.ppm",original,FORAGE_CELL,FORAGE_CELL);
                        sc_ppm(crop+"-reconstruction.ppm",recon,FORAGE_CELL,FORAGE_CELL);
                    }
                }
            }
            frames++;
        }
    }
    ffc_require(!fclose(csv)&&!fclose(features),"close spatial metrics");
    ae_checked(cudaMemcpy(after.data(),s->parameters.mem,s->parameters.total_bytes,cudaMemcpyDeviceToHost));
    ffc_require(before==after,"reconstruction evaluation mutated parameters");
    printf("{\"status\":\"passed\",\"frames\":%d,\"latent_values\":%ld,\"export_features_exact\":%s,\"parameters_unchanged\":true}\n",frames,Z,verify?"true":"false");
    return 0;
}
#endif

static int sc_policy_check(const char* recipe,const char* artifact,int frozen) {
    Ini ini={};puf_ini_load_file(&ini,recipe);Dict* cfg=puf_ini_section(&ini,"policy",0);
    dict_set(cfg,"compression_frozen",frozen);dict_set_str(cfg,"compression_init",artifact);
    Encoder enc={.in_dim=OBS_SIZE,.out_dim=128};create_sc_policy_encoder(&enc,cfg);
    SCPolicy* w=(SCPolicy*)enc.create_weights(&enc);SCPolicyActs a={};Allocator p={},acts={},g={};
    enc.reg_params(w,&p);enc.reg_train(w,&a,&acts,&g,3);alloc_create(&p);alloc_create(&acts);alloc_create(&g);
    cublas_init_handle();cudaStream_t stream;ae_checked(cudaStreamCreate(&stream));ulong seed=7;enc.init_weights(w,&seed,stream);
    // Smooth-positive consumer fixture; prefix retains the actual imported weights.
    for(int i=0;i<4;i++) {
        std::vector<float> v(numel(w->cnn->net.weight[i].shape),.005f);
        ae_checked(cudaMemcpy(w->cnn->net.weight[i].data,v.data(),v.size()*4,cudaMemcpyHostToDevice));
        v.assign(numel(w->cnn->net.bias[i].shape),.01f);
        ae_checked(cudaMemcpy(w->cnn->net.bias[i].data,v.data(),v.size()*4,cudaMemcpyHostToDevice));
    }
    Prec x={.shape={3,OBS_SIZE}},up={.shape={3,128}};
    ae_checked(cudaMalloc(&x.data,(long)3*OBS_SIZE*4));ae_checked(cudaMalloc(&up.data,3*128*4));
    std::vector<float> input((long)3*OBS_SIZE),dy(384,.01f),out(384),grad(p.total_elems),again(p.total_elems);
    for(size_t i=0;i<input.size();i++)input[i]=(i%251)/250.f;
    ae_checked(cudaMemcpy(x.data,input.data(),input.size()*4,cudaMemcpyHostToDevice));
    ae_checked(cudaMemcpy(up.data,dy.data(),dy.size()*4,cudaMemcpyHostToDevice));
    long P=0;for(int i=0;i<3;i++)P+=numel(w->prefix->net.weight[i].shape)+numel(w->prefix->net.bias[i].shape);
    std::vector<float> before(P),after(P);long off=0;
    for(int i=0;i<3;i++)for(Prec* t:{&w->prefix->net.weight[i],&w->prefix->net.bias[i]}) {
        ae_checked(cudaMemcpy(before.data()+off,t->data,numel(t->shape)*4,cudaMemcpyDeviceToHost));off+=numel(t->shape);
    }
    auto run=[&]() {Prec y=enc.forward(w,&a,x,stream);enc.backward(w,&a,up,stream);return y;};
    Prec y=run();ae_checked(cudaStreamSynchronize(stream));
    ae_checked(cudaMemcpy(out.data(),y.data,384*4,cudaMemcpyDeviceToHost));ae_checked(cudaMemcpy(grad.data(),g.mem,g.total_bytes,cudaMemcpyDeviceToHost));
    for(float v:grad)ffc_require(isfinite(v),"finite CNN-consumer derivatives");
    // Independent CUDA oracle for the entire consumer, using the actual frozen
    // prefix features as inputs. No production patches/GEMM in this reference.
    NatureLayer last=w->prefix->net.layer[2];long Z=(long)3*last.oh*last.ow*last.co;
    FFCReference reference(0,last.oh,last.ow,3,0,16,last.co,true);Storage memory;
    std::vector<float> native_z(Z),consumer_weights(reference.parameters);
    ae_checked(cudaMemcpy(native_z.data(),a.prefix.net.output[2].data,Z*4,cudaMemcpyDeviceToHost));
    long copied=0;for(int i=0;i<4;i++)for(Prec* t:{&w->cnn->net.weight[i],&w->cnn->net.bias[i]}) {
        ae_checked(cudaMemcpy(consumer_weights.data()+copied,t->data,numel(t->shape)*4,cudaMemcpyDeviceToHost));copied+=numel(t->shape);
    }
    std::vector<double> rz(native_z.begin(),native_z.end()),rw(consumer_weights.begin(),consumer_weights.end()),ru(dy.begin(),dy.end());
    double *obs=memory.add(Z),*weights=memory.add(rw.size()),*upstream=memory.add(384),*gradient=memory.add(rw.size());
    ae_checked(cudaMemcpy(obs,rz.data(),Z*8,cudaMemcpyHostToDevice));ae_checked(cudaMemcpy(weights,rw.data(),rw.size()*8,cudaMemcpyHostToDevice));
    ae_checked(cudaMemcpy(upstream,ru.data(),384*8,cudaMemcpyHostToDevice));reference.run(obs,weights,upstream,gradient);
    std::vector<double> expected(384),expected_gradient(rw.size());
    ae_checked(cudaMemcpy(expected.data(),reference.nodes.back().y,384*8,cudaMemcpyDeviceToHost));
    ae_checked(cudaMemcpy(expected_gradient.data(),gradient,rw.size()*8,cudaMemcpyDeviceToHost));
    ae_compare(out,expected,"spatial consumer forward");
    ae_compare(std::vector<float>(grad.begin()+(frozen?0:P),grad.end()),expected_gradient,"spatial consumer all gradients");
    if(!frozen) {
        std::vector<float> actual(Z);std::vector<double> expected(Z);
        ae_checked(cudaMemcpy(actual.data(),a.cnn.grad_input.data,Z*4,cudaMemcpyDeviceToHost));
        ae_checked(cudaMemcpy(expected.data(),reference.nodes[0].g,Z*8,cudaMemcpyDeviceToHost));
        ae_compare(actual,expected,"spatial consumer latent derivatives");
    }
    double* loss;ae_checked(cudaMalloc(&loss,8));
    auto value=[&]() {Prec v=enc.forward(w,&a,x,stream);ae_native_probe<<<1,1,0,stream>>>(v.data,up.data,loss,384);
        double r;ae_checked(cudaMemcpyAsync(&r,loss,8,cudaMemcpyDeviceToHost,stream));ae_checked(cudaStreamSynchronize(stream));return r;};
    // One native float32 finite difference in each consumer tensor.
    int probes=0;long offset=frozen?0:P;
    for(int i=0;i<4;i++)for(Prec* t:{&w->cnn->net.weight[i],&w->cnn->net.bias[i]}) {
        long count=numel(t->shape),at=count/2;float center;
        ae_checked(cudaMemcpy(&center,t->data+at,4,cudaMemcpyDeviceToHost));float lo=center-.0002f,hi=center+.0002f;
        ae_checked(cudaMemcpy(t->data+at,&lo,4,cudaMemcpyHostToDevice));double a=value();
        ae_checked(cudaMemcpy(t->data+at,&hi,4,cudaMemcpyHostToDevice));double b=value();
        ae_checked(cudaMemcpy(t->data+at,&center,4,cudaMemcpyHostToDevice));
        ffc_require(fabs((b-a)/(hi-lo)-grad[offset+at])<3e-4+3e-2*fabs(grad[offset+at]),"consumer finite difference");
        probes++;offset+=count;
    }
    cudaGraph_t graph;cudaGraphExec_t exec;
    ae_checked(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));run();ae_checked(cudaStreamEndCapture(stream,&graph));
    ae_checked(cudaGraphInstantiate(&exec,graph,0));
    for(int i=0;i<3;i++) {
        if(i)ae_checked(cudaGraphLaunch(exec,stream));else run();ae_checked(cudaStreamSynchronize(stream));
        std::vector<float> yy(384);ae_checked(cudaMemcpy(yy.data(),y.data,384*4,cudaMemcpyDeviceToHost));
        ae_checked(cudaMemcpy(again.data(),g.mem,g.total_bytes,cudaMemcpyDeviceToHost));
        ffc_require(out==yy&&grad==again,"consumer eager/graph repeat");
    }
    off=0;for(int i=0;i<3;i++)for(Prec* t:{&w->prefix->net.weight[i],&w->prefix->net.bias[i]}) {
        ae_checked(cudaMemcpy(after.data()+off,t->data,numel(t->shape)*4,cudaMemcpyDeviceToHost));off+=numel(t->shape);
    }
    ffc_require(before==after,"consumer check mutated imported prefix");
    printf("{\"status\":\"passed\",\"frozen\":%s,\"finite_difference_probes\":%d,\"eager_graph_exact\":true,\"prefix_unchanged\":true}\n",frozen?"true":"false",probes);
    return 0;
}
