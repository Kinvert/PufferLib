// Separately normalize object classes/instances and the remaining background.
// Factors are computed from visible labels; pixels and labels stay separate.
__global__ static void forage_weight_pixels(const unsigned char* labels,
        const unsigned char* masks, const float* factors, float* weights, int B) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int pixels=OBS_HEIGHT*OBS_WIDTH;
    if(i>=B*pixels)return;
    int b=i/pixels,x=i%OBS_WIDTH,y=i/OBS_WIDTH%OBS_HEIGHT;
    int cell=y/FORAGE_CELL*FORAGE_WIDTH+x/FORAGE_CELL;
    int kind=labels[b*FORAGE_AREA+cell];
    int at=(y%FORAGE_CELL)*FORAGE_CELL+x%FORAGE_CELL;
    // Masks are packed for this batch, one full atlas per sample.
    int fg=masks[(b*FORAGE_TILE_TYPES+kind)*FORAGE_CELL*FORAGE_CELL+at];
    weights[i]=factors[b*(FORAGE_TILE_TYPES+1)+(fg?kind:FORAGE_TILE_TYPES)];
}

static std::vector<unsigned char> forage_mask_atlas(int skin) {
    std::vector<unsigned char> masks(FORAGE_TILE_TYPES*FORAGE_CELL*FORAGE_CELL);
    for(int kind=0;kind<FORAGE_TILE_TYPES;kind++)
        for(int p=0;p<FORAGE_CELL*FORAGE_CELL;p++)
            masks[kind*FORAGE_CELL*FORAGE_CELL+p]=forage_foreground(skin,kind,p);
    return masks;
}

static void forage_weights(AEStream* s,int B,float object_weight,float background_weight) {
    ffc_require(s->H==OBS_HEIGHT && s->W==OBS_WIDTH,"Forage annotation/image shape");
    const int atlas=FORAGE_TILE_TYPES*FORAGE_CELL*FORAGE_CELL;
    if(!s->labels_device) {
        ae_checked(cudaMalloc(&s->labels_device,B*FORAGE_AREA));
        ae_checked(cudaMalloc(&s->masks_device,B*atlas));
        ae_checked(cudaMalloc(&s->factors_device,B*(FORAGE_TILE_TYPES+1)*4));
        ae_checked(cudaMalloc(&s->pixel_weights,(long)B*s->H*s->W*4));
    }
    std::vector<unsigned char> masks(B*atlas);
    std::vector<float> factors(B*(FORAGE_TILE_TYPES+1));
    for(int b=0;b<B;b++) {
        auto mask=forage_mask_atlas(s->skins[b]);
        memcpy(masks.data()+b*atlas,mask.data(),atlas);
        int instances[FORAGE_TILE_TYPES]={},area[FORAGE_TILE_TYPES]={};
        for(int kind=1;kind<FORAGE_TILE_TYPES;kind++)
            for(int p=0;p<FORAGE_CELL*FORAGE_CELL;p++)area[kind]+=mask[kind*FORAGE_CELL*FORAGE_CELL+p];
        for(int cell=0;cell<FORAGE_AREA;cell++)instances[s->labels_host[b*FORAGE_AREA+cell]]++;
        int classes=0,background=s->H*s->W;
        for(int kind=1;kind<FORAGE_TILE_TYPES;kind++) {
            if(instances[kind])classes++;
            background-=instances[kind]*area[kind];
        }
        ffc_require(classes>0 && background>0,"visible object/background counts");
        for(int kind=1;kind<FORAGE_TILE_TYPES;kind++)if(instances[kind])
            factors[b*(FORAGE_TILE_TYPES+1)+kind]=(float)(s->H*s->W)*object_weight/(classes*instances[kind]*area[kind]);
        factors[b*(FORAGE_TILE_TYPES+1)+FORAGE_TILE_TYPES]=(float)(s->H*s->W)*background_weight/background;
    }
    ae_checked(cudaMemcpy(s->labels_device,s->labels_host.data(),B*FORAGE_AREA,cudaMemcpyHostToDevice));
    ae_checked(cudaMemcpy(s->masks_device,masks.data(),B*atlas,cudaMemcpyHostToDevice));
    ae_checked(cudaMemcpy(s->factors_device,factors.data(),factors.size()*4,cudaMemcpyHostToDevice));
    forage_weight_pixels<<<grid_size((long)B*s->H*s->W),BLOCK_SIZE>>>(
        s->labels_device,s->masks_device,s->factors_device,s->pixel_weights,B);
    ae_checked(cudaDeviceSynchronize());
}
