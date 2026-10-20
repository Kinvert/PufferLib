// Artifact validation runs at initialization only, before any imported mutation.
#include "../vendor/cJSON.c"
#include <stdint.h>

static void ffc_require(bool ok, const char* message) {
    if (!ok) { fprintf(stderr, "fullframe compression: %s\n", message); exit(1); }
}

static uint32_t ffc_rotate(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

static void ffc_sha256(const unsigned char* data, size_t bytes, char digest[65]) {
    static const uint32_t k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    size_t blocks = (bytes + 9 + 63) / 64;
    for (size_t block = 0; block < blocks; block++) {
        unsigned char buffer[64] = {};
        for (int i = 0; i < 64; i++) {
            size_t at = block * 64 + i;
            if (at < bytes) buffer[i] = data[at];
            else if (at == bytes) buffer[i] = 0x80;
            if (at >= blocks * 64 - 8) buffer[i] = (uint64_t(bytes) * 8) >> ((blocks * 64 - 1 - at) * 8);
        }
        uint32_t words[64];
        for (int i = 0; i < 16; i++) words[i] = uint32_t(buffer[4*i])<<24 | uint32_t(buffer[4*i+1])<<16
            | uint32_t(buffer[4*i+2])<<8 | buffer[4*i+3];
        for (int i = 16; i < 64; i++) {
            uint32_t a = words[i-15], b = words[i-2];
            words[i] = words[i-16] + (ffc_rotate(a,7)^ffc_rotate(a,18)^(a>>3)) + words[i-7]
                + (ffc_rotate(b,17)^ffc_rotate(b,19)^(b>>10));
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],z=h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t t = z + (ffc_rotate(e,6)^ffc_rotate(e,11)^ffc_rotate(e,25))
                + ((e&f)^(~e&g)) + k[i] + words[i];
            uint32_t u = (ffc_rotate(a,2)^ffc_rotate(a,13)^ffc_rotate(a,22)) + ((a&b)^(a&c)^(b&c));
            z=g; g=f; f=e; e=d+t; d=c; c=b; b=a; a=t+u;
        }
        uint32_t state[] = {a,b,c,d,e,f,g,z};
        for (int i=0;i<8;i++) h[i] += state[i];
    }
    for (int i=0;i<8;i++) snprintf(digest+8*i,9,"%08x",h[i]);
}

static char* ffc_read_file(const char* path, size_t* size) {
    FILE* file = fopen(path, "rb");
    ffc_require(file != NULL, "cannot open encoder artifact");
    ffc_require(!fseek(file,0,SEEK_END), "cannot seek encoder artifact");
    long length = ftell(file);
    ffc_require(length >= 0 && length <= 256*1024*1024, "invalid encoder artifact size");
    rewind(file);
    char* data = (char*)malloc(length + 1);
    ffc_require(data && fread(data,1,length,file) == (size_t)length && !fclose(file), "cannot read encoder artifact");
    data[length] = 0; *size = length; return data;
}

static bool ffc_string(cJSON* object, const char* key, const char* expected) {
    cJSON* value = cJSON_GetObjectItemCaseSensitive(object,key);
    return cJSON_IsString(value) && !strcmp(value->valuestring,expected);
}

static bool ffc_number(cJSON* object, const char* key, long expected) {
    cJSON* value = cJSON_GetObjectItemCaseSensitive(object,key);
    return cJSON_IsNumber(value) && value->valuedouble == expected;
}

static void ffc_import(FFCWeights* w, const char* directory, cudaStream_t stream) {
    char path[4096];
    ffc_require(snprintf(path,sizeof(path),"%s/encoder.json",directory) < (int)sizeof(path), "encoder path too long");
    size_t size; char* text = ffc_read_file(path,&size);
    cJSON* root = cJSON_ParseWithLength(text,size+1);
    ffc_require(root && ffc_string(root,"schema","fullframe-compression-encoder-v1")
        && ffc_string(root,"dtype","float32") && ffc_string(root,"layout","NHWC")
        && ffc_string(root,"range","0,1") && ffc_number(root,"model",w->model)
        && ffc_number(root,"height",w->height) && ffc_number(root,"width",w->width)
        && ffc_number(root,"channels",3) && ffc_string(root,"activation","relu-zero-derivative-0")
        && ffc_string(root,"area","box4-rowmajor-partial-edge")
        && ffc_string(root,"adaptive","floor-start-ceil-end-4x4")
        && ffc_string(root,"padding","same-trailing-extra-or-nature-valid"), "incompatible encoder manifest");
    cJSON* tensors = cJSON_GetObjectItemCaseSensitive(root,"tensors");
    cJSON* layers = cJSON_GetObjectItemCaseSensitive(root,"layers");
    ffc_require(cJSON_GetArraySize(tensors) == 8 && cJSON_GetArraySize(layers) == 4, "invalid encoder tensor/layer count");
    long count = 0;
    for (int i=0;i<4;i++) {
        NatureLayer d = w->net.layer[i];
        int values[] = {d.ih,d.iw,d.ci,d.co,d.k,d.stride,d.oh,d.ow,w->ph[i],w->pw[i]};
        cJSON* layer = cJSON_GetArrayItem(layers,i);
        ffc_require(cJSON_GetArraySize(layer) == 10, "invalid layer descriptor");
        for (int j=0;j<10;j++) {
            cJSON* item = cJSON_GetArrayItem(layer,j);
            ffc_require(cJSON_IsNumber(item) && item->valuedouble == values[j], "encoder graph mismatch");
        }
        for (int bias=0;bias<2;bias++) {
            cJSON* t = cJSON_GetArrayItem(tensors,2*i+bias);
            char name[32]; snprintf(name,sizeof(name),"%s_%d",bias?"bias":"weight",i);
            long elements = bias ? d.co : (long)d.co*d.k*d.k*d.ci;
            cJSON* shape = cJSON_GetObjectItemCaseSensitive(t,"shape");
            ffc_require(ffc_string(t,"name",name) && ffc_number(t,"offset",count)
                && ffc_number(t,"elements",elements) && cJSON_GetArraySize(shape) == (bias?1:2)
                && cJSON_IsNumber(cJSON_GetArrayItem(shape,0)) && cJSON_GetArrayItem(shape,0)->valuedouble == d.co,
                "encoder tensor layout mismatch");
            if (!bias) ffc_require(cJSON_IsNumber(cJSON_GetArrayItem(shape,1))
                && cJSON_GetArrayItem(shape,1)->valuedouble == d.k*d.k*d.ci, "encoder weight shape mismatch");
            count += elements;
        }
    }
    ffc_require(ffc_number(root,"elements",count), "encoder element count mismatch");
    ffc_require(snprintf(path,sizeof(path),"%s/encoder.bin",directory) < (int)sizeof(path), "encoder path too long");
    char* data = ffc_read_file(path,&size);
    ffc_require(size == count*sizeof(float), "encoder byte count mismatch");
    char digest[65]; ffc_sha256((unsigned char*)data,size,digest);
    ffc_require(ffc_string(root,"sha256",digest), "encoder hash mismatch");
    for (long i=0;i<count;i++) ffc_require(isfinite(((float*)data)[i]), "nonfinite encoder weight");
    // All validation completed before the first device write.
    long offset=0;
    for (int i=0;i<4;i++) for (Prec* tensor : {&w->net.weight[i], &w->net.bias[i]}) {
        long n = numel(tensor->shape);
        ffc_require(cudaMemcpyAsync(tensor->data,data+offset*4,n*4,cudaMemcpyHostToDevice,stream) == cudaSuccess,
            "encoder upload failed"); offset += n;
    }
    ffc_require(cudaStreamSynchronize(stream) == cudaSuccess,"encoder upload synchronization failed");
    free(data); free(text); cJSON_Delete(root);
}
