// Test-only independent CUDA arithmetic. No production encoder/pool/GEMM calls.
#include "native_encoder_reference.cu"

__global__ static void ffc_ref_mean(const double* x, double* y, int B,
        int H, int W, int C, bool area) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int OH=area?(H+3)/4:4, OW=area?(W+3)/4:4;
    if (i>=B*OH*OW*C) return;
    int c=i%C, ox=i/C%OW, oy=i/(C*OW)%OH, b=i/(C*OW*OH);
    int y0=area?oy*4:oy*H/OH, x0=area?ox*4:ox*W/OW;
    int y1=area?min(y0+4,H):((oy+1)*H+OH-1)/OH;
    int x1=area?min(x0+4,W):((ox+1)*W+OW-1)/OW;
    double sum=0;
    for(int yy=y0;yy<y1;yy++) for(int xx=x0;xx<x1;xx++) sum+=x[((b*H+yy)*W+xx)*C+c];
    y[i]=sum/((y1-y0)*(x1-x0));
}

__global__ static void ffc_ref_mean_back(const double* dy, double* dx, int B,
        int H, int W, int C, bool area) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=B*H*W*C) return;
    int c=i%C,x=i/C%W,y=i/(C*W)%H,b=i/(C*W*H);
    int OH=area?(H+3)/4:4,OW=area?(W+3)/4:4;
    if(area) {
        int count=min(4,H-y/4*4)*min(4,W-x/4*4);
        dx[i]=dy[((b*OH+y/4)*OW+x/4)*C+c]/count;
        return;
    }
    double sum=0;
    for(int oy=0;oy<OH;oy++) {
        int y0=area?oy*4:oy*H/OH, y1=area?min(y0+4,H):((oy+1)*H+OH-1)/OH;
        if(y<y0||y>=y1) continue;
        for(int ox=0;ox<OW;ox++) {
            int x0=area?ox*4:ox*W/OW, x1=area?min(x0+4,W):((ox+1)*W+OW-1)/OW;
            if(x>=x0&&x<x1) sum+=dy[((b*OH+oy)*OW+ox)*C+c]/((y1-y0)*(x1-x0));
        }
    }
    dx[i]=sum;
}

__global__ static void ffc_ref_relu_back(const double* dy,const double* y,
        const float* actual,double* delta,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) delta[i]=(actual?actual[i]>0:y[i]>0)?dy[i]:0;
}

struct FFCRefNode {
    int op,h,w,c,conv;
    Layer layer;
    double *y,*g,*delta;
};

struct FFCReference {
    Storage storage;
    std::vector<FFCRefNode> nodes;
    int batch,parameters=0;

    FFCReference(int model,int H,int W,int B,int spatial_stride=0,int spatial_channels=16,int input_channels=3,bool consumer=false):batch(B) {
        nodes.push_back({0,H,W,input_channels,-1});
        if(model) mean(true);
        if(model==2) {
            conv(32,8,4,false,0);conv(64,4,2,false,1);conv(64,3,1,false,2);
        } else {
            conv(8,consumer?3:model?3:4,consumer?1:model?1:4,true,0);
            conv(16,3,2,true,1);conv(spatial_stride?spatial_channels:16,spatial_stride?4:3,spatial_stride?spatial_stride/8:2,true,2);
            if(!spatial_stride)mean(false);
        }
        if(!spatial_stride)conv(128,1,1,false,3,true);
        for(auto& n:nodes) {
            int size=B*n.h*n.w*n.c;
            n.y=storage.add(size);n.g=storage.add(size);n.delta=storage.add(size);
        }
    }
    void mean(bool area) {
        auto p=nodes.back();
        nodes.push_back({area?1:2,area?(p.h+3)/4:4,area?(p.w+3)/4:4,p.c,-1});
    }
    void conv(int co,int k,int s,bool same,int index,bool flat=false) {
        auto p=nodes.back();
        int H=flat?1:p.h,W=flat?1:p.w,C=flat?p.h*p.w*p.c:p.c;
        int OH=same?(H+s-1)/s:(H-k)/s+1,OW=same?(W+s-1)/s:(W-k)/s+1;
        int ph=same?max(0,(OH-1)*s+k-H)/2:0,pw=same?max(0,(OW-1)*s+k-W)/2:0;
        Layer l={H,W,C,OH,OW,co,k,s,ph,pw,parameters};
        parameters+=co*(C*k*k+1);
        nodes.push_back({3,OH,OW,co,index,l});
    }
    void run(const double* input,const double* weights,const double* up,double* gradient,
            const float* const* branches=nullptr) {
        checked(cudaMemcpy(nodes[0].y,input,batch*nodes[0].h*nodes[0].w*nodes[0].c*sizeof(double),cudaMemcpyDeviceToDevice));
        for(int i=1;i<(int)nodes.size();i++) {
            auto& n=nodes[i];auto& p=nodes[i-1];int size=batch*n.h*n.w*n.c;
            if(n.op==3) forward<<<(size+127)/128,128>>>(p.y,weights,n.y,n.layer,batch);
            else ffc_ref_mean<<<(size+127)/128,128>>>(p.y,n.y,batch,p.h,p.w,p.c,n.op==1);
        }
        auto last=nodes.back();
        checked(cudaMemcpy(nodes.back().g,up,batch*last.h*last.w*last.c*sizeof(double),cudaMemcpyDeviceToDevice));
        for(int i=nodes.size()-1;i>0;i--) {
            auto& n=nodes[i];auto& p=nodes[i-1];int size=batch*n.h*n.w*n.c;
            if(n.op==3) {
                ffc_ref_relu_back<<<(size+127)/128,128>>>(n.g,n.y,branches?branches[n.conv]:nullptr,n.delta,size);
                int count=n.layer.co*n.layer.k*n.layer.k*n.layer.ci;
                weight_backward<<<(count+127)/128,128>>>(p.y,n.delta,gradient,n.layer,batch);
                bias_backward<<<(n.c+127)/128,128>>>(n.delta,gradient,n.layer,batch);
                int dx=batch*p.h*p.w*p.c;
                input_backward<<<(dx+127)/128,128>>>(weights,n.delta,p.g,n.layer,batch);
            } else {
                int dx=batch*p.h*p.w*p.c;
                ffc_ref_mean_back<<<(dx+127)/128,128>>>(n.g,p.g,batch,p.h,p.w,p.c,n.op==1);
            }
        }
        checked(cudaDeviceSynchronize());
    }
};
