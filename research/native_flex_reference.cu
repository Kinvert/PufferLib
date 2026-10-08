// Independent CUDA graph reference; includes only our direct float64 oracle.
// No production convolution, GEMM, im2col or pooling implementation is included.
#include "native_encoder_reference.cu"
#include <algorithm>
#include <cmath>

struct Op { Layer l; int type; bool skip; };

static int graph(int model, int hidden, const int* settings, std::vector<Op>& ops) {
    if (model == 1) {
        std::vector<Layer> layers;
        int n = topology(1, hidden, layers);
        for (Layer l : layers) ops.push_back({l, 0, false});
        return n;
    }
    if (model != 0 || hidden != 128 || !settings) return -1;
    int depth = settings[0], projection = settings[1], gap = settings[2];
    if (depth < 1 || depth > 3 || projection < 16 || projection > 128
            || (projection & (projection - 1)) || gap < 0 || gap > 1) return -1;
    int h = 36, w = 44, ci = 1, offset = 0;
    auto conv = [&](int co, int k, int s, bool skip) {
        int oh = (h + s - 1) / s, ow = (w + s - 1) / s;
        int ph = std::max((oh - 1) * s + k - h, 0) / 2;
        int pw = std::max((ow - 1) * s + k - w, 0) / 2;
        ops.push_back({{h, w, ci, oh, ow, co, k, s, ph, pw, offset}, 0, skip});
        offset += co * (k * k * ci + 1); h = oh; w = ow; ci = co;
    };
    for (int j = 0; j < depth; j++) {
        const int* p = settings + 3 + 5 * j;
        if ((p[0] != 8 && p[0] != 16 && p[0] != 32) || p[1] < 1
                || p[1] > (j == 0 ? 8 : 5) || p[2] < (j == 0 ? 4 : 1)
                || p[2] > (j == 0 ? 8 : 4) || (p[2] & (p[2]-1))
                || p[3] < 0 || p[3] > 2 || p[4] < 0 || p[4] > 1) return -1;
        conv(p[0], p[1], p[2], false);
        if (p[4]) conv(ci, 3, 1, true);
        if (p[3]) {
            int oh = (h + 1)/2, ow = (w + 1)/2;
            ops.push_back({{h,w,ci,oh,ow,ci,3,2,h%2,w%2,offset},p[3],false});
            h = oh; w = ow;
        }
    }
    if (gap) { ops.push_back({{h,w,ci,1,1,ci,1,1,0,0,offset},3,false}); h=w=1; }
    ci *= h*w; h=w=1; conv(projection,1,1,false);
    if (projection != hidden) conv(hidden,1,1,false);
    return offset;
}

__global__ static void graph_forward(const double* x, const double* weights,
                                    double* y, Op op, int batch) {
    Layer l = op.l;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch*l.oh*l.ow*l.co) return;
    int c=i%l.co, ox=(i/l.co)%l.ow, oy=(i/(l.co*l.ow))%l.oh;
    int b=i/(l.co*l.ow*l.oh);
    double sum=0;
    if (!op.type) {
        int cols=l.k*l.k*l.ci;
        for(int ky=0;ky<l.k;ky++) for(int kx=0;kx<l.k;kx++) {
            int y0=oy*l.stride+ky-l.ph, x0=ox*l.stride+kx-l.pw;
            if(y0<0||y0>=l.h||x0<0||x0>=l.w) continue;
            for(int ic=0;ic<l.ci;ic++) sum += x[((b*l.h+y0)*l.w+x0)*l.ci+ic]
                * weights[l.offset+c*cols+(ky*l.k+kx)*l.ci+ic];
        }
        sum += weights[l.offset+l.co*cols+c];
        if(op.skip) sum += x[i];
        y[i]=sum>0?sum:0;
        return;
    }
    int top=op.type==3?0:oy*2-l.ph, left=op.type==3?0:ox*2-l.pw;
    int bottom=op.type==3?l.h:min(top+3,l.h), right=op.type==3?l.w:min(left+3,l.w);
    int count=0;
    if(op.type==1) sum=-1e30;
    for(int iy=max(top,0);iy<bottom;iy++) for(int ix=max(left,0);ix<right;ix++) {
        double v=x[((b*l.h+iy)*l.w+ix)*l.ci+c];
        if(op.type==1) { if(v>sum) sum=v; } else sum+=v;
        count++;
    }
    y[i]=op.type==1?sum:sum/count;
}

__global__ static void graph_pool_backward(const double* x,const double* g,double* dx,Op op,int batch) {
    Layer l=op.l;
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=batch*l.h*l.w*l.ci) return;
    int c=i%l.ci, ix=(i/l.ci)%l.w, iy=(i/(l.ci*l.w))%l.h, b=i/(l.ci*l.w*l.h);
    double sum=0;
    for(int oy=0;oy<l.oh;oy++) for(int ox=0;ox<l.ow;ox++) {
        int top=op.type==3?0:oy*2-l.ph, left=op.type==3?0:ox*2-l.pw;
        int bottom=op.type==3?l.h:min(top+3,l.h), right=op.type==3?l.w:min(left+3,l.w);
        top=max(top,0); left=max(left,0);
        if(iy<top||iy>=bottom||ix<left||ix>=right) continue;
        double delta=g[((b*l.oh+oy)*l.ow+ox)*l.co+c];
        if(op.type!=1) { sum+=delta/((bottom-top)*(right-left)); continue; }
        double best=-1e30; int winner=-1;
        for(int y0=top;y0<bottom;y0++) for(int x0=left;x0<right;x0++) {
            int at=((b*l.h+y0)*l.w+x0)*l.ci+c;
            if(x[at]>best) { best=x[at]; winner=at; }
        }
        if(winner==i) sum+=delta;
    }
    dx[i]=sum;
}

__global__ static void graph_skip(double* dx,const double* delta,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) dx[i]+=delta[i];
}

extern "C" int flexref_layout(int model,int hidden,const int* settings,int* rows) {
    std::vector<Op> ops;
    int params=graph(model,hidden,settings,ops);
    if(params<0) return params;
    if(rows) {
        rows[0]=ops.size(); int i=1;
        for(Op op:ops) for(int v:{op.l.h,op.l.w,op.l.ci,op.l.oh,op.l.ow,op.l.co,
                op.l.k,op.l.stride,op.l.ph,op.l.pw,op.l.offset,op.type,int(op.skip)}) rows[i++]=v;
    }
    return params; // Scalar metadata only, no device query/allocation.
}

extern "C" int flexref_run(int model,int hidden,const int* settings,int batch,
        const double* input,const double* weights,const double* upstream,
        const double* native_trace,double* trace,double* gradient,double* loss) {
    if(batch<1||batch>2048||!input||!weights||!upstream) return -1;
    std::vector<Op> ops;
    int params=graph(model,hidden,settings,ops);
    if(params<0) return -1;
    try {
        Storage storage;
        double* x=storage.add(size_t(batch)*1584), *w=storage.add(params), *g=storage.add(batch*hidden);
        double* dw=storage.add(params);
        checked(cudaMemcpy(x,input,size_t(batch)*1584*sizeof(double),cudaMemcpyHostToDevice));
        checked(cudaMemcpy(w,weights,params*sizeof(double),cudaMemcpyHostToDevice));
        checked(cudaMemcpy(g,upstream,batch*hidden*sizeof(double),cudaMemcpyHostToDevice));
        std::vector<double*> xs={x}, ys;
        int offset=0;
        for(Op op:ops) {
            int n=batch*op.l.oh*op.l.ow*op.l.co;
            double* y=storage.add(n);
            graph_forward<<<(n+127)/128,128>>>(xs.back(),w,y,op,batch);
            checked(cudaGetLastError()); ys.push_back(y);
            if(trace) checked(cudaMemcpy(trace+offset,y,n*sizeof(double),cudaMemcpyDeviceToHost));
            if(native_trace) {
                double* actual=storage.add(n);
                checked(cudaMemcpy(actual,native_trace+offset,n*sizeof(double),cudaMemcpyHostToDevice));
                xs.push_back(actual);
            } else xs.push_back(y);
            offset+=n;
        }
        if(loss) {
            double* device_loss=storage.add(1);
            loss_sum<<<1,1>>>(ys.back(),g,device_loss,batch*hidden);
            checked(cudaMemcpy(loss,device_loss,sizeof(double),cudaMemcpyDeviceToHost));
        }
        if(gradient) for(int j=int(ops.size())-1;j>=0;j--) {
            Op op=ops[j]; Layer l=op.l;
            int n=batch*l.oh*l.ow*l.co, ni=batch*l.h*l.w*l.ci;
            double* dx=storage.add(ni);
            if(op.type) graph_pool_backward<<<(ni+127)/128,128>>>(xs[j],g,dx,op,batch);
            else {
                double* delta=storage.add(n);
                // Native float32 branches are part of this backward contract.
                // No native gradient, GEMM result or pooling winner is reused.
                relu_backward<<<(n+127)/128,128>>>(xs[j+1],g,delta,n);
                int nw=l.co*l.k*l.k*l.ci;
                weight_backward<<<(nw+127)/128,128>>>(xs[j],delta,dw,l,batch);
                bias_backward<<<(l.co+127)/128,128>>>(delta,dw,l,batch);
                input_backward<<<(ni+127)/128,128>>>(w,delta,dx,l,batch);
                if(op.skip) graph_skip<<<(ni+127)/128,128>>>(dx,delta,ni);
            }
            checked(cudaGetLastError()); g=dx;
        }
        checked(cudaDeviceSynchronize());
        if(gradient) checked(cudaMemcpy(gradient,dw,params*sizeof(double),cudaMemcpyDeviceToHost));
    } catch(int error) { return error; }
    return 0;
}
