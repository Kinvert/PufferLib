// Isolated float64 CUDA oracle. No production encoder/GEMM/im2col includes.
// NHWC convolutions and their derivatives use direct, ordered scalar sums.
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

struct Layer {
    int h, w, ci, oh, ow, co, k, stride, ph, pw, offset;
};

static void checked(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA reference: %s\n", cudaGetErrorString(status));
        throw int(status);
    }
}

struct Storage {
    std::vector<double*> buffers;
    Storage() = default;
    Storage(const Storage&) = delete;
    ~Storage() { for (double* buffer : buffers) cudaFree(buffer); }
    double* add(size_t n) {
        double* buffer = nullptr;
        checked(cudaMalloc(&buffer, n * sizeof(double)));
        buffers.push_back(buffer);
        return buffer;
    }
};

static int topology(int model, int hidden, std::vector<Layer>& layers) {
    if (hidden != 64 && hidden != 128 && hidden != 256) return -1;
    int h = 36, w = 44, ci = 1, offset = 0;
    auto conv = [&](int co, int k, int stride, bool same) {
        int oh = same ? (h + stride - 1) / stride : (h - k) / stride + 1;
        int ow = same ? (w + stride - 1) / stride : (w - k) / stride + 1;
        int ph = same ? ((oh - 1) * stride + k - h) / 2 : 0;
        int pw = same ? ((ow - 1) * stride + k - w) / 2 : 0;
        layers.push_back({h, w, ci, oh, ow, co, k, stride, ph, pw, offset});
        offset += co * (k * k * ci + 1);
        h = oh; w = ow; ci = co;
    };
    auto linear = [&](int co) { ci *= h * w; h = w = 1; conv(co, 1, 1, false); };
    if (model == 0) {
        conv(16, 7, 4, true); linear(64);
        if (hidden != 64) linear(hidden);
    } else if (model == 1) {
        conv(32, 8, 4, false); conv(64, 4, 2, false); conv(64, 3, 1, false);
        linear(hidden);
    } else return -1;
    return offset;
}

__global__ static void forward(const double* x, const double* weights,
                               double* y, Layer l, int batch) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int n = batch * l.oh * l.ow * l.co;
    if (index >= n) return;
    int oc = index % l.co, ox = (index / l.co) % l.ow;
    int oy = (index / (l.co * l.ow)) % l.oh;
    int b = index / (l.co * l.ow * l.oh), cols = l.k * l.k * l.ci;
    double sum = 0;
    for (int ky = 0; ky < l.k; ky++) for (int kx = 0; kx < l.k; kx++) {
        int iy = oy * l.stride + ky - l.ph, ix = ox * l.stride + kx - l.pw;
        if (iy < 0 || iy >= l.h || ix < 0 || ix >= l.w) continue;
        for (int ic = 0; ic < l.ci; ic++) {
            int xi = ((b * l.h + iy) * l.w + ix) * l.ci + ic;
            int wi = l.offset + oc * cols + (ky * l.k + kx) * l.ci + ic;
            sum += x[xi] * weights[wi];
        }
    }
    sum += weights[l.offset + l.co * cols + oc];
    y[index] = sum > 0 ? sum : 0;
}

__global__ static void relu_backward(const double* y, const double* upstream,
                                    double* delta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) delta[i] = y[i] > 0 ? upstream[i] : 0;
}

__global__ static void weight_backward(const double* x, const double* delta,
                                       double* gradient, Layer l, int batch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int cols = l.k * l.k * l.ci;
    if (i >= l.co * cols) return;
    int oc = i / cols, ic = i % l.ci;
    int kx = (i / l.ci) % l.k, ky = (i / (l.ci * l.k)) % l.k;
    double sum = 0;
    for (int b = 0; b < batch; b++) for (int oy = 0; oy < l.oh; oy++) for (int ox = 0; ox < l.ow; ox++) {
        int iy = oy * l.stride + ky - l.ph, ix = ox * l.stride + kx - l.pw;
        if (iy < 0 || iy >= l.h || ix < 0 || ix >= l.w) continue;
        sum += x[((b * l.h + iy) * l.w + ix) * l.ci + ic] * delta[((b * l.oh + oy) * l.ow + ox) * l.co + oc];
    }
    gradient[l.offset + i] = sum;
}

__global__ static void bias_backward(const double* delta, double* gradient, Layer l, int batch) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= l.co) return;
    double sum = 0;
    for (int row = 0; row < batch * l.oh * l.ow; row++) sum += delta[row * l.co + oc];
    gradient[l.offset + l.co * l.k * l.k * l.ci + oc] = sum;
}

__global__ static void input_backward(const double* weights, const double* delta,
                                      double* dx, Layer l, int batch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch * l.h * l.w * l.ci) return;
    int ic = i % l.ci, ix = (i / l.ci) % l.w;
    int iy = (i / (l.ci * l.w)) % l.h, b = i / (l.ci * l.w * l.h);
    double sum = 0;
    for (int ky = 0; ky < l.k; ky++) for (int kx = 0; kx < l.k; kx++) {
        int py = iy + l.ph - ky, px = ix + l.pw - kx;
        if (py < 0 || px < 0 || py % l.stride || px % l.stride) continue;
        int oy = py / l.stride, ox = px / l.stride;
        if (oy >= l.oh || ox >= l.ow) continue;
        for (int oc = 0; oc < l.co; oc++)
            sum += weights[l.offset + ((oc * l.k + ky) * l.k + kx) * l.ci + ic] * delta[((b * l.oh + oy) * l.ow + ox) * l.co + oc];
    }
    dx[i] = sum;
}

__global__ static void loss_sum(const double* y, const double* upstream, double* loss, int n) {
    if (blockIdx.x || threadIdx.x) return;
    double sum = 0;
    for (int i = 0; i < n; i++) sum += y[i] * upstream[i];
    *loss = sum;
}

static void execute(const std::vector<Layer>& layers, int batch, int params,
                    const double* input, const double* weights, const double* upstream,
                    double* output, double* gradient, double* loss, double* input_gradient = nullptr) {
    Storage storage;
    size_t ni = size_t(batch) * layers.front().h * layers.front().w * layers.front().ci;
    int no = batch * layers.back().oh * layers.back().ow * layers.back().co;
    double* x = storage.add(ni); double* w = storage.add(params);
    double* g = storage.add(no); double* dw = storage.add(params);
    double* device_loss = storage.add(1);
    checked(cudaMemcpy(x, input, ni * sizeof(double), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(w, weights, params * sizeof(double), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(g, upstream, no * sizeof(double), cudaMemcpyHostToDevice));
    std::vector<double*> activations = {x};
    for (const Layer& layer : layers) {
        int n = batch * layer.oh * layer.ow * layer.co;
        double* y = storage.add(n);
        forward<<<(n + 127) / 128, 128>>>(activations.back(), w, y, layer, batch);
        checked(cudaGetLastError()); activations.push_back(y);
    }
    if (loss) {
        loss_sum<<<1, 1>>>(activations.back(), g, device_loss, no);
        checked(cudaGetLastError());
    }
    if (gradient || input_gradient) for (int j = int(layers.size()) - 1; j >= 0; j--) {
        const Layer& layer = layers[j];
        int n = batch * layer.oh * layer.ow * layer.co;
        int nx = batch * layer.h * layer.w * layer.ci;
        int nw = layer.co * layer.k * layer.k * layer.ci;
        double* delta = storage.add(n); double* dx = storage.add(nx);
        relu_backward<<<(n + 127) / 128, 128>>>(activations[j+1], g, delta, n);
        weight_backward<<<(nw + 127) / 128, 128>>>(activations[j], delta, dw, layer, batch);
        bias_backward<<<(layer.co + 127) / 128, 128>>>(delta, dw, layer, batch);
        input_backward<<<(nx + 127) / 128, 128>>>(w, delta, dx, layer, batch);
        checked(cudaGetLastError()); g = dx;
    }
    checked(cudaDeviceSynchronize());
    if (output) checked(cudaMemcpy(output, activations.back(), no * sizeof(double), cudaMemcpyDeviceToHost));
    if (gradient) checked(cudaMemcpy(gradient, dw, params * sizeof(double), cudaMemcpyDeviceToHost));
    if (input_gradient) checked(cudaMemcpy(input_gradient, g, ni * sizeof(double), cudaMemcpyDeviceToHost));
    if (loss) checked(cudaMemcpy(loss, device_loss, sizeof(double), cudaMemcpyDeviceToHost));
}

extern "C" int cnnref_parameters(int model, int hidden) {
    std::vector<Layer> layers;
    return topology(model, hidden, layers); // Host scalar metadata only.
}

extern "C" int cnnref_run(int model, int batch, int hidden,
    const double* input, const double* weights, const double* upstream,
    double* output, double* gradient, double* loss) {
    if (batch < 1 || batch > 2048 || !input || !weights || !upstream) return -1;
    std::vector<Layer> layers;
    int params = topology(model, hidden, layers);
    if (params < 0) return -1;
    try { execute(layers, batch, params, input, weights, upstream, output, gradient, loss); }
    catch (int error) { return error; }
    return 0;
}

extern "C" int cnnref_selftest() {
    // Exact dyadics, including zero-preactivation ReLU derivative and dX.
    const double input[] = {2, 3}, weights[] = {1, 2, -1, 1, 1, -1}, upstream[] = {4, 5};
    const double expected_y[] = {9, 0}, expected_dw[] = {8, 12, 0, 0, 4, 0}, expected_dx[] = {4, 8};
    const std::vector<Layer> layers = {{1, 1, 2, 1, 1, 2, 1, 1, 0, 0, 0}};
    double y[2], dw[6], dx[2], loss;
    try { execute(layers, 1, 6, input, weights, upstream, y, dw, &loss, dx); }
    catch (int error) { return error; }
    for (int i = 0; i < 2; i++) if (y[i] != expected_y[i] || dx[i] != expected_dx[i]) return -2;
    for (int i = 0; i < 6; i++) if (dw[i] != expected_dw[i]) return -3;
    return loss == 36 ? 0 : -4;
}
