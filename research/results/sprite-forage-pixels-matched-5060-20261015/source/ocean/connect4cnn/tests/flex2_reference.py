"""Independent float64 oracle for comparisons with the native CUDA encoder.

Test-only: no trainer, production kernel imports, or native layer descriptors.
Geometry and parameter order are derived from the fixture INI. Backpropagation
scatters gradients through image windows, unlike the CUDA gather kernels.
Only invoke numerical routines as part of the GPU verification suite.
"""
import configparser
import math

import numpy as np


def plan(path, hidden):
    ini = configparser.ConfigParser()
    ini.read(path)
    p = ini["policy"]
    height, width, channels = 36, 44, 1
    ops, metadata = [], []
    offset = 0

    def conv(co, kernel, stride, dilation, activation, residual=False, dense=False):
        nonlocal height, width, channels, offset
        shape = (height, width, channels)
        if dense:
            height, width, channels = 1, 1, math.prod(shape)
        oh, ow = math.ceil(height/stride), math.ceil(width/stride)
        op = dict(kind="conv", shape=shape, co=co, kernel=kernel, stride=stride,
                  dilation=dilation, activation=activation, residual=residual,
                  dense=dense, output=(oh, ow, co), offset=offset)
        sizes = [co*kernel*kernel*channels, co]
        if activation >= 3:
            sizes.append(12)
        for kind, size in enumerate(sizes):
            metadata.append((offset, size, kind, activation))
            offset += size
        op["sizes"] = sizes
        ops.append(op)
        height, width, channels = oh, ow, co

    for stage in range(1, p.getint("cnn_depth", 2)+1):
        def option(key, default):
            return p.getint(f"cnn_{key}_{stage}", default)
        co = option("channels", 8)
        dilation, activation = option("dilation", 1), option("activation", 0)
        conv(co, option("kernel", 7 if stage == 1 else 3),
             option("stride", 4 if stage == 1 else 1), dilation, activation)
        for _ in range(option("residual", 0)):
            conv(co, 3, 1, dilation, activation, residual=True)
        pool = option("pool", 0)
        if pool:
            output = (math.ceil(height/2), math.ceil(width/2), channels)
            ops.append(dict(kind="max" if pool == 1 else "average",
                            shape=(height, width, channels), output=output))
            height, width, channels = output
    readout = p.getint("cnn_readout", 0)
    if readout:
        grid = {1: 1, 2: 2, 3: 4}[readout]
        ops.append(dict(kind="adaptive", shape=(height, width, channels),
                        output=(grid, grid, channels)))
        height = width = grid
    projection = p.getint("cnn_projection", 32)
    activation = p.getint("cnn_projection_activation", 0)
    conv(projection, 1, 1, 1, activation, dense=True)
    if projection != hidden:
        conv(hidden, 1, 1, 1, activation, dense=True)
    return ops, np.asarray(metadata, dtype=np.int64), offset


def activation(x, kind, coefficients):
    """Value, input derivative and per-element coefficient derivatives.

    Explicit subgradients: ReLU'(0)=0, PReLU'(0)=1, sign(Q=0)=0.
    Polynomial evaluation uses NumPy's Horner method, not CUDA's power sums.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(coefficients, dtype=np.float64)
    dp = np.zeros((*x.shape, 12), dtype=np.float64)
    if kind == 0:
        return np.maximum(x, 0), (x > 0).astype(float), dp
    if kind == 1:
        sigmoid = np.exp(-np.logaddexp(0, -x))
        return x*sigmoid, sigmoid*(1+x*(1-sigmoid)), dp
    if kind == 2:
        cdf = np.fromiter((.5*math.erfc(-v/math.sqrt(2)) for v in x.flat),
                          dtype=np.float64, count=x.size).reshape(x.shape)
        return x*cdf, cdf+x*np.exp(-x*x/2)/math.sqrt(2*math.pi), dp
    if kind == 3:
        derivative = np.where(x >= 0, 1, p[0])
        dp[..., 0] = np.minimum(x, 0)
        return x*derivative, derivative, dp
    if kind != 4:
        raise ValueError(f"Unknown activation {kind}")
    poly = np.polynomial.polynomial
    numerator = poly.polyval(x, p[:6])
    denominator_poly = np.r_[0., p[6:10]]
    q = poly.polyval(x, denominator_poly)
    denominator = 1+np.abs(q)
    derivative = (poly.polyval(x, poly.polyder(p[:6]))/denominator
                  -numerator*np.sign(q)*poly.polyval(x, poly.polyder(denominator_poly))/denominator**2)
    for j in range(6):
        dp[..., j] = x**j/denominator
    for j in range(1, 5):
        dp[..., 5+j] = -numerator*np.sign(q)*x**j/denominator**2
    return numerator/denominator, derivative, dp


def pooling(x, kind, oh, ow):
    """Reference output and backward closure, operating on valid image slices.

    SAME max/average use 3x3 stride 2. Average excludes padding; max ties choose
    the first valid row-major entry. Adaptive bins use floor/ceil endpoints.
    """
    batch, height, width, channels = x.shape
    output = np.empty((batch, oh, ow, channels), dtype=np.float64)
    bins = []
    for y in range(oh):
        for col in range(ow):
            if kind == "adaptive":
                y0, y1 = y*height//oh, math.ceil((y+1)*height/oh)
                x0, x1 = col*width//ow, math.ceil((col+1)*width/ow)
            else:
                py = max((oh-1)*2+3-height, 0)//2
                px = max((ow-1)*2+3-width, 0)//2
                y0, y1 = max(0, 2*y-py), min(height, 2*y-py+3)
                x0, x1 = max(0, 2*col-px), min(width, 2*col-px+3)
            patch = x[:, y0:y1, x0:x1, :].reshape(batch, -1, channels)
            winner = None
            if kind == "max":
                winner = np.argmax(patch, axis=1)
                output[:, y, col, :] = np.take_along_axis(patch, winner[:, None, :], axis=1)[:, 0, :]
            else:
                output[:, y, col, :] = patch.mean(axis=1)
            bins.append((y, col, y0, y1, x0, x1, winner))

    def backward(grad):
        dx = np.zeros_like(x)
        for y, col, y0, y1, x0, x1, winner in bins:
            g = grad[:, y, col, :]
            if winner is None:
                dx[:, y0:y1, x0:x1, :] += g[:, None, None, :]/((y1-y0)*(x1-x0))
            else:
                for iy in range(y0, y1):
                    for ix in range(x0, x1):
                        dx[:, iy, ix, :] += g*(winner == (iy-y0)*(x1-x0)+ix-x0)
        return dx
    return output, backward


def reference(obs, parameters, upstream, ops):
    x = np.asarray(obs, dtype=np.float64).reshape(-1, 36, 44, 1)
    params = np.asarray(parameters, dtype=np.float64)
    cache = []
    for op in ops:
        if op["kind"] != "conv":
            x, backward = pooling(x, op["kind"], *op["output"][:2])
            cache.append((op, backward))
            continue
        original_shape = x.shape
        if op["dense"]:
            x = x.reshape(x.shape[0], 1, 1, -1)
        batch, height, width, channels = x.shape
        oh, ow, co = op["output"]
        kernel, stride, dilation = op["kernel"], op["stride"], op["dilation"]
        effective = (kernel-1)*dilation+1
        py = max((oh-1)*stride+effective-height, 0)
        px = max((ow-1)*stride+effective-width, 0)
        padded = np.pad(x, ((0, 0), (py//2, py-py//2), (px//2, px-px//2), (0, 0)))
        patches = np.stack([padded[:, ky*dilation:ky*dilation+oh*stride:stride,
                                   kx*dilation:kx*dilation+ow*stride:stride, :]
                            for ky in range(kernel) for kx in range(kernel)], axis=-2)
        rows = patches.reshape(batch*oh*ow, -1)
        offset = op["offset"]
        size = op["sizes"][0]
        weight = params[offset:offset+size].reshape(co, -1)
        bias = params[offset+size:offset+size+co]
        coefficients = params[offset+size+co:offset+sum(op["sizes"])]
        pre = (rows@weight.T+bias).reshape(batch, oh, ow, co)
        if op["residual"]:
            pre += x
        x, derivative, dp = activation(pre, op["activation"], coefficients)
        cache.append((op, original_shape, padded.shape, py//2, px//2,
                      rows, weight, derivative, dp))
    output = x.reshape(upstream.shape)
    grad = np.asarray(upstream, dtype=np.float64).reshape(x.shape)
    parameter_grad = np.zeros_like(params)
    for saved in reversed(cache):
        op = saved[0]
        if op["kind"] != "conv":
            grad = saved[1](grad)
            continue
        _, original_shape, padded_shape, py, px, rows, weight, derivative, dp = saved
        offset, size = op["offset"], op["sizes"][0]
        co = op["co"]
        if op["activation"] >= 3:
            parameter_grad[offset+size+co:offset+size+co+12] = (grad[..., None]*dp).sum(axis=(0, 1, 2, 3))
        grad = grad*derivative
        g = grad.reshape(-1, co)
        parameter_grad[offset:offset+size] = (g.T@rows).ravel()
        parameter_grad[offset+size:offset+size+co] = g.sum(axis=0)
        kernel, stride, dilation = op["kernel"], op["stride"], op["dilation"]
        batch, oh, ow, _ = grad.shape
        patch_grad = (g@weight).reshape(batch, oh, ow, kernel, kernel, padded_shape[-1])
        padded_grad = np.zeros(padded_shape, dtype=np.float64)
        for ky in range(kernel):
            for kx in range(kernel):
                padded_grad[:, ky*dilation:ky*dilation+oh*stride:stride,
                            kx*dilation:kx*dilation+ow*stride:stride, :] += patch_grad[:, :, :, ky, kx, :]
        height, width = (1, 1) if op["dense"] else original_shape[1:3]
        dx = padded_grad[:, py:py+height, px:px+width, :]
        if op["residual"]:
            dx += grad
        grad = dx.reshape(original_shape)
    return output, parameter_grad
