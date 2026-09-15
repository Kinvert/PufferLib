"""Independent NumPy Nature reference, finite differences, and native CUDA checks."""
import argparse
import ctypes
from pathlib import Path

import numpy as np


def shapes(hidden):
    return [(32, 64), (32,), (64, 512), (64,), (64, 576), (64,), (hidden, 128), (hidden,)]


def unpack(values, hidden):
    arrays, offset = [], 0
    for shape in shapes(hidden):
        count = int(np.prod(shape))
        arrays.append(values[offset:offset + count].reshape(shape))
        offset += count
    assert offset == len(values)
    return arrays


def reference(obs, values, upstream):
    B, hidden = upstream.shape
    params = unpack(values, hidden)
    x = obs.reshape(B, 36, 44, 1)
    cache = []
    for i, (kernel, stride) in enumerate([(8, 4), (4, 2), (3, 1)]):
        H, W, C = x.shape[1:]
        oh, ow = (H - kernel) // stride + 1, (W - kernel) // stride + 1
        patches = np.empty((B, oh, ow, kernel * kernel * C), dtype=x.dtype)
        for y in range(oh):
            for col in range(ow):
                patches[:, y, col] = x[:, y * stride:y * stride + kernel, col * stride:col * stride + kernel].reshape(B, -1)
        pre = patches @ params[2 * i].T + params[2 * i + 1]
        cache.append((x.shape, patches, pre, kernel, stride))
        x = np.maximum(pre, 0)
    flat = x.reshape(B, -1)
    final = flat @ params[6].T + params[7]
    output = np.maximum(final, 0)
    g = upstream * (final > 0)
    grads = [None] * 8
    grads[6], grads[7] = g.T @ flat, g.sum(axis=0)
    g = (g @ params[6]).reshape(x.shape)
    for i in range(2, -1, -1):
        input_shape, patches, pre, kernel, stride = cache[i]
        g = g * (pre > 0)
        rows = g.reshape(-1, g.shape[-1])
        grads[2 * i] = rows.T @ patches.reshape(-1, patches.shape[-1])
        grads[2 * i + 1] = rows.sum(axis=0)
        if i:
            gp = (g @ params[2 * i]).reshape(B, g.shape[1], g.shape[2], kernel, kernel, input_shape[-1])
            gx = np.zeros(input_shape, dtype=g.dtype)
            for y in range(g.shape[1]):
                for col in range(g.shape[2]):
                    gx[:, y * stride:y * stride + kernel, col * stride:col * stride + kernel] += gp[:, y, col]
            g = gx
    return output, np.concatenate([g.ravel() for g in grads])


def fixture(B, hidden, dtype):
    rng = np.random.default_rng(918 + B)
    arrays = []
    for shape in shapes(hidden):
        if len(shape) == 2:
            a = rng.normal(0, 0.5 / np.sqrt(shape[1]), shape)
        else:
            a = rng.choice([-0.2, 0.2], shape)
        arrays.append(a.astype(dtype).ravel())
    return (rng.choice([0, 0.5, 1], (B, 1584)).astype(dtype),
            np.concatenate(arrays), rng.normal(0, 0.1, (B, hidden)).astype(dtype))


def check_reference():
    data = fixture(1, 16, np.float64)
    expected = reference(*data)[1]
    offset = 0
    for shape in shapes(16):
        count = int(np.prod(shape))
        # Include a large-gradient entry so an inactive ReLU cannot make every check trivial.
        indices = {offset, offset + count - 1, offset + int(np.argmax(np.abs(expected[offset:offset + count])))}
        for idx in indices:
            old = data[1][idx]
            data[1][idx] = old + 1e-6
            plus = np.sum(reference(*data)[0] * data[2])
            data[1][idx] = old - 1e-6
            minus = np.sum(reference(*data)[0] * data[2])
            data[1][idx] = old
            np.testing.assert_allclose((plus - minus) / 2e-6, expected[idx], rtol=2e-5, atol=1e-8)
        offset += count
    obs = np.zeros((1, 1584), dtype=np.float64)
    values = np.full_like(data[1], 0.01)
    base = reference(obs, values, data[2])[0]
    for row in range(6):
        for col in range(7):
            changed = obs.copy().reshape(1, 36, 44)
            changed[:, row * 6:(row + 1) * 6, 1 + col * 6:1 + (col + 1) * 6] = 1
            assert np.all(reference(changed.reshape(obs.shape), values, data[2])[0] > base)
    print("PASS: Nature CPU reference weight/bias finite differences and all 42 board cells", flush=True)


def check_cuda(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    array = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    lib.naturetest_init.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.naturetest_init.restype = ctypes.c_int
    lib.naturetest_run.argtypes = [array] * 5 + [ctypes.c_int]
    lib.naturetest_run.restype = None
    lib.naturetest_close.argtypes = []
    lib.naturetest_close.restype = None
    for B, hidden in [(1, 16), (3, 32), (32, 128)]:
        data = fixture(B, hidden, np.float32)
        assert lib.naturetest_init(B, hidden) == len(data[1])
        try:
            for blank in [False, True]:
                if blank:
                    data[0].fill(0)
                    # Also test exactly zero preactivations and inactive ReLUs.
                    for a in unpack(data[1], hidden)[1::2]:
                        a.fill(0)
                expected = reference(*(a.astype(np.float64) for a in data))
                first = None
                for graph in [0, 0, 1, 1]:
                    actual = [np.empty((B, hidden), np.float32), np.empty_like(data[1])]
                    lib.naturetest_run(*data, *actual, graph)
                    for got, want in zip(actual, expected):
                        np.testing.assert_allclose(got, want, rtol=3e-4, atol=3e-5)
                    if first is None:
                        first = [a.copy() for a in actual]
                    else:
                        for got, want in zip(actual, first):
                            np.testing.assert_array_equal(got, want)
            print(f"PASS: Nature native forward/all parameter gradients, rollout/train parity, eager/graph repeatability B={B} H={hidden}", flush=True)
        finally:
            lib.naturetest_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library")
    args = parser.parse_args()
    check_reference()
    if args.library:
        check_cuda(args.library)
    else:
        print("GPU checks not run; pass --library to validate the native implementation.")
