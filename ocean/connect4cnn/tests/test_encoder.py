"""Independent NumPy reference; optional comparison with the actual CUDA encoder."""
import argparse
import ctypes
from pathlib import Path

import numpy as np


def reference(obs, conv_w, proj_w, upstream):
    batch = len(obs)
    image = obs.reshape(batch, 36, 44)
    # Independent spatial slicing, keeping row/column/channel order explicit.
    conv = np.empty((batch, 9, 11, 8), dtype=obs.dtype)
    for y in range(9):
        for x in range(11):
            patch = image[:, y * 4:y * 4 + 4, x * 4:x * 4 + 4].reshape(batch, 16)
            conv[:, y, x] = patch @ conv_w.T
    flat = np.maximum(conv, 0).reshape(batch, 792)
    output = flat @ proj_w.T
    proj_grad = upstream.T @ flat
    grad_conv = (upstream @ proj_w).reshape(conv.shape) * (conv > 0)
    conv_grad = np.zeros_like(conv_w)
    for y in range(9):
        for x in range(11):
            patch = image[:, y * 4:y * 4 + 4, x * 4:x * 4 + 4].reshape(batch, 16)
            conv_grad += grad_conv[:, y, x].T @ patch
    return output, conv_grad, proj_grad


def fixture(batch, hidden, dtype):
    rng = np.random.default_rng(731 + batch)
    obs = rng.choice([0, 0.5, 1], size=(batch, 36 * 44)).astype(dtype)
    conv_w = rng.uniform(0.01, 0.1, size=(8, 16)).astype(dtype)
    conv_w[1::2] *= -1  # Both active and inactive ReLUs, safely away from kinks.
    proj_w = rng.normal(0, 0.03, size=(hidden, 792)).astype(dtype)
    upstream = rng.normal(0, 0.1, size=(batch, hidden)).astype(dtype)
    return obs, conv_w, proj_w, upstream


def check_reference():
    data = fixture(2, 16, np.float64)
    _, conv_grad, proj_grad = reference(*data)
    for weights, analytic in [(data[1], conv_grad), (data[2], proj_grad)]:
        for index in np.linspace(0, weights.size - 1, 16, dtype=int):
            old = weights.flat[index]
            eps = 1e-6
            weights.flat[index] = old + eps
            plus = np.sum(reference(*data)[0] * data[3])
            weights.flat[index] = old - eps
            minus = np.sum(reference(*data)[0] * data[3])
            weights.flat[index] = old
            np.testing.assert_allclose((plus - minus) / (2 * eps), analytic.flat[index], rtol=1e-6, atol=1e-8)
    obs, conv_w, proj_w, upstream = data
    base = reference(*data)[0]
    # Each board cell must affect the encoder, including the rightmost column.
    for row in range(6):
        for column in range(7):
            changed = obs.copy().reshape(2, 36, 44)
            changed[0, row * 6:(row + 1) * 6, 1 + column * 6:1 + (column + 1) * 6] += 0.25
            assert not np.array_equal(base, reference(changed.reshape(obs.shape), conv_w, proj_w, upstream)[0])
    print("PASS: independent CPU reference finite differences and all 42 board-cell coverage checks")


def check_cuda(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    array = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    lib.c4test_init.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.c4test_init.restype = ctypes.c_int
    lib.c4test_run.argtypes = [array] * 7 + [ctypes.c_int]
    lib.c4test_run.restype = None
    lib.c4test_close.argtypes = []
    lib.c4test_close.restype = None
    for batch, hidden in [(1, 16), (3, 32), (32, 128)]:
        if not lib.c4test_init(batch, hidden):
            raise SystemExit("GPU checks NOT RUN: no accessible CUDA device")
        try:
            data = fixture(batch, hidden, np.float32)
            for blank in [False, True]:
                if blank:
                    data[0].fill(0)
                expected = reference(*(x.astype(np.float64) for x in data))
                first = None
                for graph in [0, 0, 1, 1]:
                    actual = [np.empty((batch, hidden), np.float32),
                              np.empty_like(data[1]), np.empty_like(data[2])]
                    lib.c4test_run(*data, *actual, graph)
                    for got, want in zip(actual, expected):
                        np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-5)
                    if first is None:
                        first = [x.copy() for x in actual]
                    else:
                        for got, want in zip(actual, first):
                            np.testing.assert_array_equal(got, want)
            print(f"PASS: native CUDA forward/weight gradients, rollout/train parity, eager/graph repeatability B={batch} H={hidden}")
        finally:
            lib.c4test_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", help="Compiled float32 test_encoder.so; omitted for CPU reference checks only")
    args = parser.parse_args()
    check_reference()
    if args.library:
        check_cuda(args.library)
    else:
        print("GPU implementation has NOT been checked by this CPU-only run.")
