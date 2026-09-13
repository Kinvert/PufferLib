"""Independent NumPy IMPALA/GAP reference and native float32 validation."""
import argparse
import ctypes
from pathlib import Path

import numpy as np


def shapes(hidden, gap):
    result = []
    ci = 1
    for co in (16, 32, 32):
        for j in range(5):
            result += [(co, 9 * (ci if j == 0 else co)), (co,)]
        ci = co
    return result + [(hidden, 32 if gap else 960), (hidden,)]


def unpack(values, hidden, gap):
    result, offset = [], 0
    for shape in shapes(hidden, gap):
        n = int(np.prod(shape))
        result.append(values[offset:offset + n].reshape(shape))
        offset += n
    assert offset == len(values)
    return result


def pool_reference(x):
    B, H, W, C = x.shape
    PH, PW = (H + 1) // 2, (W + 1) // 2
    padded = np.pad(x, ((0, 0), (H % 2, 1), (W % 2, 1), (0, 0)), constant_values=-np.inf)
    windows = np.stack([padded[:, y:y + PH * 2:2, col:col + PW * 2:2] for y in range(3) for col in range(3)], axis=-2)
    winner = windows.argmax(axis=-2)
    output = windows.max(axis=-2)

    def backward(g):
        dx = np.zeros_like(padded)
        for y in range(3):
            for col in range(3):
                dx[:, y:y + PH * 2:2, col:col + PW * 2:2] += g * (winner == y * 3 + col)
        return dx[:, H % 2:H % 2 + H, W % 2:W % 2 + W]

    return output, backward


def reference(obs, values, upstream, gap, backward=True):
    B, hidden = upstream.shape
    params = unpack(values, hidden, gap)
    grads = [None] * 32

    def conv(x, i, relu):
        B, H, W, C = x.shape
        activated = np.maximum(x, 0) if relu else x
        padded = np.pad(activated, ((0, 0), (1, 1), (1, 1), (0, 0)))
        patches = np.stack([padded[:, y:y + H, col:col + W] for y in range(3) for col in range(3)], axis=-2).reshape(B, H, W, 9 * C)
        y = patches @ params[2 * i].T + params[2 * i + 1]

        def back(g):
            rows = g.reshape(-1, g.shape[-1])
            grads[2 * i] = rows.T @ patches.reshape(-1, 9 * C)
            grads[2 * i + 1] = rows.sum(axis=0)
            if i == 0:
                return None
            gp = (g @ params[2 * i]).reshape(B, H, W, 3, 3, C)
            dx = np.zeros_like(padded)
            for ky in range(3):
                for kx in range(3):
                    dx[:, ky:ky + H, kx:kx + W] += gp[:, :, :, ky, kx]
            dx = dx[:, 1:-1, 1:-1]
            return dx * (x > 0) if relu else dx

        return y, back

    x = obs.reshape(B, 36, 44, 1)
    stages = []
    for s in range(3):
        x, entry_back = conv(x, s * 5, False)
        x, pool_back = pool_reference(x)
        residuals = []
        for r in range(2):
            skip = x
            x, first = conv(x, s * 5 + 1 + 2 * r, True)
            x, second = conv(x, s * 5 + 2 + 2 * r, True)
            x = x + skip
            residuals.append((first, second))
        stages.append((entry_back, pool_back, residuals))
    activated = np.maximum(x, 0)
    readout = activated.mean(axis=(1, 2)) if gap else activated.reshape(B, -1)
    final = readout @ params[30].T + params[31]
    output = np.maximum(final, 0)
    if not backward:
        return output, None
    g = upstream * (final > 0)
    grads[30], grads[31] = g.T @ readout, g.sum(axis=0)
    g = g @ params[30]
    g = np.broadcast_to(g[:, None, None, :] / 30, x.shape) if gap else g.reshape(x.shape)
    g = g * (x > 0)
    for entry, pool, blocks in reversed(stages):
        for first, second in reversed(blocks):
            g = g + first(second(g))
        g = entry(pool(g))
    return output, np.concatenate([g.ravel() for g in grads])


def fixture(B, hidden, gap, dtype):
    rng = np.random.default_rng(351 + B)
    values = []
    for shape in shapes(hidden, gap):
        a = rng.normal(0, 0.45 / np.sqrt(shape[1]), shape) if len(shape) == 2 else rng.uniform(-0.04, 0.04, shape)
        values.append(a.astype(dtype).ravel())
    return rng.uniform(0, 1, (B, 1584)).astype(dtype), np.concatenate(values), rng.normal(0, 0.1, (B, hidden)).astype(dtype)


def check_reference():
    for gap in [False, True]:
        data = fixture(1, 16, gap, np.float64)
        expected = reference(*data, gap)[1]
        offset = 0
        for shape in shapes(16, gap):
            count = int(np.prod(shape))
            for idx in {offset + count - 1, offset + int(np.argmax(np.abs(expected[offset:offset + count])))}:
                old = data[1][idx]
                data[1][idx] = old + 1e-6
                plus = np.sum(reference(*data, gap, backward=False)[0] * data[2])
                data[1][idx] = old - 1e-6
                minus = np.sum(reference(*data, gap, backward=False)[0] * data[2])
                data[1][idx] = old
                np.testing.assert_allclose((plus - minus) / 2e-6, expected[idx], rtol=5e-4, atol=2e-8)
            offset += count
        obs = np.zeros_like(data[0])
        values = np.full_like(data[1], 0.001)
        base = reference(obs, values, data[2], gap, backward=False)[0]
        for row in range(6):
            for col in range(7):
                changed = obs.copy().reshape(1, 36, 44)
                changed[:, row * 6:(row + 1) * 6, 1 + col * 6:1 + (col + 1) * 6] = 1
                assert np.all(reference(changed.reshape(obs.shape), values, data[2], gap, backward=False)[0] > base)
        print(f"PASS: IMPALA gap={gap} finite differences for all 16 weight/bias pairs and all-cell coverage", flush=True)


def check_cuda(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    array = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    lib.impalatest_init.argtypes = [ctypes.c_int] * 3
    lib.impalatest_init.restype = ctypes.c_int
    lib.impalatest_run.argtypes = [array] * 5 + [ctypes.c_int]
    lib.impalatest_run.restype = None
    lib.impalatest_close.argtypes = []
    lib.impalatest_close.restype = None
    lib.impalatest_pool.argtypes = [ctypes.c_int] * 4 + [array] * 4
    lib.impalatest_pool.restype = None
    rng = np.random.default_rng(7301)
    for H, W in [(4, 6), (5, 7), (1, 1), (2, 3)]:
        x = rng.integers(-4, 0, (2, H, W, 3)).astype(np.float32)
        y, backward = pool_reference(x)
        g = rng.integers(-3, 4, y.shape).astype(np.float32)
        got, dx = np.empty_like(y), np.empty_like(x)
        lib.impalatest_pool(2, H, W, 3, x, g, got, dx)
        np.testing.assert_array_equal(got, y)
        np.testing.assert_array_equal(dx, backward(g))
    print("PASS: native SAME max-pool borders, negative values, ties and overlapping gradients", flush=True)
    for gap in [False, True]:
        for B, hidden in [(1, 16), (3, 32), (32, 128)]:
            data = fixture(B, hidden, gap, np.float32)
            assert lib.impalatest_init(B, hidden, gap) == len(data[1])
            try:
                for blank in [False, True]:
                    if blank:
                        data[0].fill(0)
                        for a in unpack(data[1], hidden, gap)[1::2]:
                            a.fill(0)
                    expected = reference(*(a.astype(np.float64) for a in data), gap)
                    first = None
                    for graph in [0, 0, 1, 1]:
                        actual = [np.empty((B, hidden), np.float32), np.empty_like(data[1])]
                        lib.impalatest_run(*data, *actual, graph)
                        for got, want in zip(actual, expected):
                            np.testing.assert_allclose(got, want, rtol=4e-4, atol=3e-5)
                        if first is None:
                            first = [a.copy() for a in actual]
                        else:
                            for got, want in zip(actual, first):
                                np.testing.assert_array_equal(got, want)
                print(f"PASS: native IMPALA gap={gap} forward/all gradients, rollout/train and eager/graph repeatability B={B} H={hidden}", flush=True)
            finally:
                lib.impalatest_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library")
    args = parser.parse_args()
    check_reference()
    if args.library:
        check_cuda(args.library)
    else:
        print("Native GPU checks not run; pass --library.")
