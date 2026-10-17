"""Independent float64 reference for the 54 compact CNN shapes."""
import argparse
import ctypes
import itertools
from pathlib import Path

import numpy as np


def layers(channels, depth, stride, projection, hidden=128):
    spec = [(8, stride, channels), (4, 2, 2 * channels), (3, 1, 2 * channels)][:depth]
    return spec + [(0, 1, projection)] + ([(0, 1, hidden)] if projection != hidden else [])


def fixture(spec, batch=2):
    rng = np.random.default_rng(713)
    H, W, C = 36, 44, 1
    shapes = []
    for k, s, co in spec:
        shapes.extend([(co, k * k * C if k else H * W * C), (co,)])
        H, W, C = ((H - k) // s + 1, (W - k) // s + 1, co) if k else (1, 1, co)
    params = [rng.normal(0, .5 / np.sqrt(sh[1]), sh) if len(sh) == 2 else rng.choice([-.2, .2], sh) for sh in shapes]
    return (rng.choice([0, .5, 1], (batch, 1584)).astype(np.float32),
            np.concatenate([p.ravel() for p in params]).astype(np.float32),
            rng.normal(0, .1, (batch, C)).astype(np.float32)), shapes


def reference(data, spec, shapes):
    obs, values, upstream = data
    arrays, offset = [], 0
    for shape in shapes:
        n = int(np.prod(shape))
        arrays.append(values[offset:offset+n].reshape(shape))
        offset += n
    x = obs.reshape(len(obs), 36, 44, 1)
    cache = []
    for i, (k, s, co) in enumerate(spec):
        original = x.shape
        if not k:
            patches = x.reshape(len(x), 1, 1, -1)
        else:
            H, W = (x.shape[1] - k) // s + 1, (x.shape[2] - k) // s + 1
            patches = np.stack([x[:, y*s:y*s+k, z*s:z*s+k].reshape(len(x), -1)
                                for y in range(H) for z in range(W)], axis=1).reshape(len(x), H, W, -1)
        pre = patches @ arrays[2*i].T + arrays[2*i+1]
        cache.append((original, patches, pre))
        x = np.maximum(pre, 0)
    output = x.reshape(upstream.shape)
    g = upstream.reshape(x.shape)
    grads = [None] * len(arrays)
    for i in range(len(spec)-1, -1, -1):
        k, s, co = spec[i]
        original, patches, pre = cache[i]
        g = g * (pre > 0)
        rows = g.reshape(-1, co)
        grads[2*i] = rows.T @ patches.reshape(-1, patches.shape[-1])
        grads[2*i+1] = rows.sum(axis=0)
        if i:
            gp = g @ arrays[2*i]
            if not k:
                g = gp.reshape(original)
            else:
                dx = np.zeros(original, dtype=g.dtype)
                gp = gp.reshape(len(g), g.shape[1], g.shape[2], k, k, original[-1])
                for y in range(g.shape[1]):
                    for z in range(g.shape[2]):
                        dx[:, y*s:y*s+k, z*s:z*s+k] += gp[:, y, z]
                g = dx
    return output, np.concatenate([g.ravel() for g in grads])


def check(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    array = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    lib.compacttest_init.argtypes = [ctypes.c_int] * 6
    lib.compacttest_init.restype = ctypes.c_int
    lib.naturetest_init.argtypes = [ctypes.c_int] * 2
    lib.naturetest_init.restype = ctypes.c_int
    lib.naturetest_run.argtypes = [array] * 5 + [ctypes.c_int]
    lib.naturetest_close.argtypes = []
    for c, d, s, p in itertools.product((8, 16, 32), (1, 2, 3), (2, 4), (32, 64, 128)):
        spec = layers(c, d, s, p)
        data, shapes = fixture(spec)
        precise = [a.astype(np.float64) for a in data]
        expected = reference(precise, spec, shapes)
        assert lib.compacttest_init(2, 128, c, d, s, p) == len(data[1])
        try:
            first = None
            for graph in (0, 0, 1, 1):
                actual = [np.empty_like(data[2]), np.empty_like(data[1])]
                lib.naturetest_run(*data, *actual, graph)
                for got, want in zip(actual, expected):
                    np.testing.assert_allclose(got, want, rtol=5e-4, atol=4e-5)
                if first is None:
                    first = [a.copy() for a in actual]
                else:
                    for got, want in zip(actual, first):
                        np.testing.assert_array_equal(got, want)
        finally:
            lib.naturetest_close()
        if c == 8 and p == 32:
            offset = 0
            for shape in shapes:
                n = int(np.prod(shape))
                idx = offset + int(np.abs(expected[1][offset:offset+n]).argmax())
                old = precise[1][idx]
                precise[1][idx] = old + 1e-6
                plus = np.sum(reference(precise, spec, shapes)[0] * precise[2])
                precise[1][idx] = old - 1e-6
                minus = np.sum(reference(precise, spec, shapes)[0] * precise[2])
                precise[1][idx] = old
                np.testing.assert_allclose((plus-minus)/2e-6, expected[1][idx], rtol=5e-4, atol=2e-8)
                offset += n
        if (c, d, s, p) == (32, 3, 4, 128):
            assert lib.naturetest_init(2, 128) == len(data[1])
            try:
                actual = [np.empty_like(data[2]), np.empty_like(data[1])]
                lib.naturetest_run(*data, *actual, 1)
                for got, want in zip(actual, first):
                    np.testing.assert_array_equal(got, want)
            finally:
                lib.naturetest_close()
        print(f"PASS compact channels={c} depth={d} stride={s} projection={p}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    check(parser.parse_args().library)
