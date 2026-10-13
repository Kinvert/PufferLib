"""Independent NumPy checks for flexible convolution, residuals, and pooling."""
import argparse
import ctypes
from pathlib import Path

import numpy as np

from test_impala import pool_reference


def plan(settings, hidden=128):
    depth, projection, gap, *stages = settings
    H, W, C = 36, 44, 1
    ops = []
    for stage in range(depth):
        co, k, s, pool, skip = stages[stage*5:stage*5+5]
        ops.append(("conv", H, W, C, co, k, s, False))
        H, W, C = (H+s-1)//s, (W+s-1)//s, co
        if skip:
            ops.append(("conv", H, W, C, C, 3, 1, True))
        if pool:
            ops.append(("max" if pool == 1 else "average", H, W, C, C, 3, 2, False))
            H, W = (H+1)//2, (W+1)//2
    if gap:
        ops.append(("global", H, W, C, C, 1, 1, False))
        H = W = 1
    ops.append(("linear", 1, 1, H*W*C, projection, 1, 1, False))
    if projection != hidden:
        ops.append(("linear", 1, 1, projection, hidden, 1, 1, False))
    return ops


def fixture(ops, blank=False):
    rng = np.random.default_rng(173)
    shapes = []
    for kind, H, W, ci, co, k, s, skip in ops:
        if kind in ("conv", "linear"):
            shapes.extend([(co, k*k*ci), (co,)])
    arrays = [rng.normal(0, .4/np.sqrt(sh[1]), sh) if len(sh) == 2 else rng.choice([-.2, .2], sh) for sh in shapes]
    obs = np.zeros((2, 1584)) if blank else rng.uniform(0, 1, (2, 1584))
    return (obs.astype(np.float32), np.concatenate([a.ravel() for a in arrays]).astype(np.float32),
            rng.normal(0, .1, (2, 128)).astype(np.float32)), shapes


def reference(data, ops, shapes):
    obs, packed, upstream = data
    params, offset = [], 0
    for shape in shapes:
        n = int(np.prod(shape)); params.append(packed[offset:offset+n].reshape(shape)); offset += n
    x = obs.reshape(len(obs), 36, 44, 1)
    cache, cursor = [], 0
    for kind, H, W, ci, co, k, s, skip in ops:
        original = x.shape
        if kind == "global":
            cache.append((kind, original)); x = x.mean(axis=(1, 2), keepdims=True); continue
        if kind == "max":
            x, back = pool_reference(x); cache.append((kind, back)); continue
        if kind == "average":
            out = np.empty((len(x), (H+1)//2, (W+1)//2, ci), dtype=x.dtype)
            windows = []
            for y in range(out.shape[1]):
                for z in range(out.shape[2]):
                    top, left = y*2-H%2, z*2-W%2
                    ys, xs = slice(max(top, 0), min(top+3, H)), slice(max(left, 0), min(left+3, W))
                    out[:, y, z] = x[:, ys, xs].mean(axis=(1, 2))
                    windows.append((y, z, ys, xs))
            cache.append((kind, original, windows)); x = out; continue
        if kind == "linear":
            x = x.reshape(len(x), 1, 1, ci)
        oh, ow = (H+s-1)//s, (W+s-1)//s
        dh, dw = max((oh-1)*s+k-H, 0), max((ow-1)*s+k-W, 0)
        padded = np.pad(x, ((0,0), (dh//2, dh-dh//2), (dw//2, dw-dw//2), (0,0)))
        patches = np.stack([padded[:, y*s:y*s+k, z*s:z*s+k].reshape(len(x), -1)
                            for y in range(oh) for z in range(ow)], axis=1).reshape(len(x), oh, ow, -1)
        pre = patches @ params[cursor].T + params[cursor+1]
        if skip:
            pre += x
        cache.append((kind, original, patches, pre, padded.shape, dh//2, dw//2, cursor, k, s, skip))
        cursor += 2; x = np.maximum(pre, 0)
    out = x.reshape(upstream.shape)
    g = upstream.reshape(x.shape)
    grads = [None] * len(params)
    for entry in reversed(cache):
        kind = entry[0]
        if kind == "global":
            original = entry[1]
            g = np.broadcast_to(g/(original[1]*original[2]), original).copy(); continue
        if kind == "max":
            g = entry[1](g); continue
        if kind == "average":
            original, windows = entry[1:]
            dx = np.zeros(original, dtype=g.dtype)
            for y, z, ys, xs in windows:
                count = (ys.stop-ys.start)*(xs.stop-xs.start)
                dx[:, ys, xs] += g[:, y:y+1, z:z+1] / count
            g = dx; continue
        _, original, patches, pre, padded_shape, ph, pw, cursor, k, s, skip = entry
        g = g * (pre > 0)
        rows = g.reshape(-1, g.shape[-1])
        grads[cursor] = rows.T @ patches.reshape(-1, patches.shape[-1])
        grads[cursor+1] = rows.sum(axis=0)
        gp = (g @ params[cursor]).reshape(len(g), g.shape[1], g.shape[2], k, k, padded_shape[-1])
        dx = np.zeros(padded_shape, dtype=g.dtype)
        for y in range(g.shape[1]):
            for z in range(g.shape[2]):
                dx[:, y*s:y*s+k, z*s:z*s+k] += gp[:, y, z]
        H = original[1] if kind == "conv" else 1
        W = original[2] if kind == "conv" else 1
        dx = dx[:, ph:ph+H, pw:pw+W]
        if skip:
            dx += g
        g = dx.reshape(original)
    return out, np.concatenate([g.ravel() for g in grads])


def cases():
    base = [3, 32, 0, 8, 8, 4, 0, 0, 16, 3, 2, 0, 0, 8, 3, 1, 0, 0]
    for depth in (1, 2, 3):
        for gap in (0, 1):
            for pool in (0, 1, 2):
                for skip in (0, 1):
                    x = base.copy(); x[:3] = depth, 16 if gap else 32, gap
                    for stage in range(3):
                        x[6+stage*5], x[7+stage*5] = pool, skip
                    yield x
    for stride in (4, 8):
        for kernel in range(1, 9):
            x = base.copy(); x[4:6] = kernel, stride
            x[9], x[14] = kernel%5+1, (kernel+2)%5+1
            x[6], x[11], x[16] = 1, 2, 1
            yield x
    rng = np.random.default_rng(307)
    for _ in range(24):
        x = [int(rng.integers(1,4)), int(rng.choice([16,32,64,128])), int(rng.integers(2))]
        for stage in range(3):
            x.extend([int(rng.choice([8,16,32])), int(rng.integers(1,9 if stage == 0 else 6)),
                      int(rng.choice([4,8] if stage == 0 else [1,2,4])), int(rng.integers(3)), int(rng.integers(2))])
        yield x
    yield [3,128,0, 32,8,4,0,1, 32,5,1,0,1, 32,5,1,0,1]
    yield [3,16,1, 8,1,8,2,1, 8,5,4,1,1, 8,5,4,2,1]


def check(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    array = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    integers = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    lib.flextest_init.argtypes = [ctypes.c_int, ctypes.c_int, integers]
    lib.flextest_init.restype = ctypes.c_int
    lib.naturetest_run.argtypes = [array]*5+[ctypes.c_int]
    lib.naturetest_close.argtypes = []
    specs = list(cases())
    for index, settings in enumerate(specs):
        ops = plan(settings)
        data, shapes = fixture(ops, blank=index in (5, 17, 29))
        precise = [a.astype(np.float64) for a in data]
        expected = reference(precise, ops, shapes)
        assert lib.flextest_init(2, 128, np.array(settings, np.int32)) == len(data[1])
        try:
            first = None
            for graph in (0,0,1,1):
                actual = [np.empty_like(data[2]), np.empty_like(data[1])]
                lib.naturetest_run(*data, *actual, graph)
                for got, want in zip(actual, expected):
                    np.testing.assert_allclose(got, want, rtol=8e-4, atol=6e-5)
                if first is None:
                    first = [a.copy() for a in actual]
                else:
                    for got, want in zip(actual, first):
                        np.testing.assert_array_equal(got, want)
        finally:
            lib.naturetest_close()
        if index in (1, 3, 5, 13, 15, 17, 25, 27, 29, len(specs)-1):
            offset = 0
            for shape in shapes:
                n = int(np.prod(shape)); j = offset+int(np.abs(expected[1][offset:offset+n]).argmax())
                old = precise[1][j]; precise[1][j] = old+1e-6
                plus = np.sum(reference(precise, ops, shapes)[0]*precise[2])
                precise[1][j] = old-1e-6
                minus = np.sum(reference(precise, ops, shapes)[0]*precise[2]); precise[1][j] = old
                # Blank inputs intentionally exercise tied maxima; use numerical checks on nonblank cases.
                if index not in (5,17,29):
                    np.testing.assert_allclose((plus-minus)/2e-6, expected[1][j], rtol=8e-4, atol=3e-8)
                offset += n
        print(f"PASS flex case={index} settings={settings}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--library", required=True)
    check(parser.parse_args().library)
