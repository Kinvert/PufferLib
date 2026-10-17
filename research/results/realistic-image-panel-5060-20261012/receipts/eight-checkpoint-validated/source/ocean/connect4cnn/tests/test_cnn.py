"""Validate every allowed CNN shape, including zero-block stages and GAP."""
import argparse
import ctypes
from pathlib import Path

import numpy as np

from test_impala import fixture, reference, unpack


def check(library):
    lib = ctypes.CDLL(str(Path(library).resolve()))
    array = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    lib.impalatest_init.argtypes = [ctypes.c_int] * 5
    lib.impalatest_init.restype = ctypes.c_int
    lib.impalatest_run.argtypes = [array] * 5 + [ctypes.c_int]
    lib.impalatest_run.restype = None
    lib.impalatest_close.argtypes = []
    lib.impalatest_close.restype = None
    for channels in (8, 16, 32):
        for blocks in (0, 1, 2):
            for gap in (0, 1):
                B, hidden = 2, 16
                data = fixture(B, hidden, gap, np.float32, channels, blocks)
                assert lib.impalatest_init(B, hidden, channels, blocks, gap) == len(data[1])
                try:
                    expected = reference(*(a.astype(np.float64) for a in data), gap,
                                         channels=channels, blocks=blocks)
                    first = None
                    for graph in (0, 0, 1, 1):
                        actual = [np.empty((B, hidden), np.float32), np.empty_like(data[1])]
                        lib.impalatest_run(*data, *actual, graph)
                        for got, want in zip(actual, expected):
                            np.testing.assert_allclose(got, want, rtol=5e-4, atol=4e-5)
                        if first is None:
                            first = [a.copy() for a in actual]
                        else:
                            for got, want in zip(actual, first):
                                np.testing.assert_array_equal(got, want)
                    if channels == 8:
                        precise = [a.astype(np.float64) for a in data]
                        params = unpack(precise[1], hidden, gap, channels, blocks)
                        grads = unpack(expected[1], hidden, gap, channels, blocks)
                        for param, grad in zip(params, grads):
                            idx = np.unravel_index(np.abs(grad).argmax(), grad.shape)
                            original = param[idx]
                            param[idx] = original + 1e-6
                            plus = np.sum(reference(*precise, gap, backward=False, channels=channels, blocks=blocks)[0] * precise[2])
                            param[idx] = original - 1e-6
                            minus = np.sum(reference(*precise, gap, backward=False, channels=channels, blocks=blocks)[0] * precise[2])
                            param[idx] = original
                            np.testing.assert_allclose((plus - minus) / 2e-6, grad[idx], rtol=5e-4, atol=2e-8)
                    print(f"PASS channels={channels} blocks={blocks} gap={gap}: forward/all gradients, eager/graph repeatability, rollout parity", flush=True)
                finally:
                    lib.impalatest_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    check(parser.parse_args().library)
