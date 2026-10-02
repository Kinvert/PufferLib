"""GPU-only sampled finite differences and eager/graph/actor repeatability.

No CPU encoder implementation or CPU training. Each perturbed forward and
analytical backward executes the native CUDA encoder. Independent forward
references and training/checkpoint checks remain separate acceptance gates.
"""
import argparse
import ctypes
import json
from pathlib import Path
import tempfile

import numpy as np


def cases():
    # Depth/readout/activation coverage, odd padding, dilation 1..4, residuals,
    # pool choices, 1x1 kernels and an adaptive grid larger than its input map.
    for activation in range(5):
        yield f"activation-{activation}", 2, 2, activation, [(8, 3, 4, 2, 0, 1), (8, 3, 2, 3, 2, 0)]
    yield "relu-residual-max", 2, 0, 0, [(8, 8, 8, 1, 1, 2), (16, 1, 1, 4, 0, 0)]
    yield "silu-four-stages", 4, 1, 1, [(8, 3, 4, 1, 0, 0), (8, 3, 2, 2, 0, 0), (8, 3, 1, 4, 0, 0), (8, 1, 1, 4, 0, 0)]
    yield "gelu-upsample-bins", 1, 3, 2, [(8, 7, 8, 4, 2, 0)]
    yield "rational-residual-max", 1, 1, 4, [(8, 5, 4, 1, 1, 2)]
    yield "prelu-stride-one", 1, 1, 3, [(8, 1, 1, 1, 0, 0)]


def run_case(lib, directory, name, depth, readout, activation, stages):
    lines = ["[policy]", "encoder = 5", f"cnn_depth = {depth}",
             "cnn_projection = 16", f"cnn_readout = {readout}",
             f"cnn_projection_activation = {activation}"]
    for stage, settings in enumerate(stages, 1):
        for key, value in zip(("channels", "kernel", "stride", "dilation", "pool", "residual"), settings):
            lines.append(f"cnn_{key}_{stage} = {value}")
        lines.append(f"cnn_activation_{stage} = {activation}")
    config = directory / f"{name}.ini"
    config.write_text("\n".join(lines) + "\n")
    n = lib.flex2test_init(str(config).encode(), 2, 32, 173)
    try:
        metadata = np.zeros((72, 4), dtype=np.int64)
        count = lib.flex2test_tensors(metadata)
        metadata = metadata[:count]
        initialized = np.empty(n, dtype=np.float32)
        lib.flex2test_params(initialized)
        assert np.isfinite(initialized).all()
        rng = np.random.default_rng(173)
        parameters = np.zeros(n, dtype=np.float32)
        for offset, size, kind, act in metadata:
            segment = parameters[offset:offset+size]
            if kind == 0:
                segment[:] = rng.normal(0, .025, size)
            elif kind == 1:
                segment[:] = rng.uniform(-.1, .1, size)
            elif act == 3:
                segment[0] = .25
            else:
                segment[:10] = [.05, .85, .03, -.01, .005, -.001, .03, .01, .005, .001]
        obs = rng.uniform(0, 1, (2, 1584)).astype(np.float32)
        upstream = rng.normal(0, .1, (2, 32)).astype(np.float32)

        def evaluate(values, graph=0):
            output = np.empty((2, 32), dtype=np.float32)
            gradient = np.empty(n, dtype=np.float32)
            lib.flex2test_run(obs, values, upstream, output, gradient, graph)
            assert np.isfinite(output).all() and np.isfinite(gradient).all()
            return output, gradient

        output, gradient = evaluate(parameters)
        for graph in (0, 1, 1):
            repeated = evaluate(parameters, graph)
            assert np.array_equal(output, repeated[0]), (name, "forward repeatability")
            assert np.array_equal(gradient, repeated[1]), (name, "gradient repeatability")
        # Prefer coordinates carrying a nonzero gradient; also sample endpoints
        # and every coefficient (including padded slots expected to be unused).
        indices = set()
        for offset, size, kind, _ in metadata:
            if kind == 2:
                indices.update(range(int(offset), int(offset+size)))
            else:
                local = gradient[offset:offset+size]
                indices.update(int(offset+i) for i in np.argsort(np.abs(local))[-3:])
                indices.update((int(offset), int(offset+size-1)))
        worst, checked, skipped_kinks = 0., 0, 0
        for index in sorted(indices):
            step = np.float32(.001)
            plus, minus = parameters.copy(), parameters.copy()
            plus[index] += step; minus[index] -= step
            out_plus, _ = evaluate(plus)
            out_minus, _ = evaluate(minus)
            # Scalar objective uses float64 host summation of GPU outputs only.
            forward = np.sum((out_plus.astype(np.float64)-output)*upstream)/float(step)
            backward = np.sum((output.astype(np.float64)-out_minus)*upstream)/float(step)
            if abs(forward-backward) > .01 + .15*max(abs(forward), abs(backward)):
                # ReLU/max/abs kinks need a dedicated boundary test, not a false
                # smooth finite-difference pass. Preserve the number skipped.
                skipped_kinks += 1
                continue
            numerical = (forward+backward)/2
            error = abs(numerical-float(gradient[index]))
            tolerance = .00015 + .025*max(abs(numerical), abs(float(gradient[index])))
            assert error <= tolerance, (name, index, numerical, float(gradient[index]), tolerance)
            worst = max(worst, error); checked += 1
        assert checked > 0 and skipped_kinks <= max(3, len(indices)//10), (name, checked, skipped_kinks)
        return dict(case=name, allocated_parameters=int(n), checked_coordinates=checked,
                    skipped_nonsmooth_coordinates=skipped_kinks, max_absolute_error=worst)
    finally:
        lib.flex2test_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    lib = ctypes.CDLL(str(args.library.resolve()))
    array = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    lib.flex2test_init.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
    lib.flex2test_init.restype = ctypes.c_long
    lib.flex2test_tensors.argtypes = [np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")]
    lib.flex2test_tensors.restype = ctypes.c_int
    lib.flex2test_params.argtypes = [array]; lib.flex2test_params.restype = None
    lib.flex2test_run.argtypes = [array, array, array, array, array, ctypes.c_int]
    lib.flex2test_run.restype = None
    lib.flex2test_close.argtypes = []; lib.flex2test_close.restype = None
    report = dict(status="running", method="native GPU sampled finite differences", cases=[])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    try:
        with tempfile.TemporaryDirectory(prefix="flex2-test-", dir=args.report.parent) as directory:
            for case in cases():
                report["current_case"] = case[0]
                args.report.write_text(json.dumps(report, indent=2) + "\n")
                result = run_case(lib, Path(directory), *case)
                report["cases"].append(result)
                args.report.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(result), flush=True)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
