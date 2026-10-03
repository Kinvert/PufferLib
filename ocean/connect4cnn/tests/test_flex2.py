"""Verify native CUDA encoder 5 against an independent float64 oracle.

Requires the GPU: full forward/all-parameter backward comparisons, boundary
probes, sampled GPU finite differences and eager/graph/actor repeatability.
The oracle is test-only, not a CPU training/validation substitute.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np

from flex2_reference import activation as activation_reference
from flex2_reference import plan, pooling, reference


# Fixed before GPU execution. Never relax these merely to make a failure pass.
RTOL, ATOL = 8e-4, 3e-5
OP_RTOL, OP_ATOL = 2e-5, 2e-6


def compare(actual, expected, label, rtol=RTOL, atol=ATOL):
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), (label, "nonfinite values")
    error = np.abs(actual.astype(np.float64)-expected)
    normalized = error/(atol+rtol*np.abs(expected))
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=label)
    return dict(max_absolute_error=float(error.max()),
                max_tolerance_fraction=float(normalized.max()),
                max_reference_magnitude=float(np.abs(expected).max()),
                nonzero_reference_values=int(np.count_nonzero(expected)),
                values=int(expected.size))


def operator_cases(lib):
    x = np.array([-40, -10, -2, -1, -1e-6, -0., 0., 1e-6, .5, 1, 2, 10, 40], dtype=np.float32)
    for kind in range(5):
        variants = (("learned", "zero-denominator", "denominator-root") if kind == 4
                    else ("default", "negative-slope", "zero-slope") if kind == 3 else ("default",))
        for variant in variants:
            p = np.zeros(12, dtype=np.float32)
            if kind == 3:
                p[0] = {"default": .25, "negative-slope": -.5, "zero-slope": 0}[variant]
            elif kind == 4:
                p[:10] = [.05, .85, .03, -.01, .005, -.001, .03, .01, .005, .001]
                if variant == "zero-denominator":
                    p[6:10] = 0
                elif variant == "denominator-root":
                    p[6:10] = [1, -1, 0, 0]  # Q=0 at x=0 and x=1.
            y, dx, dp = np.empty_like(x), np.empty_like(x), np.empty((len(x), 12), dtype=np.float32)
            lib.flex2test_activation(x, p, y, dx, dp, kind, len(x))
            expected = activation_reference(x, kind, p)
            result = dict(case=f"activation-boundary-{kind}-{variant}")
            for label, actual, target in zip(("output", "input_gradient", "coefficients"), (y, dx, dp), expected):
                result[label] = compare(actual, target, result["case"]+"/"+label, OP_RTOL, OP_ATOL)
            assert np.count_nonzero(dp[:, 10:]) == 0
            yield result
    rng = np.random.default_rng(619)
    for height, width in ((1, 1), (2, 3), (5, 4), (5, 7)):
        for kind, label in ((1, "max"), (2, "average"), (3, "adaptive")):
            for fixture in ("negative", "ties", "ramp"):
                shape = (2, height, width, 3)
                if fixture == "negative":
                    x = -rng.uniform(.1, 2, shape).astype(np.float32)
                elif fixture == "ties":
                    x = np.full(shape, -1, dtype=np.float32)
                else:
                    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)/16-4
                grids = ((1, 1), (2, 2), (4, 4)) if kind == 3 else (((height+1)//2, (width+1)//2),)
                for oh, ow in grids:
                    upstream = rng.normal(0, .25, (2, oh, ow, 3)).astype(np.float32)
                    y, dx = np.empty_like(upstream), np.empty_like(x)
                    lib.flex2test_pool(x, upstream, y, dx, kind, 2, height, width, 3, oh, ow)
                    expected, backward = pooling(x.astype(np.float64), label, oh, ow)
                    name = f"{label}-{height}x{width}-to-{oh}x{ow}-{fixture}"
                    yield dict(case=name,
                               output=compare(y, expected, name+"/output", OP_RTOL, OP_ATOL),
                               input_gradient=compare(dx, backward(upstream.astype(np.float64)), name+"/gradient", OP_RTOL, OP_ATOL))


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
    yield "mixed-stage-activations", 3, 2, 4, [(8, 4, 4, 2, 0, 0, 1), (16, 2, 2, 1, 1, 1, 3), (8, 3, 1, 3, 2, 0, 2)]
    yield "equal-projection-no-extra-head", 1, 0, 1, [(16, 8, 4, 1, 0, 0)], 32, 32, 1
    yield "production-core-size", 2, 1, 3, [(32, 7, 4, 1, 0, 0), (16, 3, 1, 2, 0, 1)], 64, 128, 2
    yield "deep-wide-grammar", 4, 3, 1, [(64, 8, 8, 4, 1, 2), (32, 5, 4, 3, 2, 2), (16, 2, 2, 2, 1, 2), (8, 1, 1, 4, 2, 2)], 128, 128, 1


def run_case(lib, directory, name, depth, readout, activation, stages, projection=16, hidden=32, batch=2):
    lines = ["[policy]", "encoder = 5", f"cnn_depth = {depth}",
             f"cnn_projection = {projection}", f"cnn_readout = {readout}",
             f"cnn_projection_activation = {activation}"]
    for stage, settings in enumerate(stages, 1):
        for key, value in zip(("channels", "kernel", "stride", "dilation", "pool", "residual"), settings):
            lines.append(f"cnn_{key}_{stage} = {value}")
        lines.append(f"cnn_activation_{stage} = {settings[6] if len(settings) == 7 else activation}")
    config = directory / f"{name}.ini"
    config.write_text("\n".join(lines) + "\n")
    ops, expected_metadata, expected_n = plan(config, hidden)
    n = lib.flex2test_init(str(config).encode(), batch, hidden, 173)
    try:
        metadata = np.zeros((72, 4), dtype=np.int64)
        count = lib.flex2test_tensors(metadata)
        metadata = metadata[:count]
        np.testing.assert_array_equal(metadata, expected_metadata, err_msg=name+" parameter layout")
        assert n == expected_n, (name, n, expected_n)
        initialized = np.empty(n, dtype=np.float32)
        lib.flex2test_params(initialized)
        assert np.isfinite(initialized).all()
        for offset, size, kind, act in expected_metadata:
            if kind == 1:
                np.testing.assert_array_equal(initialized[offset:offset+size], 0)
            elif kind == 2:
                expected = np.zeros(12, dtype=np.float32)
                if act == 3:
                    expected[0] = .25
                else:
                    expected[1], expected[7] = 1, .001
                np.testing.assert_array_equal(initialized[offset:offset+size], expected)
        rng = np.random.default_rng(173)
        parameters = np.zeros(n, dtype=np.float32)
        for offset, size, kind, act in metadata:
            segment = parameters[offset:offset+size]
            if kind == 0:
                # Fan-in scaling makes early-layer gradients observable too.
                co = next(op["co"] for op in ops if op.get("offset") == offset)
                segment[:] = rng.normal(0, .5/np.sqrt(size/co), size)
            elif kind == 1:
                segment[:] = rng.uniform(-.1, .1, size)
            elif act == 3:
                segment[0] = .25
            else:
                segment[:10] = [.05, .85, .03, -.01, .005, -.001, .03, .01, .005, .001]
        obs = rng.uniform(0, 1, (batch, 1584)).astype(np.float32)
        upstream = rng.normal(0, .1, (batch, hidden)).astype(np.float32)
        fixture = directory / f"{name}.npz"
        np.savez_compressed(fixture, obs=obs, parameters=parameters, upstream=upstream,
                            metadata=expected_metadata, initialized=initialized)

        def evaluate(values, graph=0):
            output = np.empty((batch, hidden), dtype=np.float32)
            gradient = np.empty(n, dtype=np.float32)
            lib.flex2test_run(obs, values, upstream, output, gradient, graph)
            assert np.isfinite(output).all() and np.isfinite(gradient).all()
            return output, gradient

        output, gradient = evaluate(parameters)
        expected_output, expected_gradient = reference(obs, parameters, upstream, ops)
        output_check = compare(output, expected_output, name+" forward")
        gradient_checks = []
        for offset, size, kind, act in expected_metadata:
            check = compare(gradient[offset:offset+size], expected_gradient[offset:offset+size],
                            f"{name} gradient offset={offset} kind={kind}")
            check.update(offset=int(offset), kind=int(kind), activation=int(act))
            gradient_checks.append(check)
            if kind == 2:
                unused = 1 if act == 3 else 10
                assert np.count_nonzero(gradient[offset+unused:offset+size]) == 0
        for graph in (0, 1, 1):
            repeated = evaluate(parameters, graph)
            assert output.tobytes() == repeated[0].tobytes(), (name, "forward repeatability")
            assert gradient.tobytes() == repeated[1].tobytes(), (name, "gradient repeatability")
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
        worst, checked, skipped_kinks = 0., 0, []
        for index in sorted(indices):
            step = np.float32(.001)
            plus, minus = parameters.copy(), parameters.copy()
            plus[index] += step; minus[index] -= step
            out_plus, _ = evaluate(plus)
            out_minus, _ = evaluate(minus)
            # Scalar objective uses float64 host summation of GPU outputs only.
            forward = np.sum((out_plus.astype(np.float64)-output)*upstream)/float(plus[index]-parameters[index])
            backward = np.sum((output.astype(np.float64)-out_minus)*upstream)/float(parameters[index]-minus[index])
            if abs(forward-backward) > .01 + .15*max(abs(forward), abs(backward)):
                # ReLU/max/abs kinks need a dedicated boundary test, not a false
                # smooth finite-difference pass. Preserve the number skipped.
                skipped_kinks.append(dict(index=index, forward=float(forward), backward=float(backward)))
                continue
            numerical = (forward+backward)/2
            error = abs(numerical-float(gradient[index]))
            tolerance = .00015 + .025*max(abs(numerical), abs(float(gradient[index])))
            assert error <= tolerance, (name, index, numerical, float(gradient[index]), tolerance)
            worst = max(worst, error); checked += 1
        assert checked > 0 and len(skipped_kinks) <= max(3, len(indices)//10), (name, checked, skipped_kinks)
        return dict(case=name, allocated_parameters=int(n), checked_coordinates=checked,
                    skipped_nonsmooth_coordinates=skipped_kinks, max_absolute_error=worst,
                    forward=output_check, parameter_gradients=gradient_checks,
                    fixture=str(fixture), fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                    config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
    finally:
        lib.flex2test_close()


def main():
    if not __debug__:
        raise RuntimeError("Run without -O/PYTHONOPTIMIZE: assertions are acceptance gates")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report already exists; use a fresh path to preserve prior evidence")
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
    lib.flex2test_activation.argtypes = [array]*5 + [ctypes.c_int]*2
    lib.flex2test_activation.restype = None
    lib.flex2test_pool.argtypes = [array]*4 + [ctypes.c_int]*7
    lib.flex2test_pool.restype = None
    args.report.parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=args.report.stem+"-fixtures-", dir=args.report.parent))
    root = Path(__file__).resolve().parents[3]
    source_paths = subprocess.check_output(
        ["git", "ls-files", "--", "src", "vendor", "ocean/connect4cnn", "build.sh"], cwd=root, text=True).splitlines()
    source_paths += [str(Path(__file__).resolve().relative_to(root)),
                     str(Path(__file__).with_name("flex2_reference.py").resolve().relative_to(root))]
    sources = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
               for name in sorted(set(source_paths)) if (root/name).is_file()}
    diff = directory / "working-tree.diff"
    diff.write_bytes(subprocess.check_output(
        ["git", "diff", "HEAD", "--binary", "--", "src", "vendor", "ocean/connect4cnn", "build.sh"], cwd=root))
    network_cases = list(cases())
    report = dict(status="running", method="CUDA versus independent float64 forward/all-parameter backward; GPU finite differences",
                  reference_rtol=RTOL, reference_atol=ATOL, operator_rtol=OP_RTOL, operator_atol=OP_ATOL,
                  command=sys.argv, fixture_seed=173, operator_seed=619,
                  python=platform.python_version(), numpy=np.__version__,
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                  library=str(args.library.resolve()), library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  source_sha256=sources, working_diff=str(diff), fixture_directory=str(directory),
                  expected_operator_cases=69, expected_network_cases=[case[0] for case in network_cases],
                  operators=[], cases=[])
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    try:
        report["current_case"] = "operator boundary probes"
        for result in operator_cases(lib):
            report["operators"].append(result)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        assert len(report["operators"]) == report["expected_operator_cases"]
        for case in network_cases:
            report["current_case"] = case[0]
            args.report.write_text(json.dumps(report, indent=2) + "\n")
            result = run_case(lib, directory, *case)
            report["cases"].append(result)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        assert [case["case"] for case in report["cases"]] == report["expected_network_cases"]
        report.pop("current_case", None)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
