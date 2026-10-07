#!/usr/bin/env python3
"""Bounded GPU-only diagnostic of the retained H256/B2048 quality failure.

Identity readout probes expose first-convolution activations through existing
native/reference APIs. No production/harness/oracle library changes, CPU CNN,
tolerance change, dropped case or acceptance-report upgrade.
"""
import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import secrets
import shutil
import sys

import numpy as np
import gpu_encoder_smoke as math_gate

CASE = "flex_quality-h256-b2048-nonblank"
require, sha, save = math_gate.require, math_gate.sha, math_gate.save


def load_request(root, current=False):
    request = json.loads((root / "request.json").read_text())
    require(request["case"] == CASE and request["acceptance_qualified"] is False, "Wrong diagnostic scope")
    for name, digest in request["inputs_sha256"].items(): require(sha(name) == digest, "Diagnostic input changed")
    if current: require(sha(Path(__file__)) == request["tool_sha256"], "Diagnostic tool changed")
    return request


def worker(root):
    request = load_request(root, current=True)
    require(os.getppid() == request["supervisor_pid"] and os.environ.get("PUFFER_RELU_DIAGNOSTIC_TOKEN") == request["nonce"],
            "Private worker requires its actual scheduled supervisor")
    packet_root = Path(request["failed_packet"])
    math_gate.load(packet_root, current=True)
    source = packet_root / "execution/repeat-0/arrays" / CASE
    obs = np.fromfile(source / "input.f32", dtype="<f4").reshape(2048, 1584)
    original = np.fromfile(source / "parameters.f32", dtype="<f4")
    expected_gradient = np.fromfile(source / "expected-gradient.f64", dtype="<f8")
    actual_gradient = np.fromfile(source / "eager-0-gradient.f32", dtype="<f4")
    difference = actual_gradient - expected_gradient
    bad = np.flatnonzero(np.abs(difference) > 6e-5 + 8e-4 * np.abs(expected_gradient))
    require(len(bad) and np.all(bad < 800), "Diagnostic only covers a first-convolution discrepancy")
    channels = {int(i // 49 if i < 784 else i - 784) for i in bad}
    require(len(channels) == 1, "Need one localized failing convolution channel")
    channel = channels.pop()
    native = ctypes.CDLL(str(packet_root / "libraries/native.so"))
    reference = ctypes.CDLL(str(packet_root / "libraries/reference.so"))
    floats = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    integers = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    native.flextest_init.argtypes = [ctypes.c_int, ctypes.c_int, integers]; native.flextest_init.restype = ctypes.c_int
    native.naturetest_run.argtypes = [floats]*5 + [ctypes.c_int]; native.naturetest_run.restype = None
    native.naturetest_close.restype = None
    reference.cnnref_run.argtypes = [ctypes.c_int]*3 + [ctypes.POINTER(ctypes.c_double)]*6
    reference.cnnref_run.restype = ctypes.c_int
    upstream = np.zeros((2048, 256), dtype=np.float32)
    mismatches, probes = [], []
    for start in (0, 64):
        count = min(64, 99-start)
        weights = np.zeros_like(original); weights[:800] = original[:800]
        for j in range(count):
            weights[800 + j*1584 + (start+j)*16 + channel] = 1
            weights[102240 + j*64 + j] = 1
        directory = root / "probes" / f"positions-{start}"; directory.mkdir(parents=True)
        weights.tofile(directory / "identity-readout-parameters.f32")
        upstream.tofile(directory / "zero-upstream.f32")
        output = np.empty_like(upstream); gradient = np.empty_like(weights)
        expected_output = np.empty(upstream.shape, dtype=np.float64)
        precise = tuple(a.astype(np.float64) for a in (obs, weights, upstream))
        math_gate.reference_call(reference, 0, precise, output=expected_output)
        expected_output.tofile(directory / "reference-output.f64")
        require(native.flextest_init(2048, 256, np.array(math_gate.fixtures.QUALITY, dtype=np.int32)) == weights.size,
                "Probe graph parameter layout differs")
        try: native.naturetest_run(obs, weights, upstream, output, gradient, 0)
        finally: native.naturetest_close()
        output.tofile(directory / "native-output.f32"); gradient.tofile(directory / "native-gradient.f32")
        require(np.isfinite(output).all() and np.isfinite(expected_output).all(), "Nonfinite probe")
        indices = np.argwhere((output[:, :count] > 0) != (expected_output[:, :count] > 0))
        for b, col in indices:
            position = start + int(col); y, x = divmod(position, 11)
            patch = np.zeros((7, 7), dtype=np.float64)
            image = obs[int(b)].reshape(36, 44)
            for ky in range(7):
                for kx in range(7):
                    iy, ix = y*4+ky-1, x*4+kx-1
                    if 0 <= iy < 36 and 0 <= ix < 44: patch[ky, kx] = image[iy, ix]
            # Artifact/pixel inspection only, not a CPU forward/backward pass.
            bias_delta = float(difference[784+channel])
            residual = difference[channel*49:(channel+1)*49] - bias_delta*patch.ravel()
            patch.tofile(directory / f"b{b}-p{position}-pixel-patch.f64")
            mismatches.append(dict(batch_index=int(b), row=y, column=x, channel=channel,
                probe_position=position, native_activation=float(output[b, col]),
                reference_activation=float(expected_output[b, col]),
                original_bias_gradient_delta=bias_delta,
                patch_scaled_gradient_residual_max=float(np.abs(residual).max())))
        probes.append(dict(start=start, positions=count, branch_mismatches=len(indices),
            output_max_absolute_difference=float(np.abs(output-expected_output).max())))
    record = dict(status="diagnostic-completed-not-acceptance", case=CASE, channel=channel,
        original_gradient_violations=bad.tolist(), branch_mismatches=mismatches, probes=probes,
        original_panel_accepted=False, acceptance_qualified=False, cpu_neural_reference=False,
        explanation_scope="identity readouts preserve the original first-convolution weights; downstream probe weights differ")
    save(root / "diagnosis.json", record)
    print(json.dumps(record, indent=2), flush=True)


def run(args):
    packet = args.packet.resolve(); root = args.out.resolve()
    math_gate.load(packet, current=True)
    require((packet / "execution/failure.json").is_file() and not (packet / "execution/REPORT.json").exists(),
            "Need the unchanged failed full panel")
    root.mkdir(parents=True, exist_ok=False)
    inputs = [packet / "packet.json", packet / "execution/failure.json",
              packet / "execution/repeat-0/worker.log.json",
              packet / "libraries/native.so", packet / "libraries/reference.so"]
    source = packet / "execution/repeat-0/arrays" / CASE
    inputs += [source / name for name in ("input.f32", "parameters.f32", "upstream.f32",
               "eager-0-output.f32", "eager-0-gradient.f32", "expected-output.f64", "expected-gradient.f64")]
    require(type(args.timeout) is int and 1 <= args.timeout <= 60, "Short diagnostic requires a 1–60s deadline")
    nonce = secrets.token_hex(16)
    save(root / "request.json", dict(case=CASE, failed_packet=str(packet), supervisor_pid=os.getpid(), nonce=nonce,
        inputs_sha256={str(p): sha(p) for p in inputs}, tool_sha256=sha(Path(__file__)), timeout=args.timeout,
        acceptance_qualified=False, cpu_neural_reference=False))
    try:
        with math_gate.fixtures.LOCK.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
            hardware = math_gate.fixtures.idle_gpu(smi); (root / "gpu.txt").write_text(hardware)
            command = [sys.executable, str(Path(__file__).resolve()), "_worker", "--out", str(root)]
            math_gate.fixtures.claim.process(command, root, root / "worker.log", args.timeout,
                dict(os.environ, PUFFER_RELU_DIAGNOSTIC_TOKEN=nonce, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"))
            require(math_gate.fixtures.idle_gpu(smi) == hardware, "Hardware/idle state changed")
            load_request(root, current=True); save(root / "seal.json", math_gate.hashes(root))
    except BaseException as error:
        save(root / "failure.json", dict(error=str(error), acceptance_qualified=False)); raise
    return json.loads((root / "diagnosis.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="action", required=True)
    run_parser = sub.add_parser("run"); run_parser.add_argument("--packet", type=Path, required=True)
    run_parser.add_argument("--timeout", type=int, default=60)
    private = sub.add_parser("_worker")
    for command in (run_parser, private): command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "_worker": worker(args.out.resolve())
    else: print(json.dumps(run(args), indent=2))


if __name__ == "__main__": main()
