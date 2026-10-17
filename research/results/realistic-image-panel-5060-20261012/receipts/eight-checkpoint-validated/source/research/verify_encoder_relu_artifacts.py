#!/usr/bin/env python3
"""Offline checks of retained failed-panel/probe artifacts; never loads a model.

This verifies hashes, arrays and the reported branch/patch pattern only. It
does not certify backward math or convert the failed panel into acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(packet, diagnostic):
    request = json.loads((diagnostic / "request.json").read_text())
    seal = json.loads((diagnostic / "seal.json").read_text())
    require(request["acceptance_qualified"] is False, "Wrong diagnostic scope")
    for name, digest in request["inputs_sha256"].items():
        require(sha(Path(name)) == digest, f"Changed diagnostic input: {name}")
    for name, digest in seal.items():
        require(sha(diagnostic / name) == digest, f"Changed diagnostic output: {name}")
    inventory = {str(p.relative_to(diagnostic)) for p in diagnostic.rglob("*") if p.is_file()}
    require(inventory == set(seal) | {"seal.json"}, "Changed diagnostic inventory")
    require(Path(request["failed_packet"]).resolve() == packet.resolve(), "Wrong failed packet")
    require(json.loads((packet / "execution/failure.json").read_text())["status"] == "failed",
            "Original failure missing")
    process = json.loads((packet / "execution/repeat-0/worker.log.json").read_text())
    require(process["returncode"] == 1 and process["status"] == "failed", "Wrong worker outcome")
    arrays = packet / "execution/repeat-0/arrays"
    source = arrays / request["case"]
    read = lambda path, dtype: np.fromfile(path, dtype=dtype)
    params = read(source / "parameters.f32", "<f4")
    expected = read(source / "expected-gradient.f64", "<f8")
    actual = read(source / "eager-0-gradient.f32", "<f4")
    require(params.size == actual.size == expected.size == 118880, "Wrong original layout")
    delta = actual - expected
    bad = np.flatnonzero(np.abs(delta) > 6e-5 + 8e-4 * np.abs(expected))
    diagnosis = json.loads((diagnostic / "diagnosis.json").read_text())
    require(bad.tolist() == diagnosis["original_gradient_violations"], "Wrong violations")
    require(diagnosis["original_panel_accepted"] is False and
            diagnosis["acceptance_qualified"] is False, "Diagnostic claimed acceptance")
    channel = diagnosis["channel"]
    obs = read(source / "input.f32", "<f4").reshape(2048, 36, 44)
    mismatches = []
    for probe in diagnosis["probes"]:
        start, count = probe["start"], probe["positions"]
        folder = diagnostic / "probes" / f"positions-{start}"
        identity = np.zeros_like(params)
        identity[:800] = params[:800]
        for j in range(count):
            identity[800 + j * 1584 + (start + j) * 16 + channel] = 1
            identity[102240 + j * 64 + j] = 1
        require(np.array_equal(identity, read(folder / "identity-readout-parameters.f32", "<f4")),
                "Wrong identity readout weights")
        out = read(folder / "native-output.f32", "<f4").reshape(2048, 256)
        ref = read(folder / "reference-output.f64", "<f8").reshape(2048, 256)
        require(np.isfinite(out).all() and np.isfinite(ref).all(), "Nonfinite probe")
        require(float(np.abs(out - ref).max()) == probe["output_max_absolute_difference"],
                "Wrong output difference")
        require(not read(folder / "zero-upstream.f32", "<f4").any() and
                not read(folder / "native-gradient.f32", "<f4").any(), "Wrong zero-upstream probe")
        indices = np.argwhere((out[:, :count] > 0) != (ref[:, :count] > 0))
        require(len(indices) == probe["branch_mismatches"], "Wrong branch count")
        for batch, col in indices:
            pos = start + int(col)
            y, x = divmod(pos, 11)
            patch = np.zeros((7, 7), dtype=np.float64)
            for ky in range(7):
                for kx in range(7):
                    iy, ix = y * 4 + ky - 1, x * 4 + kx - 1
                    if 0 <= iy < 36 and 0 <= ix < 44:
                        patch[ky, kx] = obs[batch, iy, ix]
            require(np.array_equal(patch.ravel(), read(folder / f"b{batch}-p{pos}-pixel-patch.f64", "<f8")),
                    "Wrong observed image patch")
            bias = float(delta[784 + channel])
            residual = float(np.abs(delta[channel * 49:(channel + 1) * 49] - bias * patch.ravel()).max())
            mismatches.append(dict(batch_index=int(batch), row=y, column=x, channel=channel,
                probe_position=pos, native_activation=float(out[batch, col]),
                reference_activation=float(ref[batch, col]), original_bias_gradient_delta=bias,
                patch_scaled_gradient_residual_max=residual))
    require(mismatches == diagnosis["branch_mismatches"], "Wrong branch/patch diagnosis")
    records = [json.loads(p.read_text()) for p in arrays.glob("*/case.json")]
    require(len(records) == 33 and all(r["status"] == "passed" for r in records), "Wrong partial records")
    native_calls = len(list(arrays.glob("*/*-output.f32")))
    probes = sum(len(r["finite_differences"]) for r in records)
    require(native_calls == 133 and probes == 16, "Wrong executed allocation")
    return dict(status="retained-artifact-consistency-passed", original_panel_status="failed",
        acceptance_qualified=False, neural_execution=False, successful_first_worker_records=len(records),
        started_cases=len(list(arrays.iterdir())), actual_native_calls=native_calls,
        completed_selected_finite_difference_probes=probes, gradient_violations=len(bad),
        sealed_diagnostic_files=len(seal), diagnostic_inputs=len(request["inputs_sha256"]),
        branch_mismatches=mismatches,
        worker_seconds=(process["end_monotonic_ns"] - process["launch_monotonic_ns"]) / 1e9)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.packet, args.diagnostic), indent=2))
