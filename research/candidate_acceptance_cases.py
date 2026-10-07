#!/usr/bin/env python3
"""Export audited development checkpoints to the existing exact-evaluator gate.

Offline artifact glue only: no native process, GPU discovery or model execution.
Final means last scheduled checkpoint, never best score. Connect4 keeps its
separate checker. This command does not prepare or launch GPU acceptance.
"""
import argparse
import json
from pathlib import Path

import candidate_panel as panel
import eval_acceptance as gate
from claim import require, save, sha

PROTOCOL = "audited-candidate-evaluator-cases-v1"


def export(plan_path, out, all_drawings=False):
    plan_path = plan_path.resolve(); out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    digest = sha(plan_path)
    save(out / "request.json", dict(protocol=PROTOCOL, plan=str(plan_path),
        plan_sha256=digest, all_drawings=all_drawings, checkpoint_rule="last-scheduled",
        gpu_queried=False, policy_executed=False), exclusive=True)
    try:
        audited = panel.audit(plan_path, out / "panel-audit")
        require(audited["status"] == "ok" and audited["audited_jobs"] == audited["planned_jobs"]
                and audited["audited_evaluations"] == audited["planned_evaluations"]
                and not audited["missing_evaluations"], "Need a complete successful audited panel")
        value = json.loads(plan_path.read_text()); root = plan_path.parent
        require(sha(plan_path) == digest and audited["plan_sha256"] == digest, "Plan changed during export")
        require(audited["planned_jobs"] == len(value["jobs"]), "Audited allocation differs from plan")
        builds = panel.registry_builds(value["native_registry"])
        cases, bindings, excluded = [], [], []
        for index, job in enumerate(value["jobs"]):
            task = job["environment"]
            if task == "connect4cnn":
                excluded.append(dict(job_index=index, reason="separate-connect4-checker")); continue
            require(task in gate.TASKS and job["policy_family"] in gate.FAMILIES,
                    "Unsupported task/family; do not silently exclude a candidate")
            steps = job["checkpoint_steps"]
            require(steps and steps == sorted(set(steps)), "Need ordered scheduled checkpoints")
            step = steps[-1]
            targets = job["targets"] if all_drawings else [t for t in job["targets"] if t["representation"] == 0]
            require(targets and (all_drawings or len(targets) == 1), "Missing/duplicate drawing-zero target")
            for target in targets:
                drawing = target["representation"]
                case = dict(name=f"job-{index:04d}-r{drawing}", task=task, family=job["policy_family"],
                    binary=job["command"][0], config=str(root / job["config"]),
                    checkpoint=str(root / job["id"] / "checkpoints" / task / job["candidate"] / f"{step:016d}.bin"),
                    suite=str(root / target["suite"]))
                proof = {}
                for key in ("binary", "config", "checkpoint", "suite"):
                    source = Path(case[key]); file_hash = sha(source)
                    if key == "binary":
                        binary = builds[task][job.get("binary_family", "default")]
                        require(str(source) == binary["path"] and file_hash == binary["sha256"],
                                "Selected native binary differs from audited build binding")
                    else:
                        require(audited["inputs_sha256"].get(str(source)) == file_hash,
                            "Selected input missing/changed since complete audit: " + key)
                    proof[key] = file_hash
                cases.append(case)
                bindings.append(dict(name=case["name"], job_index=index, candidate=job["candidate"],
                    training_seed=job["seed"], evaluation_representation=drawing,
                    checkpoint_step=step, inputs_sha256=proof))
        require(1 <= len(cases) <= 64, "Existing acceptance tool requires 1–64 cases; split a declared allocation explicitly")
        require(sha(plan_path) == digest, "Plan changed during export")
        save(out / "cases.json", cases, exclusive=True)
        record = dict(protocol=PROTOCOL, status="exported-not-prepared-or-executed", plan=str(plan_path),
            plan_sha256=digest, source_sha256=sha(Path(__file__)),
            audit_sha256=sha(out / "panel-audit/analysis.json"), cases_sha256=sha(out / "cases.json"),
            cases=len(cases), bindings=bindings, excluded=excluded, checkpoint_rule="last-scheduled",
            all_drawings=all_drawings, gpu_queried=False, policy_executed=False,
            quality_selection_allowed=False, publication_claim_qualified=False)
        save(out / "export.json", record, exclusive=True)
        return record
    except BaseException as error:
        save(out / "failure.json", dict(status="failed", error=f"{type(error).__name__}: {error}",
            gpu_queried=False, policy_executed=False), exclusive=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--all-drawings", action="store_true",
                        help="Every declared fixed drawing; default is drawing zero per non-Connect4 job")
    args = parser.parse_args()
    result = export(args.plan, args.out, args.all_drawings)
    print(json.dumps({k: result[k] for k in ("status", "cases", "gpu_queried", "policy_executed")}, indent=2))


if __name__ == "__main__": main()
