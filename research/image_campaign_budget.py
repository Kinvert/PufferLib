"""Measured complete-panel admission; scalar clocks/artifacts only, no GPU work.

An empirical reserve is an estimate, not a hard runtime guarantee. Unexpected
overruns remain failures; they never authorize partial PROTEIN observations.
"""
import math
import hashlib
import json
from pathlib import Path

from candidate_panel import require

SCHEMA = "realistic-image-campaign-budget-v1"
MAX_SECONDS = 12 * 60 * 60


def identities(recipe, reference):
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    envelope={s:dict(recipe[s]) for s in recipe.sections() if s=="policy" or s.startswith("sweep.")}
    if recipe.has_option("search","image_memory_policy"):
        import image_memory_budget as memory
        policy,limit=memory.budget(recipe)
        envelope["image_memory_admission"]=dict(policy=policy,storage_limit_mib=limit,
            headroom_mib=memory.HEADROOM_MIB if policy==memory.POLICY else None)
    return digest(reference["conditions"]), digest(envelope)


def freeze_pilot(plan_path, out):
    """Offline replay plus timed report measurement; never execute a policy."""
    import sys
    import cross_game_feedback as feedback
    from claim import process, save, sha
    root=plan_path.resolve().parent
    value=feedback.inspect(root)
    result_path=root/"execution/result.json"
    result=json.loads(result_path.read_text())
    require(value["protocol"]==feedback.CURVE_PROTOCOL and result["mode"]=="curve-pilot5090"
            and result["status"]=="ok","Use a completed full image-curve 5090 runtime pilot")
    out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,str(feedback.ROOT/"research/report_image_feedback.py"),
             "--plan",str(plan_path.resolve()),"--out",str((out/"review").resolve())]
    timing=process(command,feedback.ROOT,out/"report.txt",1800)
    audit_path=out/"review/review/analysis.json"; audit=json.loads(audit_path.read_text())
    recipe,_=feedback.recipe(root/"recipe.ini")
    reference=json.loads((root/"nature-reference.json").read_text())
    fixed,envelope=identities(recipe,reference)
    result_sha=sha(result_path)
    frozen=calibration(result,audit,result_sha,fixed,envelope,[timing])
    frozen["pilot_proof"]=dict(plan=str(plan_path.resolve()),plan_sha256=sha(plan_path),
        result=str(result_path),result_sha256=result_sha,audit=str(audit_path.resolve()),audit_sha256=sha(audit_path),
        report_process=str((out/"report.txt.json").resolve()),report_process_sha256=sha(out/"report.txt.json"))
    save(out/"runtime-calibration.json",frozen,exclusive=True)
    verify_pilot(frozen)
    return frozen


def verify_pilot(value):
    """Reconstruct reserves from retained native result/audit/process receipts."""
    import cross_game_feedback as feedback
    from claim import sha
    validate(value); proof=value.get("pilot_proof")
    require(isinstance(proof,dict),"Bare runtime estimates are not a measured pilot")
    for key in ("plan","result","audit","report_process"):
        require(sha(proof[key])==proof[key+"_sha256"],"Changed measured runtime pilot input: "+key)
    path=Path(proof["plan"]); root=path.parent; plan=feedback.inspect(root)
    result=json.loads(Path(proof["result"]).read_text()); audit=json.loads(Path(proof["audit"]).read_text())
    timing=json.loads(Path(proof["report_process"]).read_text())
    require(plan["protocol"]==feedback.CURVE_PROTOCOL and result["mode"]=="curve-pilot5090"
            and result["plan_sha256"]==proof["plan_sha256"]
            and Path(proof["result"])==root/"execution/result.json"
            and Path(proof["audit"]).name=="analysis.json", "Foreign runtime pilot receipts")
    command=timing["command"]
    require(len(command)==6 and Path(command[1])==feedback.ROOT/"research/report_image_feedback.py"
            and command[2:]==["--plan",str(path),"--out",str(Path(proof["audit"]).parent.parent)]
            and timing["cwd"]==str(feedback.ROOT),"Measured reporting command is not the retained full pilot audit")
    report_path=Path(proof["audit"]).parent.parent/"REPORT.json"
    report=json.loads(report_path.read_text())
    require(report["schema"]=="audited-image-feedback-report-v1" and report["status"]=="ok"
            and report["plan_sha256"]==proof["plan_sha256"] and report["result_sha256"]==proof["result_sha256"]
            and report["policy_executed"] is False and report["gpu_queried"] is False,
            "Runtime pilot must measure complete exports and interactive curves, not just its audit")
    require(all(sha(report_path.parent/name)==digest for name,digest in report["files_sha256"].items()),
            "Changed measured pilot report exports")
    recipe,_=feedback.recipe(root/"recipe.ini")
    fixed,envelope=identities(recipe,json.loads((root/"nature-reference.json").read_text()))
    rebuilt=calibration(result,audit,proof["result_sha256"],fixed,envelope,[timing])
    require({k:v for k,v in value.items() if k!="pilot_proof"}==rebuilt,"Runtime reserves differ from measured pilot clocks")


def seconds(receipt):
    require(receipt.get("status") == "ok" and receipt.get("returncode") == 0,
            "Failed/incomplete process cannot calibrate a runtime reserve")
    start, end = receipt["launch_monotonic_ns"], receipt["end_monotonic_ns"]
    require(type(start) is int and type(end) is int and 0 < start < end,
            "Need actual positive monotonic process clocks")
    return (end - start) / 1e9


def calibration(result, audit, result_sha256, conditions_sha256, envelope_sha256,
                report_receipts, safety_factor=2.0):
    """Use a complete, independently audited pilot on the execution hardware.

    Trial windows include proposal, admission, CUDA checks, preparation, every
    game/drawing/replica, and audit. Final optimizer replay and measured report
    commands have their own post-panel reserve. No SPS extrapolation is used.
    The caller must bind the conditions and calibrated search envelope before
    calling this scalar function; those identities are enforced at admission.
    """
    require(result.get("status") == "ok" and audit.get("status") == "ok"
            and result["trials"] and len(audit["trials"]) == len(result["trials"])
            and audit["result_sha256"] == result_sha256
            and result["final_observations_replayed"] == len(result["trials"]),
            "Calibration requires an audited complete feedback pilot")
    require(all(isinstance(h, str) and len(h) == 64 and set(h) <= set("0123456789abcdef")
                for h in (result_sha256, conditions_sha256, envelope_sha256)),
            "Missing calibration result/condition/search-envelope identities")
    require(type(safety_factor) in (int, float) and math.isfinite(safety_factor)
            and safety_factor >= 2, "Use at least twice the measured worst complete-panel time")
    trials = result["trials"]
    starts = [t["optimizer_process"]["launch_monotonic_ns"] for t in trials]
    final = result["final_optimizer_process"]
    ends = starts[1:] + [final["launch_monotonic_ns"]]
    measured = []
    for index, (trial, start, end) in enumerate(zip(trials, starts, ends)):
        seconds(trial["optimizer_process"])
        require(type(start) is int and type(end) is int and 0 < start < end,
                "Nonchronological complete-panel windows")
        require(trial["index"] == index and math.isfinite(trial["feedback"]["score"])
                and math.isfinite(trial["feedback"]["cost"]) and trial["feedback"]["cost"] > 0,
                "Incomplete pilot feedback")
        measured.append((end - start) / 1e9)
    require(report_receipts, "Reporting reserve needs measured report commands")
    final_seconds = seconds(final)
    report_seconds = math.fsum(seconds(r) for r in report_receipts)
    return dict(schema=SCHEMA, hardware=result["hardware"], pilot_result_sha256=result_sha256,
        conditions_sha256=conditions_sha256, search_envelope_sha256=envelope_sha256,
        safety_factor=safety_factor, measured_panel_seconds=measured,
        next_panel_seconds=safety_factor * max(measured),
        measured_final_replay_seconds=final_seconds, measured_report_seconds=report_seconds,
        post_panel_seconds=safety_factor * (final_seconds + report_seconds),
        empirical_estimate=True, hard_runtime_guarantee=False, publication_claim_qualified=False)


def validate(value):
    require(value["schema"] == SCHEMA and value["empirical_estimate"] is True
            and value["hard_runtime_guarantee"] is False
            and value["publication_claim_qualified"] is False,
            "Wrong measured campaign reserve contract")
    times = value["measured_panel_seconds"]
    factor = value["safety_factor"]
    require(times and all(type(t) in (int, float) and math.isfinite(t) and t > 0 for t in times)
            and type(factor) in (int, float) and math.isfinite(factor) and factor >= 2,
            "Invalid measured campaign times")
    require(value["next_panel_seconds"] == factor * max(times)
            and value["post_panel_seconds"] == factor *
                (value["measured_final_replay_seconds"] + value["measured_report_seconds"])
            and all(math.isfinite(value[k]) and value[k] > 0 for k in
                    ("measured_final_replay_seconds", "measured_report_seconds")),
            "Changed measured reserve arithmetic")


def admit(value, hardware, conditions_sha256, envelope_sha256, started_ns, now_ns,
          budget_seconds=MAX_SECONDS, observed_panel_seconds=(), observed_optimizer_seconds=(),inherited_observations=0):
    """Call only between panels, before any new native proposal or GPU work."""
    validate(value)
    require(hardware == value["hardware"] and conditions_sha256 == value["conditions_sha256"]
            and envelope_sha256 == value["search_envelope_sha256"],
            "Runtime calibration hardware/conditions/search envelope differ")
    require(type(started_ns) is int and type(now_ns) is int and 0 < started_ns <= now_ns
            and type(budget_seconds) in (int, float) and math.isfinite(budget_seconds)
            and 0 < budget_seconds <= MAX_SECONDS, "Invalid overnight budget/monotonic clock")
    require(isinstance(observed_panel_seconds,(list,tuple))
            and all(type(t) in (int, float) and math.isfinite(t) and t > 0
                for t in observed_panel_seconds), "Invalid completed-panel timing observation")
    # An unexpectedly slower completed trial can increase, never lower, the
    # frozen reserve. Fast trials do not justify shrinking the safety margin.
    estimate = max(value["next_panel_seconds"], value["safety_factor"] *
                   max(observed_panel_seconds, default=0))
    remaining = budget_seconds - (now_ns - started_ns) / 1e9
    import image_reporting_reserve as reporting
    require(type(inherited_observations) is int and 0<=inherited_observations<=512,"Invalid inherited reporting coverage")
    reporting_observations=inherited_observations+len(observed_panel_seconds)+1
    post_seconds=reporting.seconds(value,reporting_observations,observed_optimizer_seconds)
    needed = estimate + post_seconds
    return dict(status="admit-complete-panel" if remaining >= needed else "budget_stop",
        checked_monotonic_ns=now_ns,
        remaining_seconds=remaining, next_panel_seconds=estimate,
        post_panel_seconds=post_seconds, required_seconds=needed,
        reporting_observations=reporting_observations,
        native_observation_submitted=False, partial_feedback_allowed=False)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-plan",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(); result=freeze_pilot(args.pilot_plan,args.out.resolve())
    print(json.dumps({k:result[k] for k in ("schema","next_panel_seconds","post_panel_seconds")}))
