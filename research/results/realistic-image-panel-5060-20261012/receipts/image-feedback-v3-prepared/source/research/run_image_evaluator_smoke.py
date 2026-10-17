#!/usr/bin/env python3
"""Explicit short reserved GPU inference for one prepared realistic-image case."""
import argparse
import json
from pathlib import Path
import sys

import candidate_panel as panels
import eval_acceptance as acceptance
from claim import process, require, save, sha


def run(args):
    packet=args.packet.resolve(); value=acceptance.load_packet(packet,current_inputs=True)
    require(value["protocol"]==acceptance.LAYOUT_PROTOCOL and len(value["cases"])==1,
            "Need one host-layout-qualified image case")
    case=value["cases"][0]
    require(case["slots"]==4 and case["suite_episodes"]==5 and case["timeout"]<=30,
            "Image evaluator smoke leaves bounded allocation")
    registry=panels.policy_layout.inspect_registry(packet.parent/"policy-metadata",current=True)
    require(registry.get("schema")==panels.policy_layout.IMAGE_SCHEMA,"Need image-specific metadata")
    root=args.out.resolve(); root.mkdir(parents=True,exist_ok=False)
    try:
        hardware=panels.hardware(args.hardware)
        # The existing native adapter takes the common reservation and checks
        # for competing processes for every native inference call.
        timed=process([sys.executable,str(panels.ROOT/"research/eval_acceptance.py"),
                       "run","--packet",str(packet),"--out",str(root/"execution")],
                      panels.ROOT,root/"acceptance.txt",150)
        require(panels.hardware(args.hardware)==hardware,"Hardware changed or a competing compute job appeared")
        process([sys.executable,str(panels.ROOT/"research/eval_acceptance.py"),
                 "audit","--packet",str(packet),"--execution",str(root/"execution")],
                panels.ROOT,root/"offline-audit.txt",30)
        audit=json.loads((root/"offline-audit.txt").read_text())
        require(audit["status"]=="artifact-comparisons-passed" and len(audit["cases"])==1,
                "Incomplete evaluator comparisons")
        seconds=(timed["end_monotonic_ns"]-timed["launch_monotonic_ns"])/1e9
        result=dict(status="passed",hardware=hardware,image_contract=registry["image_contracts"][case["task"]],
                    packet_sha256=sha(packet),comparison=audit["cases"][0],process_seconds=seconds,
                    gpu_executed=True,policy_math_certified=False,learning_or_frontier_claim=False)
        save(root/"REPORT.json",result); return result
    except BaseException as error:
        save(root/"failure.json",dict(status="failed",error=str(error))); raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet",type=Path,required=True); parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--hardware",choices=("5060","5090"),required=True)
    print(json.dumps(run(parser.parse_args()),indent=2))
