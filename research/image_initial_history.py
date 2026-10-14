"""Import every audited pilot observation without training those policies again.

Only a complete, non-inherited curve pilot is accepted. Source allocations and
episode receipts stay immutable; the new optimizer replays their exact ledger.
"""
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from claim import require, save, sha

SCHEMA="audited-image-pilot-history-v1"


def same_recipe(left,right):
    require(set(left.sections())==set(right.sections()),"Pilot/search recipe sections differ")
    for section in left.sections():
        a,b=dict(left[section]),dict(right[section])
        if section=="search":
            for key in ("max_runs","campaign_timeout"):a.pop(key);b.pop(key)
        require(a==b,"Pilot/search envelope, learners, budgets or seeds differ")


def same_reference(root,parent):
    require(json.loads((root/"nature-reference.json").read_text())
            ==json.loads((parent/"nature-reference.json").read_text()),
            "Pilot/search pixels, learners, sources, hardware or measured Nature anchors differ")


def ledger(result,dimensions):
    require(result["status"]=="ok" and result["mode"]=="curve-pilot5090"
            and result["trials"] and result["final_observations_replayed"]==len(result["trials"]),
            "Only a complete measured pilot can seed discovery")
    history=f"PUFFER_CROSS_GAME_V1 {dimensions}\n"
    for index,entry in enumerate(result["trials"]):
        require(entry["index"]==index and entry["proposal"]["observations_replayed"]==index
                and len(entry["proposal"]["normalized"])==dimensions,"Pilot observations are reordered or incomplete")
        require(all(math.isfinite(number) for number in entry["proposal"]["normalized"])
                and math.isfinite(entry["feedback"]["score"]) and math.isfinite(entry["feedback"]["cost"])
                and entry["feedback"]["cost"]>0,"Invalid pilot feedback or native coordinates")
        fields=["0",format(entry["feedback"]["score"],".17g"),format(entry["feedback"]["cost"],".17g")]
        fields.extend(format(number,".9g") for number in entry["proposal"]["normalized"])
        history+=" ".join(fields)+"\n"
    require(history==result["history"],"Pilot ledger differs from its audited observations")
    return history


def freeze(root,parent_plan,config,dimensions):
    import cross_game_feedback as feedback
    parent_plan=parent_plan.resolve();parent=parent_plan.parent
    packet=feedback.inspect(parent)
    require(packet["protocol"]==feedback.CURVE_PROTOCOL and "initial_history" not in packet,
            "Use a fresh complete curve pilot; nested history imports are not implemented")
    require(packet["dimensions"]==dimensions,"Pilot/search coordinates differ")
    same_recipe(config,feedback.recipe(parent/"recipe.ini")[0])
    same_reference(root,parent)
    report=feedback.audit(SimpleNamespace(plan=parent_plan,out=root/"initial-history-review"))
    result_path=parent/"execution/result.json";result=json.loads(result_path.read_text())
    history=ledger(result,dimensions)
    value=dict(schema=SCHEMA,parent_plan=str(parent_plan),parent_plan_sha256=sha(parent_plan),
        parent_result_sha256=sha(result_path),audit_sha256=sha(root/"initial-history-review/analysis.json"),
        observations=len(result["trials"]),history=history,hardware=result["hardware"],
        history_sha256=hashlib.sha256(history.encode()).hexdigest(),
        architectures=[entry["architecture"] for entry in result["trials"]],
        raw_audit_performed=report["native_feedback_receipts_audited"],policy_executed=False)
    save(root/"initial-history.json",value,exclusive=True)
    return value


def inspect(root,config):
    import cross_game_feedback as feedback
    value=json.loads((root/"initial-history.json").read_text());parent=Path(value["parent_plan"]).parent
    packet=feedback.inspect(parent)
    require(value["schema"]==SCHEMA and "initial_history" not in packet
            and packet["protocol"]==feedback.CURVE_PROTOCOL,"Wrong inherited pilot")
    same_recipe(config,feedback.recipe(parent/"recipe.ini")[0])
    same_reference(root,parent)
    require(sha(parent/"plan.json")==value["parent_plan_sha256"]
            and sha(parent/"execution/result.json")==value["parent_result_sha256"]
            and sha(root/"initial-history-review/analysis.json")==value["audit_sha256"],"Changed inherited pilot receipts")
    result=json.loads((parent/"execution/result.json").read_text());history=ledger(result,packet["dimensions"])
    require(value["history"]==history and value["history_sha256"]==hashlib.sha256(history.encode()).hexdigest()
            and value["observations"]==len(result["trials"]) and value["hardware"]==result["hardware"]
            and value["architectures"]==[entry["architecture"] for entry in result["trials"]]
            and value["raw_audit_performed"] is True,"Imported observations differ from their complete native pilot")
    return value


def audit(root,config,out):
    import cross_game_feedback as feedback
    value=inspect(root,config)
    report=feedback.audit(SimpleNamespace(plan=Path(value["parent_plan"]),out=out))
    require(report["result_sha256"]==value["parent_result_sha256"],"Pilot raw replay differs")
    return value
