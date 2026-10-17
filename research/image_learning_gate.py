"""Fixed-control learning eligibility before realistic-image architecture search.

Scalar development evidence only. This gate does not establish a statistical
advantage or authorize confirmation-seed selection.
"""
import math

import image_curve_feedback as curves
from claim import require

MEAN_FLOOR = 0.1
SEED_FLOOR = 0.025
GAIN_FLOOR = 0.025
CONTROLS = ("quality-reference", "nature-cnn")
SPATIAL_PANEL = ("quality-reference", "spatial-control", "nature-cnn")


def qualify(plan, analysis, normalization):
    require(plan["protocol"] == curves.panels.IMAGE_CURVE_PROTOCOL
            and plan.get("seed_roles", {}).get("role") == "development"
            and len(plan["seeds"]) >= 3, "Learning calibration needs initial controls and fixed development replicas")
    names=tuple(c["name"] for c in plan["candidates"])
    require(names in (CONTROLS,SPATIAL_PANEL),
            "Use the predeclared quality/Nature calibration controls")
    nature=next(c for c in plan["candidates"] if c["name"]=="nature-cnn")
    require(nature.get("baseline") == "nature_cnn" and nature.get("kind") == "baseline"
            and nature.get("architecture") == {"encoder":2}, "Calibration Nature identity differs")
    require(set(normalization)==set(curves.panels.TASKS), "Missing game normalization")
    if names==SPATIAL_PANEL:
        # Targeted architecture evidence stays visible but cannot weaken the
        # original eligibility requirement or rescue an unlearned reference pair.
        require(plan["candidates"][1].get("architecture",{}).get("encoder")==4,
                "Spatial control must be a declared custom CNN")
        curves.panel_rows(plan,analysis,"spatial-control")
    controls=[]
    for name in CONTROLS:
        _,rows,schedules,_=curves.panel_rows(plan,analysis,name)
        for task in curves.panels.TASKS:
            offset,target=(normalization[task][key] for key in ("offset","target"))
            require(math.isfinite(offset) and math.isfinite(target) and target>offset, "Invalid learning normalization")
            def normalized(value): return max(0.,min(1.,(value-offset)/(target-offset)))
            finals=[]; initials=[]
            for seed in plan["seeds"]:
                def mean(step):
                    return math.fsum(normalized(rows[(task,seed,step,d)]["score_lower"])
                        for d in range(curves.panels.panel.ENVIRONMENTS[task]))/curves.panels.panel.ENVIRONMENTS[task]
                finals.append(mean(schedules[(task,seed)][-1])); initials.append(mean(0))
            gains=[final-initial for final,initial in zip(finals,initials)]
            average=math.fsum(finals)/len(finals)
            eligible=average>=MEAN_FLOOR and min(finals)>=SEED_FLOOR and min(gains)>=GAIN_FLOOR
            controls.append(dict(candidate=name,environment=task,mean=average,
                final_seed_means=finals,initial_seed_means=initials,paired_gains=gains,eligible=eligible))
    unresolved=[task for task in curves.panels.TASKS if not any(c["eligible"] for c in controls if c["environment"]==task)]
    return dict(schema="realistic-image-learning-eligibility-v1",
        status="eligible-development-search" if not unresolved else "blocked-learning-calibration",
        unresolved_games=unresolved,controls=controls,mean_floor=MEAN_FLOOR,seed_floor=SEED_FLOOR,gain_floor=GAIN_FLOOR,
        selection="fixed final checkpoints; all drawings/replicas retained; no best-checkpoint rescue",
        publication_claim_qualified=False)
