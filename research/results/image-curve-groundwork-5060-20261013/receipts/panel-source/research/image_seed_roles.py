"""Fixed experiment seed roles; no random search, policy or GPU operation."""
from candidate_panel import require

SCHEMA="realistic-image-seed-roles-v1"
DEFAULT={"schema":SCHEMA,"development_training":[7,42,69],"development_evaluation":[7007,7008,7009],
         "confirmation_training":[104729,130363,155921],"confirmation_evaluation":[420,1337,9001],
         "appearance_seed":17,"optimizer_seed":83,"seed_sweep_allowed":False}


def validate(value):
    require(set(value)==set(DEFAULT) and value["schema"]==SCHEMA and value["seed_sweep_allowed"] is False,
            "Incomplete/wrong fixed seed role contract")
    assigned=[]
    for role in ("development","confirmation"):
        training=value[role+"_training"]; evaluation=value[role+"_evaluation"]
        require(isinstance(training,list) and isinstance(evaluation,list) and len(training)==len(evaluation)>=2,
                "Use paired training/evaluation replicas in each role")
        for seeds in (training,evaluation):
            require(len(set(seeds))==len(seeds) and all(type(s) is int and 0<=s<2**32 for s in seeds),
                    "Invalid/duplicate fixed seed")
            assigned+=seeds
    require(len(set(assigned))==len(assigned),"Development/confirmation training/evaluation seeds overlap")
    require(all(type(value[k]) is int and 0<=value[k]<2**32 for k in ("appearance_seed","optimizer_seed")),
            "Invalid fixed appearance/optimizer seed")
    return value


def check_panel(value, role, training, evaluation, appearance):
    validate(value); require(role in ("development","confirmation"),"Unknown experiment seed role")
    require(training==value[role+"_training"] and evaluation==value[role+"_evaluation"]
            and appearance==value["appearance_seed"],"Panel changes frozen seed role assignments")


def check_search(value, role, training, evaluation, appearance, optimizer):
    require(role=="development","Confirmation results must never feed architecture search")
    check_panel(value,role,training,evaluation,appearance)
    require(optimizer==value["optimizer_seed"],"Native optimizer seed differs from fixed experiment")
