"""Export existing two-update QA states; verify native Forage encoder/core import."""
from pathlib import Path
import sys,shutil
sys.path.insert(0,str(Path.cwd()/'research'))
import autoencoder_pretrain as a
p=Path('build/forage-representation-20261020-v5').resolve();root=p/'export-smoke'
a.require(not root.exists(),'Fresh export allocation');root.mkdir()
driver=p/'stages/forage-export-smoke.driver.py';driver.write_bytes(Path(__file__).read_bytes())
data_root=Path('build/forage-representation-collection-qa-20261020-a').resolve();data=a.dataset(data_root);exports=[]
for name in ('plain','object-balanced'):
    # New export fixtures bind the existing native receipts; completed canaries stay untouched.
    run=root/name;run.mkdir();source=p/'integration-smoke'/name
    shutil.copyfile(p/'integration-smoke'/(name+'.ini'),run/'recipe.ini')
    a.write_index(data,data_root,run/'index.tsv');a.save(run/'dataset-review.json',data)
    state=run/'step-000000000002.state';shutil.copyfile(source/state.name,state);metadata=a.state(state)[0]
    contract=a.training_contract(p,run/'recipe.ini',data)
    a.save(run/'request.json',dict(contract=contract,recipe_sha256=a.sha(run/'recipe.ini'),index_sha256=a.sha(run/'index.tsv')))
    receipt=a.read(source/'train.process.json')
    a.save(state.with_suffix('.json'),dict(state=metadata,contract=contract,process=receipt))
    a.save(run/'REPORT.json',dict(checkpoints=[dict(path=state.name,**metadata)],contract=contract))
    target=root/(name+'-encoder');manifest=a.export(p,run,'sprite_forage-480x640',target);exports.append((name,target,manifest))
rows=[]
with a.Reservation(p,'5060','forage-export-smoke',30) as gpu:
    for name,target,manifest in exports:
        log=root/(name+'-import.log');receipt=gpu.run([a.trainer_binary(p),'import-check',p/'configs/sprite_forage.ini',target],log,15)
        value=a.read(log);a.require(value==dict(status='passed',core_head_unchanged=True,rng_unchanged=True,encoder_exact=True),'Actual Forage import failed')
        rows.append(dict(name=name,manifest=manifest,process=receipt,result=value))
    a.save(p/'stages/forage-export-smoke.json',dict(status='passed',hardware=gpu.identity,driver_sha256=a.sha(driver),cases=rows,full_qualification=False))
