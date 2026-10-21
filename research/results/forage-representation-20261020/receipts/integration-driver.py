"""One bounded, explicit local integration allocation; no full qualification waiver."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()/'research'))
import forage_representation as fr
ae=fr.ae
packet=Path('build/forage-representation-20261020-v5').resolve()
data_root=Path('build/forage-representation-collection-qa-20261020-a').resolve()
out=packet/'integration-smoke'
ae.require(not out.exists(),'Fresh integration allocation required');out.mkdir()
source=Path(__file__);driver=packet/'stages/forage-integration-smoke.driver.py';driver.parent.mkdir(exist_ok=True);driver.write_bytes(source.read_bytes())
data=ae.dataset(data_root);ae.write_index(data,data_root,out/'train.tsv')
cfg=ae.f.config('build/forage-representation-collection-qa.ini')
cfg['autoencoder'].update(batch='1',steps='2',checkpoint_interval='1',max_seconds='20')
cfg['probe'].update(batch='2',steps='4',max_seconds='30')
ae.f.write_config(out/'probe.ini',cfg)
progress=dict(driver_sha256=ae.sha(driver),training={},probes={})
try:
    with ae.Reservation(packet,'5060','forage-integration-smoke',120) as gpu:
        ae.require(ae.read(packet/'stages/forage-smoke.json')['status']=='passed','Require completed primitive smoke')
        cases=[('constant','constant'),('random','random')]
        for name,mode in [('plain',0),('object-balanced',1)]:
            cfg['autoencoder']['loss_mode']=str(mode);recipe=out/(name+'.ini');ae.f.write_config(recipe,cfg)
            folder=out/name;folder.mkdir();log=folder/'train.log'
            receipt=gpu.run([ae.trainer_binary(packet),'train',recipe,out/'train.tsv',folder,'None','v1'],log,30)
            rows=[fr.json.loads(line) for line in log.read_text().splitlines() if line.startswith('{')]
            ae.require(rows[-1]['status']=='complete' and rows[-1]['final_step']==2,'Short AE canary incomplete')
            state=ae.state(folder/'step-000000000002.state')[0]
            losses=[r['objective'] for r in rows if 'validation' in r]
            ae.require(losses[-1]<losses[0],'Canary objective did not decrease')
            progress['training'][name]=dict(state=state,process=receipt,initial_objective=losses[0],final_objective=losses[-1])
            cases.append((name,folder/'step-000000000002.state'))
        ae.require(ae.sha(out/'plain/step-000000000000.state')==ae.sha(out/'object-balanced/step-000000000000.state'),'Initial states differ')
        metadata,raw=ae.state(out/'plain/step-000000000000.state')
        initial_hash=fr.hashlib.sha256(raw[ae.STATE.size:ae.STATE.size+metadata['parameters']*4]).hexdigest()
        for name,checkpoint in cases:
            result=fr.probe(packet,data_root,out/'probe.ini',checkpoint,out/('probe-'+name),gpu)
            ae.require(result['status']=='complete','Short probe incomplete')
            if isinstance(checkpoint,str):ae.require(result['native']['parameter_sha256']==initial_hash,'Random initialization differs')
            progress['probes'][name]=dict(report_sha256=ae.sha(out/('probe-'+name)/'REPORT.json'),process_seconds=result['process_seconds'])
        result=fr.probe(packet,data_root,out/'probe.ini',out/'object-balanced/step-000000000002.state',out/'probe-repeat',gpu)
        for file in ('features.f32','head.f32','normalization.f32','labels.u8','predictions.csv'):
            ae.require(ae.sha(out/'probe-object-balanced'/file)==ae.sha(out/'probe-repeat'/file),'Probe repeat differs: '+file)
        progress.update(status='passed',hardware=gpu.identity,independent_training_seeds=1,no_quality_or_frontier_claim=True)
        ae.save(packet/'stages/forage-integration-smoke.json',progress)
except Exception as error:
    ae.save(out/'failure.json',dict(error=str(error),progress=progress));raise
