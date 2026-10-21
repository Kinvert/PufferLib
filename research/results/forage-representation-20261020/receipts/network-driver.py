"""Three declared CUDA math cases, not full qualification or a retry."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()/'research'))
import autoencoder_pretrain as a
p=Path('build/forage-representation-20261020-v5').resolve()
root=p/'network-smoke';a.require(not root.exists(),'Fresh allocation');root.mkdir()
driver=p/'stages/forage-network-smoke.driver.py';driver.write_bytes(Path(__file__).read_bytes())
rows=[]
with a.Reservation(p,'5060','forage-network-smoke',60) as gpu:
    for H,W in ((160,192),(161,193),(480,640)):
        folder=root/f'h{H}-w{W}-b1';log=root/f'h{H}-w{W}-b1.log'
        receipt=gpu.run([a.trainer_binary(p),'math',H,W,1,folder],log,30)
        value=a.read(folder/'result.json');a.require(value['status']=='passed','Math case failed')
        rows.append(dict(result=value,process=receipt,result_sha256=a.sha(folder/'result.json')))
    a.save(p/'stages/forage-network-smoke.json',dict(status='passed',hardware=gpu.identity,
        driver_sha256=a.sha(driver),cases=rows,full_qualification=False))
