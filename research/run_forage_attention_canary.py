"""Four-mode native canary; Python is process/artifact glue, never a learner."""
import argparse
import array
import configparser
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    config = configparser.ConfigParser(interpolation=None, strict=False)
    assert config.read(path)
    return config

def main(build, out):
    build, out = build.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    save = lambda name,value:(out/name).write_text(json.dumps(value,indent=2)+'\n')
    math_log = (build/'math.log').read_text()
    assert math_log.count('Attention CUDA PASS') == 8
    assert sha(ROOT/'ocean/sprite_forage/sprite_forage.h') == sha(
        build/'source/ocean/sprite_forage/sprite_forage.h')
    assert sha(ROOT/'ocean/sprite_forage/attention.h') == sha(
        build/'source/ocean/sprite_forage/attention.h')
    assert sha(ROOT/'research/forage_attention_encoder.cu') == sha(
        build/'source/research/forage_attention_encoder.cu')
    assert sha(ROOT/'research/forage_attention_dispatch.cu') == sha(
        build/'source/research/sprite_encoders.cu')
    binary = build/'puffer'
    binary_sha256 = sha(binary)
    recipe = read(ROOT/'research/recipes/sprite_forage_spatial_cnn.ini')
    jobs = []
    for index,(partial,pixels,parameters) in enumerate(((0,0,106368),(0,1,1332512),
            (1,0,91008),(1,1,137896))):
        name = ('quiet-otter-1','mossy-fern-2','sunny-gecko-3','happy-heron-4')[index]
        folder = out/name
        (folder/'runtime/config').mkdir(parents=True)
        config = read(ROOT/'config/sprite_forage.ini')
        config['policy'].update(dict(recipe['policy']))
        config['env'].update(partial_observation=str(partial),observation_mode=str(pixels))
        config['vec'].update(total_agents='8',num_buffers='1',num_threads='2')
        config['train'].update(total_timesteps='8192',horizon='8',minibatch_size='8')
        config['base'].update(seed='7',run_id=name,load_model_path='None',eval_episodes='0',
            checkpoint_interval='0',checkpoint_dir=str(folder/'checkpoints'),log_dir=str(folder/'logs'))
        with (folder/'runtime/config/sprite_forage.ini').open('x') as file:
            config.write(file)
        shutil.copyfile(ROOT/'config/default.ini',folder/'runtime/config/default.ini')
        jobs.append(dict(name=name,partial=partial,pixels=pixels,parameters=parameters,
            config_sha256=sha(folder/'runtime/config/sprite_forage.ini')))
    save('plan.json',dict(jobs=jobs,steps=8192,fixed_seed=7,seed_swept=False,
        binary_sha256=sha(binary),encoder_math_log_sha256=sha(build/'math.log'),
        cases='four modes, one fresh training each, six8-episode evaluations',
        scope='canary; no learning/frontier/generalization/full-learner-math claim'))
    began = time.monotonic()
    save('started.json',dict(monotonic_started=began))
    training, evaluations = [], []
    smi = shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'

    def run(label, command, cwd):
        assert sha(binary) == binary_sha256
        queries = []
        idle_deadline = time.monotonic()+5
        while True:
            busy = subprocess.check_output([smi,'--query-compute-apps=pid','--format=csv,noheader'],text=True)
            queries.append(dict(compute_pids=busy,monotonic_seconds=time.monotonic()))
            save(label+'.gpu-query.json',queries)
            if not busy.strip():
                break
            assert time.monotonic()<idle_deadline,'GPU occupied; stop without retrying policy'
            time.sleep(.2)
        save(label+'.command.json',command)
        begin = time.monotonic()
        remaining = 180-(begin-began)
        assert remaining>0
        with (out/(label+'.log')).open('x') as log:
            try:
                result = subprocess.run(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,
                    timeout=min(30,remaining),env=dict(os.environ,PUFFER_CHECKPOINT_RECEIPTS='1'))
                code,timed_out=result.returncode,False
            except subprocess.TimeoutExpired:
                code,timed_out=None,True
        end = time.monotonic()
        receipt = dict(returncode=code,timed_out=timed_out,monotonic_started=begin,
            monotonic_finished=end,process_seconds=end-begin,log_sha256=sha(out/(label+'.log')))
        save(label+'.process.json',receipt)
        assert code==0 and not timed_out,(label,receipt)
        return receipt

    def evaluate(job, weight, label, graphs=1):
        receipt = run(label,[str(binary),'eval','--headless',f'--base.load_model_path={weight}',
            '--base.eval_agents=1','--base.eval_episodes=8','--base.seed=7','--env.seed=420',
            '--vec.num_threads=1','--train.horizon=1','--train.minibatch_size=1',
            f'--base.cudagraphs={graphs}'],out/job['name']/'runtime')
        text = (out/(label+'.log')).read_text()
        match = re.search(r'CUDA_EVAL env=sprite_forage score=([\d.e+-]+) perf=([\d.e+-]+) games=(\d+) params=(\d+)',text)
        assert match and int(match[3])==8 and int(match[4])==job['parameters']
        row = dict(job=job['name'],label=label,graphs=graphs,perf=float(match[2]),**receipt)
        assert 0<=row['perf']<=1
        evaluations.append(row)
        save('evaluations.json',evaluations)
        return row

    with (ROOT/'build/connect4cnn/hardware-benchmark.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        save('hardware.json',dict(query=subprocess.check_output(
            [smi,'--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True)))
        for job in jobs:
            folder=out/job['name']
            assert sha(folder/'runtime/config/sprite_forage.ini')==job['config_sha256']
            receipt=run(job['name'],[str(binary),'train'],folder/'runtime')
            checkpoint=folder/'checkpoints/sprite_forage'/job['name']/'0000000000008192.bin'
            assert checkpoint.stat().st_size==job['parameters']*4
            assert all(math.isfinite(v) for v in array.array('f',checkpoint.read_bytes()))
            config=read(folder/'runtime/config/sprite_forage.ini')
            resolved=read(checkpoint.parent/'resolved.ini')
            for section in ('env','vec','policy','train'):
                for key,value in config[section].items():
                    assert resolved[section][key]==value,(section,key)
            metrics=read(folder/'logs/sprite_forage'/(job['name']+'.ini'))['metrics']
            last=lambda key:float(metrics[key].split(',')[-1])
            assert last('agent_steps')==8192
            row=dict(**job,**receipt,process_sps=8192/receipt['process_seconds'],
                native_last_bin_sps=last('sps'),native_uptime=last('uptime'),
                final_training_perf=last('env/perf'),checkpoint_sha256=sha(checkpoint))
            training.append(row)
            save('training.json',training)
            reference=evaluate(job,checkpoint,job['name']+'-eval')
            if job['partial'] and job['pixels']:
                for suffix,graphs in (('repeat',1),('eager',-1)):
                    other=evaluate(job,checkpoint,job['name']+'-'+suffix,graphs)
                    assert other['perf']==reference['perf']
    save('results.json',dict(training=training,evaluations=evaluations,
        campaign_seconds=time.monotonic()-began,learning_or_frontier_claim=False,
        evaluation_check='aggregate repeat/eager equality only, not episode/action traces'))
    (out/'COMPLETE').write_text('Four native mode trainings and all six evaluations complete.\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build',type=Path)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    main(args.build,args.out)
