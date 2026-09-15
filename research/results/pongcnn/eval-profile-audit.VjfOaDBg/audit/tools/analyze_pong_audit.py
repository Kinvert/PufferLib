"""Read-only analysis of finished bounded Pong audit and Nsight SQLite traces."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ocean/pongcnn'))
from confirm_report import read_ini, normalized, save, sha


def category(name):
    # cuBLAS kernels are shared by convolutions, projections, core and Muon.
    # Do not guess their caller from a generic GEMM kernel name.
    if 'bias_grad' in name or 'bias_backward' in name: return 'bias gradient'
    if any(x in name for x in ('unpatch','col2im')): return 'convolution input gradient gather'
    if any(x in name for x in ('patches','im2col')): return 'convolution patch materialization (forward and backward)'
    if 'pool' in name: return 'pooling'
    if any(x in name for x in ('imp_add','relu','imp_bias','readout')): return 'activation/residual/readout'
    if any(x in name for x in ('muon_','adam')): return 'optimizer named kernels (excludes shared GEMM)'
    if any(x in name.lower() for x in ('gemm','sgemm','gemv','splitk','cutlass')): return 'shared matrix multiplication (caller unresolved)'
    if any(x in name for x in ('lstm','scan','rnn')): return 'recurrent core named kernels'
    return 'other'


def profile(path):
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    names = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    schema = {n: [r[1] for r in db.execute(f'PRAGMA table_info("{n}")')] for n in names}
    result = dict(schema=schema)
    if 'CUPTI_ACTIVITY_KIND_KERNEL' not in schema:
        result['status'] = 'no CUDA kernel trace'
        return result
    strings = dict(db.execute('SELECT id,value FROM StringIds'))
    table = 'CUPTI_ACTIVITY_KIND_KERNEL'
    key = 'demangledName' if 'demangledName' in schema[table] else 'shortName'
    kernels = [dict(name=strings.get(n,str(n)), count=count, total_s=t/1e9, mean_us=t/count/1e3)
               for n,count,t in db.execute(f'SELECT {key},count(*),sum(end-start) FROM {table} GROUP BY {key} ORDER BY sum(end-start) DESC')]
    result.update(status='CUDA kernel trace available', kernels=kernels,
                  kernel_count=sum(r['count'] for r in kernels), kernel_time_s=sum(r['total_s'] for r in kernels))
    cats = collections.defaultdict(lambda: dict(count=0,total_s=0))
    for r in kernels:
        c = cats[category(r['name'])]
        c['count'] += r['count']; c['total_s'] += r['total_s']
    result['categories'] = cats
    apis = []
    if 'CUPTI_ACTIVITY_KIND_RUNTIME' in schema:
        apis = [dict(name=strings.get(n,str(n)),count=count,total_s=t/1e9)
                for n,count,t in db.execute('SELECT nameId,count(*),sum(end-start) FROM CUPTI_ACTIVITY_KIND_RUNTIME GROUP BY nameId ORDER BY sum(end-start) DESC')]
        result['cuda_apis'] = apis
        capture_ids = [i for i,n in strings.items() if 'cudaGraphInstantiate' in n]
        if capture_ids:
            last = db.execute('SELECT max(end) FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE nameId IN ('+','.join('?' for _ in capture_ids)+')',capture_ids).fetchone()[0]
            if last is not None:
                count,total = db.execute(f'SELECT count(*),sum(end-start) FROM {table} WHERE start>?',(last,)).fetchone()
                result['after_last_graph_instantiate'] = dict(cutoff_ns=last,kernel_count=count,kernel_time_s=(total or 0)/1e9)
        result['graph_capture_api_seconds'] = sum(r['total_s'] for r in apis if any(x in r['name'] for x in ('cudaStreamBeginCapture','cudaStreamEndCapture','cudaGraphInstantiate','cudaGraphDestroy')))
    if 'CUPTI_ACTIVITY_KIND_MEMCPY' in schema:
        result['copies'] = [dict(kind=kind,count=count,time_s=t/1e9,bytes=b) for kind,count,t,b in db.execute('SELECT copyKind,count(*),sum(end-start),sum(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind')]
    if 'NVTX_EVENTS' in schema and 'text' in schema['NVTX_EVENTS']:
        cols = schema['NVTX_EVENTS']
        query = 'SELECT start,end,text'+(',textId' if 'textId' in cols else ',NULL')+' FROM NVTX_EVENTS WHERE end IS NOT NULL AND end>start ORDER BY start'
        ranges = collections.defaultdict(list)
        for start,end,text,tid in db.execute(query):
            ranges[text or strings.get(tid,str(tid))].append((end-start)/1e9)
        result['nvtx_host_ranges'] = {k:dict(count=len(v),first_s=v[0],later_mean_s=sum(v[1:])/len(v[1:]) if len(v)>1 else None,total_s=sum(v)) for k,v in ranges.items()}
    db.close()
    return result


def main(out):
    plan = json.loads((out/'plan.json').read_text())
    original = Path(plan['original'])
    rows = json.loads((out/'measurements.json').read_text())
    for r in rows:
        directory = out/r['name']
        log = (directory/'output.log').read_text(errors='replace') if (directory/'output.log').exists() else ''
        steps = re.findall(r'│ Steps\s+([\d.]+[kKMBT]?)',log)
        def number(s): return float(s[:-1])*dict(k=1e3,K=1e3,M=1e6,B=1e9,T=1e12)[s[-1]] if s[-1].isalpha() else float(s)
        r['dashboard_steps_approx'] = [number(s) for s in steps]
        activity = (directory/'activity.log').read_text() if (directory/'activity.log').exists() else ''
        gpu = re.findall(r'^([\d.]+) %, ([\d.]+) %, ([\d.]+) MiB, ([\d.]+) W$',activity,re.M)
        r['gpu_samples'] = [dict(util_percent=float(a),memory_percent=float(b),memory_mib=float(c),power_w=float(d)) for a,b,c,d in gpu]
        cpu = []
        for line in activity.splitlines():
            cells = line.split()
            if len(cells)==7 and cells[-1]==(r['model']+'.bin')[:15]:
                cpu.append(dict(pid=int(cells[0]),pcpu_lifetime=float(cells[2]),cpu_time=cells[3],rss_kib=int(cells[4]),state=cells[5]))
        r['cpu_samples'] = cpu
        if r['group']=='performance':
            metric = directory/'metrics/pongcnn/trial.ini'
            if metric.exists():
                actual = read_ini(metric)
                expected = read_ini(original/r['id']/'metrics/pongcnn/trial.ini')
                changes = dict(seed='52001',checkpoint_interval='64',checkpoint_dir=str(directory/'checkpoints'),log_dir=str(directory/'metrics'))
                if r['version']=='instrumented': changes['profile']='1'
                for k,v in changes.items(): expected['base'][k]=v
                expected['train']['total_timesteps']='131072'
                differences = {s+'.'+k:[v,actual.get(s,k,fallback=None)] for s in expected.sections() if s!='metrics' for k,v in expected[s].items() if normalized(v)!=normalized(actual.get(s,k,fallback='MISSING'))}
                r['unexpected_resolved_config_differences'] = differences
                r['all_metrics_finite'] = all(abs(v)<float('inf') for values in r.get('metrics',{}).values() for v in values)
                r['coarse_last_interval_s'] = {k:v[-1] for k,v in r['metrics'].items() if k.startswith('perf/')}
                epochs = [int(e) for e in re.findall(r'│ Epoch\s+(\d+)',log)]
                if len(epochs)>1 and epochs[-1]>epochs[-2]:
                    r['coarse_last_interval_rollouts'] = epochs[-1]-epochs[-2]
                    r['coarse_ms_per_rollout'] = {k:v*1000/r['coarse_last_interval_rollouts'] for k,v in r['coarse_last_interval_s'].items()}
            db = directory/'profile.sqlite'
            r['profiler'] = profile(db) if db.exists() else dict(status='no exported SQLite trace')
        else:
            p = r['progress']
            r['progress_sps'] = (p[-1]['steps']-p[0]['steps'])/(p[-1]['elapsed']-p[0]['elapsed']) if len(p)>1 else None
            r['matches_between_progress_samples'] = p[-1]['games']-p[0]['games'] if len(p)>1 else None
            if r['success']: r['classification']='completed within diagnostic cap'
            elif r['exit_code'] not in ('124','137','incomplete'): r['classification']='execution error'
            elif len(p)>1 and p[-1]['steps']>p[0]['steps']:
                r['classification']='advancing simulation; zero newly completed matches between progress samples' if p[-1]['games']==p[0]['games'] else 'advancing simulation; target not reached; some matches completed'
            elif steps and number(steps[-1])>number(steps[0]): r['classification']='advancing simulation (rounded dashboard); completed-match count unavailable'
            else: r['classification']='unresolved: insufficient progress evidence'
    saved = json.loads((out/'original-identities.json').read_text())
    original_changed = [name for name,h in saved.items() if sha(original/name)!=h]
    assert not original_changed, original_changed
    pairs = {}
    for label in dict.fromkeys(r['label'] for r in rows if r['group']=='performance'):
        pair = [r for r in rows if r['group']=='performance' and r['label']==label]
        pairs[label] = dict(both_valid=all(r['success'] for r in pair),
                           checkpoint_identical=bool(pair[0]['checkpoints']) and pair[0]['checkpoints']==pair[1]['checkpoints'])
    save(out/'analysis.json',dict(attempts=rows, original_receipts_verified=len(saved), original_changed=original_changed,performance_pairs=pairs))
    with (out/'timings.csv').open('w') as f:
        keys=['name','group','version','model','recipe','exit_code','wall_s','success']
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    lines = ['# Native timing and profiler coverage', '',
             'Native values below are the last logging interval divided by its recorded rollout count. One rollout is 2,048 decisions. These are coarse local intervals, not sums of the downsampled history. Training model includes forward/backward, loss, updates and optimizer work.', '',
             '| Cell | Rollouts in interval | Rollout ms | Inference ms | Environment ms | Copy ms | Train model ms | Train misc ms |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['group']=='performance' and r['version']=='unprofiled':
            v=r['coarse_ms_per_rollout']
            lines.append('| '+r['label']+' | '+str(r['coarse_last_interval_rollouts'])+' | '+ ' | '.join(f'{v[k]:.4f}' for k in ('perf/rollout','perf/eval_model','perf/eval_env','perf/eval_copy','perf/train_model','perf/train_misc'))+' |')
    lines += ['', '## CUDA trace', '',
              'Times sum traced kernel durations across streams; they are not elapsed wall time or GPU utilization. Matrix-multiply kernels are shared across convolution forward/input-gradient/weight-gradient, projection/core and optimizer, so those callers remain unresolved. Named kernels and all CUDA API summaries are preserved in analysis.json.', '',
              '| Cell | Kernel executions | Kernel seconds | Shared GEMM % | Patch % | Bias gradient % | Input gather % | Pool % | Capture API ms | After-capture kernel seconds |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['group']=='performance' and r['version']=='instrumented':
            p=r['profiler']; cats=p['categories']; total=p['kernel_time_s']
            fractions=[100*cats.get(k,{}).get('total_s',0)/total for k in ('shared matrix multiplication (caller unresolved)','convolution patch materialization (forward and backward)','bias gradient','convolution input gradient gather','pooling')]
            lines.append(f"| {r['label']} | {p['kernel_count']} | {total:.4f} | "+' | '.join(f'{v:.2f}' for v in fractions)+f" | {p['graph_capture_api_seconds']*1000:.3f} | {p['after_last_graph_instantiate']['kernel_time_s']:.4f} |")
    lines += ['', 'Capture API time sums cudaStreamBeginCapture/EndCapture, cudaGraphInstantiate and cudaGraphDestroy CPU calls; it does not measure the complete first-use capture region. After-capture excludes kernels starting before the final graph-instantiation return. First and later NVTX host-range durations are retained separately in analysis.json and do not equal asynchronous GPU execution time.', '',
              'Nsight Systems 2025.5.2 captured CUDA node activity and NVTX under the existing base.profile CUDA profiler hooks. CPU sampling and context-switch tracing were disabled. No Nsight Compute hardware-counter replay was run. No occupancy, cache-bandwidth, per-layer GEMM attribution or isolated resource-query/logging CPU time is claimed. Five-second GPU/CPU samples may miss brief jobs and memory peaks. Instrumented full process time includes trace export overhead. Unprofiled controls remain separate.', '']
    (out/'PROFILE.md').write_text('\n'.join(lines))
    print(json.dumps(dict(original_receipts_verified=len(saved),performance_pairs=pairs)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('root',type=Path)
    main(parser.parse_args().root.resolve())
