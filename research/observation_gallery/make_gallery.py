#!/usr/bin/env python3
"""Small temporary gallery exported from the native observation buffer."""
import gzip
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import time
import zlib

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(__file__).resolve().parent
GAMES={
    'connect4cnn':dict(title='Connect4',hw=(576,672),steps=(3,5,7),names=[
        'Large squares','Medium squares / wide gaps','Small squares','Large circles',
        'Medium circles','X and O','80% squares','50% squares','25% squares','Tall rectangles'],
        legacy_names=['6×6 solid cells','4×4 squares on 6×6 cells','2×2 squares on 6×6 cells',
            '6×6 circles','4×4 circles on 6×6 cells','X and O','4×4 cells','2×2 cells','1×1 cells','2×4 cells']),
    'pongcnn':dict(title='Pong',hw=(640,620),steps=(20,60,100),names=[
        'Original colors','Swap paddle colors','Horizontal flip','Vertical court flip',
        'Invert court','Checker texture','Scanline texture'],legacy_names=[
        'Original intensities','Swap paddle intensities','Horizontal flip','Vertical court flip',
        'Invert court','Pixel checker texture','Edge-outline texture']),
    'flappycnn':dict(title='Flappy',hw=(640,420),steps=(20,50,90),names=[
        'Square bird / solid pipes','Horizontal flip','Vertical flip','Inverted colors',
        'Round bird','Outline pipes','Round bird / outline pipes']),
    'breakoutcnn':dict(title='Breakout',hw=(330,576),steps=(60,160,300),names=[
        'Original colors','Horizontal flip','Vertical flip','Inverted colors','Swap brick / paddle colors']),
    'snakecnn':dict(title='Snake',hw=(616,616),steps=(20,50,90),names=[
        'Solid cells','Cross cells','Inset cells / gaps','Inverted colors','Swap food / body colors','Horizontal flip']),
    'mazecnn':dict(title='Maze',hw=(616,616),steps=(3,6,9),names=[
        'Solid cells','Cross cells','Inset cells / gaps','Inverted colors','Swap wall / goal colors','Horizontal flip']),
    'foragecnn':dict(title='Sprite Forage',hw=(480,640),steps=(8,24,56),modes=['rgb'],names=[
        'Original pixel-art meadow','Twilight palette','Mirrored sprites','Dense grass texture','Kenney Tiny Dungeon sprites']),
}

def digest(data):return hashlib.sha256(data).hexdigest()

def png(width,height,rgb):
    def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    rows=b''.join(b'\0'+rgb[y*width*3:(y+1)*width*3] for y in range(height))
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(rows,9))+chunk(b'IEND',b'')

def run(command,**kwargs):
    result=subprocess.run(command,cwd=ROOT,capture_output=True,timeout=180,**kwargs)
    if result.returncode:raise RuntimeError(str(command)+'\n'+result.stderr.decode(errors='replace'))
    return result

def decode(data):
    frames=[];offset=0
    while offset<len(data):
        end=data.index(b'\n',offset);value=json.loads(data[offset:end]);offset=end+1
        size=int(np.prod(value['shape_hwc']))*4
        raw=data[offset:offset+size];assert len(raw)==size
        frames.append((value,raw));offset+=size
    assert len(frames)==3
    return frames

def main(out,selected):
    assert sys.byteorder=='little'
    OUT=out.resolve();OUT.mkdir(parents=True,exist_ok=False)
    games={game:GAMES[game] for game in selected}
    (OUT/'images').mkdir(exist_ok=True);(OUT/'memory').mkdir(exist_ok=True);(OUT/'native').mkdir(exist_ok=True)
    manifest=dict(schema='native-observation-gallery-v2',source_revision=run(['git','rev-parse','HEAD']).stdout.decode().strip(),
        source_worktree=str(ROOT),seed=7,frames=[],games=games,modes={
            'rgb':dict(title='Realistic RGB — new pipeline',contract='human-geometry-v1',channels=3),
            'legacy':dict(title='Historical 36×44 grayscale',contract='legacy',channels=1)},
        policy_executed=False,gpu_used=False,state_injected=False,builds=[],repeat_processes=[],source_sha256={})
    source=(SOURCE/'capture.c').read_bytes();manifest['capture_source_sha256']=digest(source)
    (OUT/'capture.c').write_bytes(source)
    (OUT/'worktree.txt').write_bytes(run(['git','status','--short']).stdout)
    start=time.monotonic();paired={}
    for game,settings in games.items():
        for mode in settings.get('modes',('rgb','legacy')):
            target=OUT/'native'/f'{game}-{mode}'
            height,width=settings['hw'];steps=settings['steps']
            flags=['clang','-std=c11','-D_POSIX_C_SOURCE=200809L','-D_DEFAULT_SOURCE','-O1','-g0',
                '-ffunction-sections','-fdata-sections','-I.','-Isrc','-Iraylib-5.5_linux_amd64/include',
                f'-DENV_HEADER="ocean/{game}/{game}.h"',f'-DPUFFER_{game.upper()}',
                f'-DGALLERY_REPS={len(settings["names"])}',*[f'-DGALLERY_T{i+1}={n}' for i,n in enumerate(steps)]]
            if mode=='rgb':flags += ['-DCNN_IMAGE_V1',f'-DCNN_IMAGE_HEIGHT={height}',f'-DCNN_IMAGE_WIDTH={width}','-DCNN_IMAGE_CHANNELS=3']
            flags += ['-MMD','-MF',str(target)+'.d',str(OUT/'capture.c'),
                'raylib-5.5_linux_amd64/lib/libraylib.a','-Wl,--gc-sections','-lGL','-lpthread','-ldl','-lrt','-lm','-o',str(target)]
            compiled=run(flags)
            deps=Path(str(target)+'.d').read_text().replace('\\\n',' ').split()[1:]
            for filename in [*deps,'config/default.ini',f'config/{game}.ini']:
                path=Path(filename);path=path if path.is_absolute() else ROOT/path
                manifest['source_sha256'][str(path)]=digest(path.read_bytes())
            (target.with_suffix('.build.txt')).write_bytes(compiled.stderr)
            manifest['builds'].append(dict(game=game,mode=mode,command=flags,binary_sha256=digest(target.read_bytes())))
            for rep,name in enumerate(settings['names']):
                command=[str(target),f'config/{game}.ini',str(rep)]
                result=run(command)
                if rep==0:
                    assert run(command).stdout==result.stdout,'Fresh-process repeat mismatch'
                    manifest['repeat_processes'].append(dict(game=game,mode=mode,exact=True))
                for meta,raw in decode(result.stdout):
                    h,w,c=meta['shape_hwc'];values=np.frombuffer(raw,dtype='<f4').reshape(h,w,c)
                    assert np.isfinite(values).all() and values.min()>=0 and values.max()<=1
                    assert meta['episode_decision']>0 and meta['repeat_exact']
                    # Same world/RNG/progress across all drawings AND RGB/legacy.
                    identity={key:meta[key] for key in ('decision','episode','episode_decision','tick','rng','state')}
                    key=(game,meta['moment']);paired.setdefault(key,identity)
                    assert paired[key]==identity,(game,mode,rep,'State is not paired')
                    if game=='connect4cnn':assert meta['state']['pieces']>=6
                    color=np.repeat(values,3,axis=2) if c==1 else values
                    pixels=np.rint(color*255).astype(np.uint8)
                    stem=f'{game}-{mode}-v{rep:02d}-m{meta["moment"]}'
                    image=OUT/'images'/(stem+'.png');image.write_bytes(png(w,h,pixels.tobytes()))
                    memory=OUT/'memory'/(stem+'.f32.gz');memory.write_bytes(gzip.compress(raw,compresslevel=9,mtime=0))
                    # The exact source float of each displayed color is retained
                    # for cursor inspection. Refuse an ambiguous 8-bit mapping.
                    palette={}
                    flat=pixels.reshape(-1,3).astype(np.uint32)
                    packed=(flat[:,0]<<16)|(flat[:,1]<<8)|flat[:,2]
                    unique,first,inverse=np.unique(packed,return_index=True,return_inverse=True)
                    actual=color.reshape(-1,3)[first]
                    assert np.array_equal(actual[inverse],color.reshape(-1,3)),'PNG color loses ambiguous floats'
                    for at,triplet in zip(first,actual):
                        palette[','.join(map(str,flat[at]))]=triplet.tolist()
                    display_name=settings.get('legacy_names',settings['names'])[rep] if mode=='legacy' else name
                    manifest['frames'].append(dict(game=game,mode=mode,name=display_name,**meta,
                        image=str(image.relative_to(OUT)),raw_file=str(memory.relative_to(OUT)),
                        raw_sha256=digest(raw),image_sha256=digest(image.read_bytes()),
                        bytes=len(raw),png_bytes=image.stat().st_size,compressed_bytes=memory.stat().st_size,
                        palette=palette,range=[float(values.min()),float(values.max())]))
            print(f'{settings["title"]} {mode}: {len(settings["names"])} variants × 3 snapshots',flush=True)
            target.unlink() # Small temporary gallery does not retain executable copies.
    expected=3*sum(len(v['names'])*len(v.get('modes',('rgb','legacy'))) for v in games.values())
    assert len(manifest['frames'])==expected
    manifest['elapsed_seconds']=time.monotonic()-start
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    template=(SOURCE/'gallery.html').read_text()
    (OUT/'index.html').write_text(template.replace('__MANIFEST__',json.dumps(manifest).replace('</','<\\/')))
    print(json.dumps(dict(status='ok',frames=len(manifest['frames']),variants=sum(len(v['names']) for v in games.values()),
        repeat_processes=len(manifest['repeat_processes']),seconds=manifest['elapsed_seconds'],html=str(OUT/'index.html')),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True,help='Fresh directory; never overwrite an existing gallery')
    parser.add_argument('--games',nargs='+',choices=tuple(GAMES),default=list(GAMES))
    args=parser.parse_args();main(args.out,args.games)
