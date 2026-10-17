"""Verify every retained PNG against its exact native float32 observation."""
import gzip
import argparse
import hashlib
import json
from pathlib import Path
import struct
import zlib
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--gallery',type=Path,required=True)
root=parser.parse_args().gallery.resolve()
data=json.loads((root/'manifest.json').read_text())
seen=set();paired={}
for frame in data['frames']:
    identity=(frame['game'],frame['mode'],frame['representation'],frame['moment'])
    assert identity not in seen;seen.add(identity)
    raw=gzip.decompress((root/frame['raw_file']).read_bytes())
    assert hashlib.sha256(raw).hexdigest()==frame['raw_sha256']
    assert len(raw)==frame['bytes']==int(np.prod(frame['shape_hwc']))*4
    pixels=(root/frame['image']).read_bytes()
    assert hashlib.sha256(pixels).hexdigest()==frame['image_sha256']
    assert pixels[:8]==b'\x89PNG\r\n\x1a\n'
    offset=8;chunks=[]
    while offset<len(pixels):
        size=struct.unpack_from('>I',pixels,offset)[0];kind=pixels[offset+4:offset+8]
        payload=pixels[offset+8:offset+8+size];crc=struct.unpack_from('>I',pixels,offset+8+size)[0]
        assert zlib.crc32(kind+payload)&0xffffffff==crc
        if kind==b'IHDR':width,height,depth,color,*_=struct.unpack('>IIBBBBB',payload)
        if kind==b'IDAT':chunks.append(payload)
        offset+=size+12
    h,w,c=frame['shape_hwc'];assert (height,width,depth,color)==(h,w,8,2)
    scanlines=np.frombuffer(zlib.decompress(b''.join(chunks)),dtype=np.uint8).reshape(h,w*3+1)
    assert not scanlines[:,0].any()
    values=np.frombuffer(raw,dtype='<f4').reshape(h,w,c)
    if c==1:values=np.repeat(values,3,axis=2)
    assert np.array_equal(scanlines[:,1:].reshape(h,w,3),np.rint(values*255).astype(np.uint8))
    key=(frame['game'],frame['moment'])
    state={k:frame[k] for k in ('decision','episode','episode_decision','tick','rng','state')}
    paired.setdefault(key,state);assert paired[key]==state
    assert frame['episode_decision']>0 and frame['repeat_exact']
expected=3*sum(len(v['names'])*len(v.get('modes',('rgb','legacy'))) for v in data['games'].values())
assert len(seen)==expected and len(paired)==len(data['games'])*3
result=dict(status='ok',pngs_verified=len(seen),lossless_memory_verified=len(seen),paired_world_states=len(paired),
    all_variants_and_formats_paired=True,gpu_used=False,policy_executed=False,
    display='round-to-nearest RGB8; exact original floats retained separately')
(root/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
