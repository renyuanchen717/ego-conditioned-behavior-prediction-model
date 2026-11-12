"""Evaluate a prepared target-local cache; no GT is passed to the model."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from recorded_k12 import RecordedK12
from recorded_k12.metrics import displacement_metrics
p=argparse.ArgumentParser();p.add_argument('cache');p.add_argument('--checkpoints',default='checkpoints/locked');p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=256)
a=p.parse_args();torch.set_num_threads(2)
model=RecordedK12(a.checkpoints).to(a.device);collected={k:{} for k in ['frozen6','cv12','recorded12']}
with np.load(a.cache,allow_pickle=False) as z:
    for start in range(0,len(z['n']),a.batch_size):
        sl=slice(start,start+a.batch_size)
        x,q,t,b,r,y,v=[torch.as_tensor(z[k][sl],device=a.device) for k in ['n','q','typ','cv','rec','y','v']]
        x,q,b,r,y=[u.float() for u in (x,q,b,r,y)]
        with torch.inference_mode():
            pools=[x,model(x,q,t,b,b),model(x,q,t,b,r)]
            for arm,pool in zip(collected,pools):
                for name,value in displacement_metrics(pool,y,v).items():collected[arm].setdefault(name,[]).append(value.cpu())
out={}
for arm,metrics in collected.items():
    out[arm]={}
    for name,parts in metrics.items():
        val=torch.cat(parts);ok=torch.isfinite(val)
        out[arm][name]={'mean_m':float(val[ok].mean()) if ok.any() else None,'valid_targets':int(ok.sum())}
Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
