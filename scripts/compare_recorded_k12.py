"""Additional historical reference on common 2Hz knots only; not a new training arm."""
import argparse,json
from pathlib import Path
import torch,numpy as np
from torch.utils.data import DataLoader
from interaction_k12 import InteractionK12
from interaction_k12.data import SceneDataset,collate,model_inputs
from interaction_k12.frames import to_local
from recorded_k12 import RecordedK12
from recorded_k12.metrics import displacement_metrics
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--recorded-checkpoints',default='checkpoints/locked');p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
a=p.parse_args();torch.set_num_threads(2);ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
new=InteractionK12(ck['identity']['config']['model']).to(a.device).eval();new.load_state_dict(ck['model'])
old=RecordedK12(a.recorded_checkpoints).to(a.device).eval();parts={k:{} for k in ['frozen6','direct12','recorded12']}
with torch.inference_mode():
 for batch in DataLoader(SceneDataset(a.manifest),batch_size=2,collate_fn=collate):
    x=model_inputs(batch,a.device);out=new(x);mask=x['target_mask']&x['native_valid']&x['context_mask']
    ego_velocity=(x['velocity']*x['ego_role'][...,None]).sum(1)
    t=torch.arange(1,17,device=a.device)*.5
    cv_scene=x['ego_current'][:,None]+ego_velocity[:,None]*t[None,:,None]
    cv=to_local(cv_scene[:,None],x['origin'][:,:,None],x['yaw'][:,:,None])[mask]
    rec=to_local(x['ego_plan'][:,None,4::5],x['origin'][:,:,None],x['yaw'][:,:,None])[mask]
    n=x['native'][mask][:,:,4::5];q=x['native_prob'][mask];typ=x['types'][mask]
    pred=old(n,q,typ,cv,rec)
    gt=batch['gt'].to(a.device)[mask][:,4::5];valid=batch['future_valid'].to(a.device)[mask][:,4::5]
    for name,pool in [('frozen6',n),('direct12',out['pool'][mask][:,:,4::5]),('recorded12',pred)]:
        for k,v in displacement_metrics(pool,gt,valid).items():parts[name].setdefault(k,[]).append(v.cpu())
report={'protocol':'same validation targets, common 2Hz future knots; historical Recorded K12 has different training exposure','metrics':{}}
for name,metrics in parts.items():
 report['metrics'][name]={}
 for k,v in metrics.items():
    v=torch.cat(v);finite=torch.isfinite(v)
    report['metrics'][name][k]={'mean':float(v[finite].mean()),'count':int(finite.sum())}
Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
