"""Functional sensitivity under two kinematically checked same-path plans; no counterfactual accuracy claim."""
import argparse,json
from pathlib import Path
import torch
from interaction_k12 import InteractionK12
from interaction_k12.data import SceneDataset,collate,model_inputs
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--limit',type=int,default=10)
a=p.parse_args();torch.set_num_threads(2)
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True);model=InteractionK12(ck['identity']['config']['model']).to(a.device).eval()
dataset=SceneDataset(a.manifest);rows=[]
with torch.inference_mode():
 for j in range(min(a.limit,len(dataset))):
    batch=collate([dataset[j]]);x=model_inputs(batch,a.device)
    plan=x['ego_plan'];current=x['ego_current'];full=torch.cat([current[:,None],plan],1)
    t=torch.arange(1,81,device=a.device)*.1;warped=t-.1*(t/8).square()
    index=warped/.1;left=index.floor().long();frac=(index-left)[None,:,None]
    alternative=full[:,left]*(1-frac)+full[:,left+1]*frac
    velocity=torch.diff(full,dim=1)*10;new_velocity=torch.diff(torch.cat([current[:,None],alternative],1),dim=1)*10
    acceleration=torch.diff(velocity,dim=1)*10;new_acceleration=torch.diff(new_velocity,dim=1)*10
    kin_ok=bool(new_velocity.norm(dim=-1).max()<=velocity.norm(dim=-1).max()+.5 and new_acceleration.norm(dim=-1).max()<=acceleration.norm(dim=-1).max()+2)
    record={'scene_id':dataset.rows[j]['scene_id'],'kinematic_check_passed':kin_ok}
    if kin_ok:
        base=model(x);changed=model({**x,'ego_plan':alternative})
        mask=x['target_mask']&x['context_mask']
        assert torch.equal(base['pool'][:,:,:6],changed['pool'][:,:,:6])
        delta=(base['new']-changed['new']).norm(dim=-1)[mask]
        record.update(mean_new_change_m=float(delta.mean()),max_new_change_m=float(delta.max()),native_exactly_unchanged=True,ego_plan_mean_change_m=float((plan-alternative).norm(dim=-1).mean()))
    rows.append(record)
report={'scope':'functional sensitivity, NOT reaction accuracy','alternative':'time-warped recorded path; terminal lag 0.1s; same spatial path','feasibility':'kinematic/path-consistency checked; traffic-rule and collision feasibility NOT certified','target_GT_used_for_alternative_accuracy':False,'rows':rows}
Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
