"""Online frozen-NDPNet + direct D proposals, no future GT or CV inputs."""
import sys,hashlib
from pathlib import Path
import torch
import bootstrap
REPO=bootstrap.REPO;sys.path.insert(0,str(REPO/'src'))
from interaction_k12 import InteractionK12
from interaction_k12.model import INPUT_KEYS
CK=REPO/'checkpoints/d_m2i/model.pt'
DIGEST='5444fdd24aee553991a7a97982b2bba6748cafc9fc6eaabbaacce54125c0b595'

def build_inputs(f,ctx,plan):
 ag=ctx['data']['agent'];ids=ctx['metadata']['ordered_track_ids'];ego=ctx['metadata']['ego_track_id'];ei=ids.index(ego)
 assert int(ag.av_index)==ei and not ag.position[:,11:].any() and not ag.valid_mask[:,11:].any()
 assert f['metadata']['ego_track_id']==ego
 pos=ag.position[:,10,:2];selected=[i for i in range(len(ids)) if ag.current_observed[i]]
 selected.sort(key=lambda i:(float((pos[i]-pos[ei]).square().sum()),ids[i]))
 kept=[ids[i] for i in selected];a=len(kept);ci={u:i for i,u in enumerate(f['metadata']['context_ids'])};ti={u:i for i,u in enumerate(f['metadata']['track_ids'])}
 assert set(ti)<=set(kept)
 native=torch.zeros(a,6,80,2);prob=torch.zeros(a,6);target=torch.tensor([u in ti for u in kept])
 origin=pos[selected].clone();yaw=ag.heading[selected,10].clone()
 for j,u in enumerate(kept):
  if u in ti:
   k=ti[u];torch.testing.assert_close(origin[j],f['origin'][k]);torch.testing.assert_close(yaw[j],f['yaw'][k]);native[j]=f['mu'][k];prob[j]=f['log_w'][k].exp()
 hist=ag.observed_history_mask[selected].clone();poly=ctx['data']['map_polygon'];mpos=poly.position[:,:2]
 ii=sorted(range(len(mpos)),key=lambda i:(float((mpos[i]-pos[ei]).square().sum()),i))[:256]
 nearest=(origin[:,None]-mpos[None]).norm(dim=-1).argmin(-1) if len(mpos) else torch.full((a,),-1)
 x=dict(native=native,native_prob=prob,native_valid=target.clone(),agent_features=torch.stack([f['Ea'][ci[u]] for u in kept]),types=ag.type[selected].long(),origin=origin,yaw=yaw,velocity=ag.velocity[selected,10,:2].clone(),history_mask=hist,ego_role=torch.tensor([u==ego for u in kept]),agent_present_mask=torch.ones(a,dtype=torch.bool),context_mask=torch.ones(a,dtype=torch.bool),target_mask=target,map_features=f['Em'][ii].clone(),map_position=mpos[ii].clone(),map_yaw=poly.orientation[ii].clone(),map_mask=torch.ones(len(ii),dtype=torch.bool),nearest_map=nearest,ego_plan=plan[:,:2].clone(),ego_plan_valid_mask=torch.ones(80,dtype=torch.bool),ego_current=pos[ei].clone())
 assert set(x)==set(INPUT_KEYS)
 return x,kept

class PlanForecast:
 def __init__(self):
  assert hashlib.sha256(CK.read_bytes()).hexdigest()==DIGEST
  ck=torch.load(CK,map_location='cpu',weights_only=True);assert ck['identity']['variant']=='D' and ck['epoch']==26
  self.model=InteractionK12(ck['identity']['config']['model']).cuda().eval();self.model.load_state_dict(ck['model'],strict=True);self.model.requires_grad_(False)
  from dm2ip_probability import ProbabilityHead
  pc=REPO/'checkpoints/dm2ip/model.pt'
  assert hashlib.sha256(pc.read_bytes()).hexdigest()=='e24256753c05c65f8da119d44a3a3b3acd2bb438cdcbba6b626f408872853766'
  headck=torch.load(pc,map_location='cpu',weights_only=False)
  assert headck['generator_sha256']==DIGEST
  self.head=ProbabilityHead().cuda().eval();self.head.load_state_dict(headck['head'],strict=True);self.head.requires_grad_(False)
  self.backbone_hash=ck['identity']['backbone_checkpoint_hash']
 @torch.inference_mode()
 def __call__(self,f,plans,ctx,history):
  assert f['metadata']['checkpoint_hash']==self.backbone_hash and plans.shape[1:]==(80,3)
  x,kept=build_inputs(f,ctx,plans[0]);pick=torch.tensor([kept.index(u) for u in f['metadata']['track_ids']],device='cuda');results=[];choices=[];probabilities=[]
  for start in range(0,len(plans),2):
   n=min(2,len(plans)-start)
   batch={k:v[None].expand(n,*v.shape).cuda() for k,v in x.items()};batch['ego_plan']=plans[start:start+n,:,:2].cuda()
   pool=self.model(batch)['pool'];prob=self.head(batch,pool).softmax(-1).index_select(1,pick);choice=prob.argmax(-1);p=pool.index_select(1,pick);assert torch.equal(p[:,:,:6],f['mu'].cuda()[None].expand(n,-1,-1,-1,-1));assert torch.isfinite(p).all();chosen=p.gather(2,choice[...,None,None,None].expand(-1,-1,1,80,2));results.append(chosen);choices.append(choice);probabilities.append(prob)
  self.last_choices=torch.cat(choices,0).cpu();self.last_probabilities=torch.cat(probabilities,0).cpu()
  return torch.cat(results,0)
