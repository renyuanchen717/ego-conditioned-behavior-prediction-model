import numpy as np
import torch
from interaction_k12.frames import to_local
VERSION='interaction-scene-v1'
def convert_memory(s,r,context,f,labels,max_agents=0,max_maps=256):
    tracks={str(t.id):t for t in s.tracks};ego=context['metadata']['ego_track_id']
    meta=f['metadata'];ids=context['metadata']['ordered_track_ids'];ag=context['data']['agent']
    if ego!=meta['ego_track_id'] or ids[int(ag.av_index)]!=ego:raise ValueError('Conditioning ego mismatch')
    if meta['history_steps']!=11 or meta['T']!=80 or meta['K']!=6 or meta['d0']!=128:raise ValueError('Unexpected backbone dimensions')
    if meta['checkpoint_hash']!=r['pretrained_hash']:raise ValueError('Backbone mismatch')
    if labels['neighbor_track_ids'].tolist()!=meta['track_ids']:raise ValueError('Target IDs mismatch')
    if not all(t.valid for t in tracks[ego].states[10:]):raise ValueError('incomplete_ego_future')
    times=np.array(s.timestamps_seconds)
    if len(times)!=91 or not np.allclose(np.diff(times),.1,atol=.005):raise ValueError('Irregular timestamps')
    current=ag.current_observed.bool();pos=ag.position[:,10,:2]
    selected=[i for i in range(len(ids)) if current[i]]
    selected.sort(key=lambda i:(float((pos[i]-pos[int(ag.av_index)]).square().sum()),ids[i]))
    if max_agents and len(selected)>max_agents:
        # Preserve all original prediction targets. A resource limit cannot silently drop them.
        needed=set(meta['track_ids'])|{ego}
        chosen=selected[:max_agents]
        if not needed<={ids[i] for i in chosen}:raise ValueError('context_cap_would_drop_prediction_targets')
        selected=chosen
    kept=[ids[i] for i in selected];a=len(kept)
    ci={uid:i for i,uid in enumerate(meta['context_ids'])};ti={uid:i for i,uid in enumerate(meta['track_ids'])}
    if not set(meta['track_ids'])<=set(kept):raise ValueError('Prediction target missing from group')
    origin=pos[selected].clone();yaw=ag.heading[selected,10].clone()
    native=torch.zeros(a,6,80,2);prob=torch.zeros(a,6);gt=torch.zeros(a,80,2);valid=torch.zeros(a,80,dtype=torch.bool)
    target=torch.tensor([uid in ti for uid in kept]);nvalid=target.clone()
    for j,uid in enumerate(kept):
        if uid in ti:
            k=ti[uid]
            torch.testing.assert_close(origin[j],f['origin'][k]);torch.testing.assert_close(yaw[j],f['yaw'][k])
            native[j]=f['mu'][k];prob[j]=f['log_w'][k].exp()
            gt[j]=to_local(torch.from_numpy(labels['neighbor_future_world'][k]),origin[j],yaw[j])
            valid[j]=torch.from_numpy(labels['neighbor_future_valid'][k])
    hist=torch.tensor([[state.valid for state in tracks[uid].states[:11]] for uid in kept])
    maps=context['data']['map_polygon'];mpos=maps.position[:,:2];myaw=maps.orientation
    ego_current=pos[int(ag.av_index)].clone()
    indices=sorted(range(len(mpos)),key=lambda i:(float((mpos[i]-ego_current).square().sum()),i))[:max_maps]
    map_features=f['Em'][indices].clone();map_position=mpos[indices].clone();map_yaw=myaw[indices].clone()
    nearest=(origin[:,None]-mpos[None]).norm(dim=-1).argmin(-1) if len(mpos) else torch.full((a,),-1)
    plan=torch.from_numpy(labels['ego_future_world'][:,:2]).clone()
    if not torch.isfinite(plan).all():raise ValueError('Nonfinite ego plan')
    out=dict(native=native,native_prob=prob,native_valid=nvalid,agent_features=torch.stack([f['Ea'][ci[u]] for u in kept]),types=ag.type[selected].long(),origin=origin,yaw=yaw,velocity=ag.velocity[selected,10,:2].clone(),history_mask=hist,history=ag.position[selected,:11,:2].clone(),history_velocity=ag.velocity[selected,:11,:2].clone(),ego_role=torch.tensor([u==ego for u in kept]),agent_present_mask=torch.ones(a,dtype=torch.bool),context_mask=torch.ones(a,dtype=torch.bool),target_mask=target,map_features=map_features,map_position=map_position,map_yaw=map_yaw,map_mask=torch.ones(len(indices),dtype=torch.bool),nearest_map=nearest,ego_plan=plan,ego_plan_valid_mask=torch.ones(80,dtype=torch.bool),ego_current=ego_current,gt=gt,future_valid=valid,current_valid=hist[:,-1],agent_ids=kept,group_id=f'{s.scenario_id}:{times[10]:.6f}:{ego}',scene_id=s.scenario_id,ego_id=ego,split_group_id=r['group_id'],timestamp=float(times[10]),backbone_hash=r['pretrained_hash'],extractor_version=VERSION,frame='ego-current-centered world axes; native/GT target-local',max_context_before_cap=int(current.sum()),map_count_before_cap=len(mpos))
    for k in ['native','agent_features','map_features','origin','yaw','velocity']:
        if not torch.isfinite(out[k]).all():raise ValueError(f'Nonfinite {k}')
    return out
