"""Export grouped real scenes from audited FrozenFeatures caches + native scenario records.

Existing backbone context/features are reused; target rows are never grouped by sid heuristically.
"""
import argparse,hashlib,json,struct,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from interaction_k12.frames import to_local
from interaction_k12.data import read_manifest
VERSION='interaction-scene-v1'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,obj):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def convert(r,scenario_pb2,max_agents=0,max_maps=256):
    if r['source_type']!='factual':raise ValueError('Only factual logs authorized for this version')
    f=torch.load(r['feature_path'],map_location='cpu',weights_only=False)
    context=torch.load(r['context_path'],map_location='cpu',weights_only=False)
    with np.load(r['target_path'],allow_pickle=False) as z:labels={k:z[k] for k in z.files}
    if digest(r['source_path'])!=r['source_sha256']:raise ValueError('Native source hash mismatch')
    with open(r['source_path'],'rb') as h:
        length=struct.unpack('<Q',h.read(8))[0];h.read(4);payload=h.read(length)
    s=scenario_pb2.Scenario();s.ParseFromString(payload)
    if s.scenario_id!=r['root_scene_id'] or s.current_time_index!=10:raise ValueError('Scene/timestamp mismatch')
    tracks={str(t.id):t for t in s.tracks};ego=str(s.tracks[s.sdc_track_index].id)
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

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--output',required=True);p.add_argument('--womd-proto-path',required=True)
    p.add_argument('--train-limit',type=int,default=0);p.add_argument('--val-limit',type=int,default=0);p.add_argument('--max-map-tokens',type=int,default=256);p.add_argument('--max-context-agents',type=int,default=0)
    a=p.parse_args();torch.set_num_threads(2);sys.path.insert(0,a.womd_proto_path)
    from waymo_open_dataset.protos import scenario_pb2
    root=Path(a.source_root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    source={s:read_manifest(root/(s+'.jsonl')) for s in ('train','val','test')}
    for key in ['root_scene_id','group_id']:
        sets=[{r[key] for r in source[s]} for s in source]
        assert all(not sets[i]&sets[j] for i in range(3) for j in range(i)),f'Leaked {key}'
    report={'version':VERSION,'split_overlap':0,'source_counts':{k:len(v) for k,v in source.items()},'source_manifest_sha256':{k:digest(root/(k+'.jsonl')) for k in source},'exported':{},'errors':[],'exclusions':[],'started':time.time(),'phase':'running'}
    for split,limit in [('train',a.train_limit),('val',a.val_limit)]:
        selected=source[split][:limit or None];rows=[];dest=out/split;dest.mkdir(exist_ok=True)
        for r in selected:
            try:
                path=dest/(r['root_scene_id']+'.pt')
                if path.exists():sample=torch.load(path,map_location='cpu',weights_only=True)
                else:
                    sample=convert(r,scenario_pb2,a.max_context_agents,a.max_map_tokens)
                    tmp=path.with_suffix('.tmp');torch.save(sample,tmp);tmp.replace(path)
                rows.append({'path':str(path),'scene_id':sample['scene_id'],'group_id':sample['group_id'],'split_group_id':sample['split_group_id'],'targets':int(sample['target_mask'].sum()),'agents':len(sample['native']),'backbone_hash':sample['backbone_hash'],'extractor_version':VERSION})
            except Exception as e:
                item={'scene':r['root_scene_id'],'split':split,'error':str(e)}
                if str(e) in ['incomplete_ego_future','context_cap_would_drop_prediction_targets','Irregular timestamps']:report['exclusions'].append(item)
                else:item['traceback']=traceback.format_exc();report['errors'].append(item)
            report['exported'][split]=len(rows)
            if len(rows)%10==0:atomic(out/'audit.json',report)
        (out/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report.update(phase='complete' if not report['errors'] else 'failed',elapsed_s=time.time()-report['started']);atomic(out/'audit.json',report);print(json.dumps(report,indent=2))
    if report['errors']:raise RuntimeError('Cache errors block training')
if __name__=='__main__':main()
