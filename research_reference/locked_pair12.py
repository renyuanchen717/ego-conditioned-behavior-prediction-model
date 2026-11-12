import hashlib,json,sys,shutil
from pathlib import Path
import numpy as np
import torch
from separate_horizon12 import R,Head,Response,features,geometry5,cues,descriptors
from ego_history_audit import read
from pair_endpoint_plan_ci import err
from threadpoolctl import threadpool_limits
O=R/'locked_pair12'
P=R.parent/'StagedImpact_WOMD/womd_pair_conditioned_evaluation'
RAW=R.parent/'NDPNet_adapter/womd_interactive_raw'
sources={'anchor.pt':R/'cv_direct_crossfit/fold_0/selected.pt','five.pt':R/'separate_horizon12/fold_0.pt','eight.pt':R/'conflict_order_response/fold_0.pt'}
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def lock():
    O.mkdir(exist_ok=True)
    files={}
    for name,p in sources.items():
        dest=O/name
        if not dest.exists(): shutil.copyfile(p,dest)
        assert digest(dest)==digest(p)
        files[name]=digest(dest)
    protocol=dict(weights=files,implementation={p:digest(R/p) for p in ('locked_pair12.py','separate_horizon12.py','signed_plan_response.py','cvdev_head.py','conflict_order_response.py')},pair_manifest_sha256=digest(P/'pair_manifest.jsonl'),arms=['frozen6','cv12','recorded12'],metrics=['5s minADE','5s minFDE','8s minADE','8s minFDE'],aggregation='class macro all eligible pairs; matched vehicle targets valid at 8s for comparison to classic. Independent scene bootstrap of difference in CV minus recorded gain, 2000 draws.')
    dest=O/'protocol.json'
    if dest.exists():assert json.loads(dest.read_text())==protocol
    else:dest.write_text(json.dumps(protocol,indent=2))
@torch.inference_mode()
def main(rank):
    cfg=json.loads((O/'protocol.json').read_text())
    for name,h in cfg['weights'].items():assert digest(O/name)==h
    for name,h in cfg['implementation'].items():assert digest(R/name)==h
    assert digest(P/'pair_manifest.jsonl')==cfg['pair_manifest_sha256']
    torch.cuda.set_device(rank);torch.set_num_threads(2)
    anchor=Head().cuda().eval()
    anchor.load_state_dict(torch.load(O/'anchor.pt',weights_only=True)['anchor'])
    nets=[];scales=[]
    for filename in ('five.pt','eight.pt'):
        ck=torch.load(O/filename,weights_only=True);net=Response().cuda().eval()
        net.load_state_dict(ck['state']);nets.append(net);scales.append(ck['scale'])
    manifest={r['root_scene_id']:r for r in map(json.loads,(P/'pair_manifest.jsonl').read_text().splitlines())}
    exclude={json.loads(line)['root_scene_id'] for split in ['train','val','test'] for line in (R/(split+'.jsonl')).open()}
    for j in range(rank,150,2):
        name=f'{j:03}.npz'
        with np.load(R/'pair_endpoint_plan_routers/recorded'/name) as prev:ids=prev['ids']
        assert set(ids)=={s for s,r in manifest.items() if r['source_shard']==j and s not in exclude}
        with np.load(P/'chunks'/name) as z:
            order={s:i for i,s in enumerate(z['scenario_id'])};ix=[order[s] for s in ids]
            x,q,t,oid,gt,valid=[z[k][ix,0] for k in ['frozen_NDPNet_mu','frozen_NDPNet_prob','object_type','object_id','gt','valid']]
            cid=z['conditioning_id'][ix]
        rec=np.empty((len(ids),16,2));cv=np.empty_like(rec)
        remaining={s:i for i,s in enumerate(ids)}
        for scene in read(RAW/f'validation_interactive.tfrecord-{j:05}-of-00150'):
            if scene.scenario_id not in remaining:continue
            i=remaining.pop(scene.scenario_id);row=manifest[scene.scenario_id]
            assert int(row['conditioning_id'])==cid[i] and int(row['prediction_id'])==oid[i] and cid[i]!=oid[i]
            tracks={tr.id:tr for tr in scene.tracks};states=tracks[int(cid[i])].states;cur=states[10]
            assert all(p.valid for p in states[10:])
            sdc=scene.tracks[scene.sdc_track_index].states[10];off=np.array([sdc.center_x,sdc.center_y])
            current=tracks[int(oid[i])].states[10]
            np.testing.assert_allclose(np.array([current.center_x,current.center_y])-off,gt[i,10,:2],atol=.002,rtol=0)
            rec[i]=np.array([[p.center_x,p.center_y] for p in states[15::5]])-off
            cv[i]=np.array([cur.center_x,cur.center_y])-off+np.arange(1,17)[:,None]*.5*[cur.velocity_x,cur.velocity_y]
        assert not remaining and valid[:,10].all()
        mask=valid[:,15::5];origin=gt[:,10,:2];yaw=gt[:,10,4];c,s=np.cos(yaw),np.sin(yaw)
        rot=np.stack([c,-s,s,c],1).reshape(-1,2,2)
        def local(a):
            return torch.from_numpy(np.einsum('n...i,nij->n...j',a-origin.reshape((len(ids),)+(1,)*(a.ndim-2)+(2,)),rot).astype(np.float32)).cuda()
        raw=err(x,gt[:,15::5,:2],mask)[:,2:]
        x,b,r,y=map(local,[x,cv,rec,gt[:,15::5,:2]])
        q=torch.from_numpy(q).cuda();typ=torch.from_numpy(t).cuda()
        ctx=features(x,q,typ,b,b);ref=x+anchor(ctx);axes=cues(x,b,b)[1]
        times=torch.arange(1,17,device=x.device)*.5
        def smooth(u):return (u*u*(3-2*u))[None,None,:,None]
        h5=smooth((times/5).clamp(max=1));h8=smooth(((times-5)/3).clamp(min=0))
        arms=[]
        for pool in [x,torch.cat([x,ref],1)]:
            arms.append(err(pool.cpu().numpy(),y.cpu().numpy(),mask)[:,2:])
        d5=scales[0]*nets[0](ctx,geometry5(x,r)-geometry5(x,b),axes)
        d8=scales[1]*nets[1](ctx,descriptors(x,r)-descriptors(x,b),axes)
        cond=ref+h5*(1-h8)*d5[:,:,None]+h8*d8[:,:,None]
        arms.append(err(torch.cat([x,cond],1).cpu().numpy(),y.cpu().numpy(),mask)[:,2:])
        np.testing.assert_allclose(raw,arms[0],atol=5e-5,rtol=0)
        np.savez(O/name,sid=ids,typ=t,e=np.stack(arms,1),v8=mask[:,-1])
if __name__=='__main__':
    with threadpool_limits(2):
        if sys.argv[1]=='lock':lock()
        else:main(int(sys.argv[1]))
