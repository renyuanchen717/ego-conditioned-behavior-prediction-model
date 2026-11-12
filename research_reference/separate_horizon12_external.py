"""Full 12-trajectory evaluation on the fifth disjoint vehicle validation slice."""
from pathlib import Path
import sys
import numpy as np
import torch
from vehicle_replication import R,Head,features,paths,run,exclusion

from separate_horizon12 import geometry5,Response,cues,O as FIT
O=FIT/'classic'
O.mkdir(exist_ok=True)

def main(rank):
    torch.cuda.set_device(rank)
    model=Head().cuda().eval()
    model.load_state_dict(torch.load(paths[0],weights_only=True)['anchor'])
    ck=torch.load(FIT/'fold_0.pt',weights_only=True)
    five=Response().cuda().eval()
    five.load_state_dict(ck['state'])
    exclude=exclusion()
    previous=set()
    for name in ['action_pair_selector','separate_gain_cost','removal_hurdle','endpoint_isolation']:
        for file in (R/name/'classic').glob('*.npz'):
            with np.load(file) as z:
                previous.update(z['sid'])
    assert len(previous)==20311
    def audit(shard,ids,oid,x,q,b,r,y,proposals):
        assert not set(ids)&previous
        with np.load(R/'classic_recorded_ego'/'shards'/f'{shard:03}.npz') as z:
            all_ids=np.repeat(z['ids'],z['sizes'])
            keep=~np.isin(all_ids,list(exclude))&(z['typ']==1)&z['valid'][:,10]&z['valid'][:,-1]
            assert np.array_equal(ids,all_ids[keep]) and np.array_equal(oid,z['oid'][keep])
            gt=z['gt'][keep]
            valid=z['valid'][keep][:,15::5]
        yaw=gt[:,10,4]
        cs,sn=np.cos(yaw),np.sin(yaw)
        rot=np.stack([cs,-sn,sn,cs],1).reshape(-1,2,2)
        future=torch.from_numpy(np.einsum('nti,nij->ntj',gt[:,15::5,:2]-gt[:,10:11,:2],rot)).cuda()
        mask=torch.from_numpy(valid).cuda().bool()
        torch.testing.assert_close(future[:,-1],y)
        ref=x+model(features(x,q,torch.ones(len(x),device=x.device,dtype=torch.long),b,b))
        torch.testing.assert_close(ref[:,:,-1],proposals[0])
        u=torch.arange(1,17,device=x.device)/16
        blend=(u*u*(3-2*u))[None,None,:,None]
        def metrics(pool):
            err=(pool-future[:,None]).norm(dim=-1)
            cols=[]
            for h in [6,10,16]:
                m=mask[:,:h]
                ade=err[:,:,:h].masked_fill(~m[:,None],0).sum(-1)/m.sum(1).clamp_min(1)[:,None]
                cols.extend([ade.amin(1).masked_fill(~m.any(1),torch.nan),err[:,:,h-1].amin(1).masked_fill(~m[:,-1],torch.nan)])
            return torch.stack(cols,1)
        vals=[metrics(x)]
        context=features(x,q,torch.ones(len(x),device=x.device,dtype=torch.long),b,b)
        axes=cues(x,b,b)[1]
        geo=geometry5(x,b)
        times=torch.arange(1,17,device=x.device)*.5
        u5=(times/5).clamp(max=1)
        u8=((times-5)/3).clamp(min=0)
        h5=(u5*u5*(3-2*u5))[None,None,:,None]
        h8=(u8*u8*(3-2*u8))[None,None,:,None]
        for end,act in zip(proposals,[b,r,2*b-r]):
            delta5=ck['scale']*five(context,geometry5(x,act)-geo,axes)
            conditional=ref+h5*(1-h8)*delta5[:,:,None]+h8*(end-ref[:,:,-1])[:,:,None]
            torch.testing.assert_close(conditional[:,:,9],ref[:,:,9]+delta5)
            torch.testing.assert_close(conditional[:,:,-1],end)
            vals.append(metrics(torch.cat([x,conditional],1)))
        current=torch.stack(vals,1).cpu().numpy()
        with np.load(R/'twelve_trajectory_external'/f'{shard:03}.npz') as baseline:
            assert np.array_equal(ids,baseline['sid'])
            np.testing.assert_allclose(current[:,:,5],baseline['metrics'][:,:,5],atol=1e-5)
            np.testing.assert_allclose(current[:,:2],baseline['metrics'][:,:2],atol=1e-5)
        np.savez(O/f'{shard:03}.npz',sid=ids,metrics=torch.stack(vals,1).cpu().numpy())
    run(rank,selection_audit=audit,shard_ids=list(range(4,150,5)))
if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(2):
        main(int(sys.argv[1]))
