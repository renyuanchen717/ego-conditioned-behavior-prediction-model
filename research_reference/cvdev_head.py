from pathlib import Path
import json,zlib,sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
R=Path(__file__).resolve().parent
O=R/'cv_deviation_head'
def save(file,data):(O/file).write_text(json.dumps(data,indent=2))
def base(n,q,t):
    s=n.new_tensor([0,.0185780134,.0108474232,.0319782235])[t.long()][:,None,None,None]
    return n+s*(n-(q[...,None,None]*n).sum(1,keepdim=True))
def features(n,q,t,plan,cv):
    # All coordinates use the target's current heading and origin.
    vel=torch.diff(cv,dim=1,prepend=2*cv[:,:1]-cv[:,1:2])*2
    direction=vel/vel.norm(dim=-1,keepdim=True).clamp_min(.2)
    delta=plan-cv
    lon=(delta*direction).sum(-1)
    lat=direction[...,0]*delta[...,1]-direction[...,1]*delta[...,0]
    dv=torch.diff(delta,dim=1,prepend=torch.zeros_like(delta[:,:1]))*2
    x=[q[:,:,None],F.one_hot(t.long()-1,3).float()[:,None].expand(-1,6,-1)]
    for i in [1,5,9,15]:
        # Include the ego deviation itself, its rate and mode-relative position.
        rel=(plan[:,None,i]-n[:,:,i])/20
        shared=torch.cat([delta[:,i]/10,dv[:,i]/5,lon[:,i,None]/10,lat[:,i,None]/10,vel[:,i]/15],1)
        x.extend([n[:,:,i]/30,rel,shared[:,None].expand(-1,6,-1)])
    for lo,hi in [(0,6),(6,10),(10,16)]:
        path=n[:,:,lo:hi];ego=plan[:,None,lo:hi]
        gap=(path-ego).norm(dim=-1)
        pairs=(path[:,:,:,None]-ego[:,:,None]).norm(dim=-1).flatten(2)
        index=pairs.argmin(2);width=hi-lo
        x.append(torch.stack([gap.amin(2)/20,(gap[:,:,-1]-gap[:,:,0])/10,pairs.amin(2)/20,
                              (index//width-index%width)*.5/4],2))
    return torch.cat(x,2).clamp(-10,10)
class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(64,96),nn.LayerNorm(96),nn.SiLU(),nn.Linear(96,64),nn.SiLU(),nn.Linear(64,8))
        nn.init.normal_(self.network[-1].weight,std=.006)
        nn.init.zeros_(self.network[-1].bias)
    def forward(self,x):
        z=self.network(x).reshape(-1,6,4,2)
        # Piecewise-linear residual anchored at t=0 with knots at 2,4,6,8s.
        knot=torch.cat([torch.zeros_like(z[:,:,:1]),4*torch.tanh(z/4)],2)
        time=torch.arange(1,17,device=x.device)/4
        left=time.floor().long().clamp_max(3);frac=(time-left)[None,None,:,None]
        return knot[:,:,left]*(1-frac)+knot[:,:,left+1]*frac
def error_modes(p,y,v):
    d=(p-y[:,None]).norm(dim=-1)
    result=[];valid=[]
    for h in [10,16]:
        m=v[:,:h]
        result.extend([(d[:,:,:h]*m[:,None]).sum(2)/m.sum(1).clamp_min(1)[:,None],d[:,:,h-1]])
        valid.extend([m.any(1),m[:,-1]])
    return torch.stack(result,2),torch.stack(valid,1)
def load(base_fn=base):
    data={k:[] for k in ['x','xc','off','y','v','t','sid']}
    for rank in [0,1]:
        with np.load(R/'classic_dispersion'/f'fit_{rank}.npz') as z:
            n,q,t,y,v=[torch.from_numpy(z[k]) for k in ['n','q','typ','y','v']]
        with np.load(O/f'fit_cv_{rank}.npz') as z:cv=torch.from_numpy(z['cv']);data['sid'].append(z['sid'])
        rec=torch.from_numpy(np.load(R/'classic_plan_mode_router/plans'/f'train_{rank}.npy'))
        for lo in range(0,len(t),512):
            sl=slice(lo,lo+512)
            off=base_fn(n[sl],q[sl],t[sl])
            for k,p in [('x',rec),('xc',cv)]:data[k].append(features(off,q[sl],t[sl],p[sl],cv[sl]))
            for k,a in [('off',off),('y',y[sl]),('v',v[sl]),('t',t[sl])]:data[k].append(a)
    ids=np.concatenate(data.pop('sid'));assert len(ids)==78539
    return {k:torch.cat(x).cuda() for k,x in data.items()},ids
def train():
    torch.manual_seed(713);torch.set_num_threads(3)
    d,ids=load()
    held=np.array([zlib.crc32(s.encode())%10==0 for s in ids])
    assert not set(ids[held])&set(ids[~held])
    test=set(json.loads((O/'test_velocity.json').read_text()));assert not set(ids)&test
    ti=torch.from_numpy(np.flatnonzero(~held)).cuda();vi=torch.from_numpy(np.flatnonzero(held)).cuda()
    net=Head().cuda();opt=torch.optim.AdamW(net.parameters(),lr=.0006,weight_decay=.002)
    denom=torch.tensor([.47,.9,.83,1.76],device='cuda');best=float('inf');history=[]
    def loss(ix,train):
        off=d['off'][ix];x=d['x'][ix]
        if train:x=torch.where(torch.rand(len(x),1,1,device='cuda')<.25,d['xc'][ix],x)
        res=net(x);of,valid=error_modes(off,d['y'][ix],d['v'][ix]);pr,_=error_modes(off+res,d['y'][ix],d['v'][ix])
        pool=of.amin(1).minimum(pr.amin(1))
        weights=torch.tensor([.15,.35,.15,.35],device='cuda')
        cost=(pool+.15*pr.amin(1) if train else pool)/denom
        if train:return ((cost*valid).sum(0)/valid.sum(0).clamp_min(1)*weights).sum()+.002*res.square().mean()
        return pool,valid
    sizes=torch.bincount(d['t'][ti].long(),minlength=4)
    sample=sizes[d['t'][ti].long()].float().rsqrt()
    batches=torch.multinomial(sample,1500*256,replacement=True).reshape(1500,256)
    for step,b in enumerate(batches,1):
        net.train();opt.zero_grad();value=loss(ti[b],True);value.backward()
        nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
        if step%250==0:
            with torch.no_grad():
                net.eval();s=torch.zeros(3,4,device='cuda');c=torch.zeros_like(s)
                for ii in vi.split(512):
                    err,v=loss(ii,False)
                    for j in [1,2,3]:
                        m=v&(d['t'][ii]==j)[:,None]
                        s[j-1]+=(err*m).sum(0);c[j-1]+=m.sum(0)
                assert (c>0).all()
                val=(s/c).mean(0);score=float((val[[1,3]]/denom[[1,3]]).mean())
                history.append(dict(step=step,val=val.tolist(),score=score))
                if score<best:
                    best=score;torch.save(dict(step=step,model={k:v.detach().cpu() for k,v in net.state_dict().items()}),O/'selected.pt')
                print(step,score,flush=True)
    save('training.json',dict(history=history,fit_scenes=len(set(ids[~held])),validation_scenes=len(set(ids[held])),fit_targets=len(ti),validation_targets=len(vi),cv_dropout=.25))
if __name__=='__main__':train()
