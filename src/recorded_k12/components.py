"""Inference components extracted unchanged from the locked research implementation."""
import torch
from torch import nn
from torch.nn import functional as F

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
def cues(n,cv,rec):
    velocity=n[:,:,-1]-n[:,:,9]
    heading=velocity/velocity.norm(dim=-1,keepdim=True).clamp_min(.1)
    heading=torch.where((velocity.norm(dim=-1)<.1)[:,:,None],n.new_tensor([1.,0.]),heading)
    normal=torch.stack([-heading[...,1],heading[...,0]],-1)
    basis=torch.stack([heading,normal],-2)
    dev=(rec-cv)[:,None].expand_as(n)
    gap=(n-cv[:,None]).norm(dim=-1)
    signed=[]
    for lo,hi in ((0,6),(6,10),(10,16)):
        att=torch.softmax(-gap[:,:,lo:hi]/6,dim=-1)
        action=(att[...,None]*dev[:,:,lo:hi]).sum(2)
        signed.append(torch.einsum('bkij,bkj->bki',basis,action))
    return torch.cat(signed,-1),basis

class Response(nn.Module):
    def __init__(self):
        super().__init__()
        self.coeff=nn.Sequential(nn.Linear(64,48),nn.SiLU(),nn.Linear(48,12))
        nn.init.zeros_(self.coeff[-1].weight);nn.init.zeros_(self.coeff[-1].bias)
    def forward(self,context,signal,basis):
        mat=self.coeff(context).reshape(*signal.shape[:2],2,6)
        local=(mat@(signal/8).clamp(-3,3).unsqueeze(-1)).squeeze(-1)
        return torch.einsum('bkij,bki->bkj',basis,2*torch.tanh(local))
def descriptors(target,plan):
    # A soft spatial conflict of the two paths. Times are compared at nearby
    # spatial points, without requiring agents to arrive simultaneously.
    sep=torch.linalg.vector_norm(target[:,:,:,None]-plan[:,None,None,:],dim=-1)
    grid=torch.arange(1,17,device=target.device)*.5
    lag=grid[None,:]-grid[:,None]  # ego arrival minus target arrival
    sync=sep.diagonal(dim1=-2,dim2=-1)
    out=[]
    for lo,hi in ((0,8),(8,16)):
        distances=sep[:,:,lo:hi]
        mass=torch.softmax(-distances.flatten(2)/1.5,-1).reshape_as(distances)
        strength=torch.exp(-distances.amin((-1,-2))/8)
        delay=lag[lo:hi]
        out.append(strength*(mass*delay).sum((-1,-2)))
        out.append(4*strength*(mass*torch.tanh(delay/.75)).sum((-1,-2)))
        out.append(8*torch.exp(-sync[:,:,lo:hi].amin(2)/8))
    return torch.stack(out,-1)
def geometry5(x,plan):
    # Arrival order and synchronous proximity before 5s only.
    d=(x[:,:,:10,None]-plan[:,None,None,:10]).norm(dim=-1)
    time=torch.arange(10,device=x.device)*.5
    lag=time[None,:]-time[:,None]
    result=[]
    for lo,hi in [(0,5),(5,10)]:
        window=d[:,:,lo:hi]
        weight=(-window.flatten(2)/1.5).softmax(-1).reshape_as(window)
        strength=(-window.amin((-1,-2))/8).exp()
        result.extend([strength*(weight*lag[lo:hi]).sum((-1,-2)),
            4*strength*(weight*(lag[lo:hi]/.75).tanh()).sum((-1,-2)),
            8*(-d.diagonal(dim1=-2,dim2=-1)[:,:,lo:hi].amin(2)/8).exp()])
    return torch.stack(result,-1)
