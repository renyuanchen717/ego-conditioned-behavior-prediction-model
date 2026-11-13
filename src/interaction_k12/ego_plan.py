import torch
from torch import nn
from .context import mlp
from .frames import to_local

class EgoPlanEncoder(nn.Module):
    def __init__(self,c):
        super().__init__();h=c['hidden_dim'];self.enabled=c['use_plan']
        self.plan=mlp(5,h,h);self.geometry=mlp(6,h,h)
        self.attention=nn.MultiheadAttention(h,c['heads'],dropout=c['dropout'],batch_first=True)
        self.null=nn.Parameter(torch.zeros(1,1,h));self.norm=nn.LayerNorm(h)
    def forward(self,x,z):
        b,a,h=z.shape
        if not self.enabled:
            # No read of supplied plan or any plan-derived geometry in history-only ablation.
            tokens=self.null.expand(b*a,80,h)
            cond=self.attention(z.reshape(b*a,1,h),tokens,tokens,need_weights=False)[0]
            return self.norm(z+cond.reshape(b,a,h)),tokens.reshape(b,a,80,h)
        plan=to_local(x['ego_plan'][:,None],x['origin'][:,:,None],x['yaw'][:,:,None])
        current=to_local(x['ego_current'][:,None],x['origin'],x['yaw'])
        velocity=torch.diff(plan,dim=2,prepend=current[:,:,None])*10
        t=torch.arange(1,81,device=z.device,dtype=z.dtype).view(1,1,80,1)/80
        tokens=self.plan(torch.cat([plan/30,velocity/15,t.expand(b,a,-1,-1)],-1))
        # Geometry: synchronous relative xy/distance + soft asynchronous proximity/lag + time.
        # Mode aggregation is permutation-invariant, independent of the new proposal queries.
        n=x['native'];delta=plan[:,:,None]-n
        sync=delta.norm(dim=-1)
        geometry=[]
        for start in range(0,a,8):
            end=start+8
            distances=(n[:,start:end,:,:,None]-plan[:,start:end,None,None]).norm(dim=-1)
            mass=(-distances/2).softmax(-1)
            times=torch.arange(80,device=z.device,dtype=z.dtype)*.1
            lag=times[None,:]-times[:,None]
            close=(mass*distances).sum(-1);arrival=(mass*lag).sum(-1)
            f=torch.cat([delta[:,start:end]/30,sync[:,start:end,...,None]/30,close[...,None]/30,arrival[...,None]/8,t[:, :,None].expand(b,min(8,a-start),6,-1,-1)],-1)
            geometry.append((self.geometry(f)*x['native_prob'][:,start:end,...,None,None]).sum(2))
        geo=torch.cat(geometry,1)*x['native_valid'][...,None,None]
        tokens=tokens+geo
        cond=self.attention(z.reshape(b*a,1,h),tokens.reshape(b*a,80,h),tokens.reshape(b*a,80,h),need_weights=False)[0]
        return self.norm(z+cond.reshape(b,a,h)),tokens
