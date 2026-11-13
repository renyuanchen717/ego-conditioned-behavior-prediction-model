import torch
from torch import nn
from .context import mlp,masked_softmax
from .frames import to_local

class AgentInteraction(nn.Module):
    def __init__(self,c):
        super().__init__();h=c['hidden_dim'];self.heads=c['heads'];self.h=h
        self.qkv=nn.Linear(h,3*h);self.edge=mlp(13,h,self.heads)
        self.out=nn.Linear(h,h);self.drop=nn.Dropout(c['dropout'])
        self.norm1=nn.LayerNorm(h);self.norm2=nn.LayerNorm(h);self.ff=mlp(h,2*h,h)
    def forward(self,z,x):
        b,a,h=z.shape
        rel=to_local(x['origin'][:,None],x['origin'][:,:,None],x['yaw'][:,:,None])/50
        vel=to_local(x['velocity'][:,None]-x['velocity'][:,:,None],torch.zeros_like(rel),x['yaw'][:,:,None])/15
        angle=x['yaw'][:,None]-x['yaw'][:,:,None]
        typ=torch.nn.functional.one_hot(x['types'].clamp(0,3),4).float()
        # Receiver-local edges, source type, history coverage and shared nearest map polygon.
        road=((x['nearest_map'][:,:,None]==x['nearest_map'][:,None])&(x['nearest_map'][:,:,None]>=0)).float()
        coverage=x['history_mask'].float().mean(-1)
        edges=torch.cat([rel,vel,angle.cos()[...,None],angle.sin()[...,None],typ[:,None].expand(-1,a,-1,-1),coverage[:,:,None,None].expand(-1,-1,a,-1),coverage[:,None,:,None].expand(-1,a,-1,-1),road[...,None]],-1)
        q,k,v=self.qkv(z).chunk(3,-1)
        q,k,v=[u.view(b,a,self.heads,h//self.heads).transpose(1,2) for u in (q,k,v)]
        logits=q@k.transpose(-1,-2)/(h//self.heads)**.5+self.edge(edges).permute(0,3,1,2)
        mask=x['context_mask'][:,None,None,:].expand_as(logits)
        value=(masked_softmax(logits,mask)@v).transpose(1,2).reshape(b,a,h)
        z=self.norm1(z+self.drop(self.out(value)));z=self.norm2(z+self.drop(self.ff(z)))
        return z*x['context_mask'][...,None]
