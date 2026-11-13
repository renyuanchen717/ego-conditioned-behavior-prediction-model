import torch
from torch import nn
from .frames import to_local

def mlp(i,h,o):return nn.Sequential(nn.Linear(i,h),nn.SiLU(),nn.Linear(h,o))
def masked_softmax(scores,mask):
    p=scores.masked_fill(~mask,-1e4).softmax(-1)*mask
    return p/p.sum(-1,keepdim=True).clamp_min(1e-9)

class FrozenContextAdapter(nn.Module):
    def __init__(self,c):
        super().__init__();h=c['hidden_dim'];d=c['feature_dim']
        self.agent=nn.Linear(d,h);self.map=nn.Linear(d,h);self.type_embedding=nn.Embedding(4,h)
        self.native=mlp(161,h,h);self.state=mlp(4,h,h);self.map_relation=mlp(4,h,h)
        self.map_query=nn.Linear(h,h);self.norm=nn.LayerNorm(h)
    def forward(self,x):
        native=x['native'];q=x['native_prob'];b,a=native.shape[:2]
        # Shared sequence encoder; fixed timestamp ordering preserves temporal information.
        mode=self.native(torch.cat([native.flatten(-2)/30,q[...,None]],-1))
        z=(mode*q[...,None]).sum(2)
        z=z*x['native_valid'][...,None]
        vel=to_local(x['velocity'],torch.zeros_like(x['velocity']),x['yaw'])/15
        state=torch.cat([vel,x['history_mask'].float().mean(-1,keepdim=True),x['ego_role'][...,None].float()],-1)
        z=z+self.agent(x['agent_features'])+self.type_embedding(x['types'].clamp(0,3))+self.state(state)
        rel=to_local(x['map_position'][:,None],x['origin'][:,:,None],x['yaw'][:,:,None])/50
        angle=x['map_yaw'][:,None]-x['yaw'][:,:,None]
        maps=self.map(x['map_features'])[:,None]+self.map_relation(torch.cat([rel,angle.cos()[...,None],angle.sin()[...,None]],-1))
        score=(self.map_query(z)[:,:,None]*maps).sum(-1)/(z.shape[-1]**.5)
        w=masked_softmax(score,x['map_mask'][:,None].expand_as(score))
        z=self.norm(z+(w[...,None]*maps).sum(2))
        return z,mode,maps
