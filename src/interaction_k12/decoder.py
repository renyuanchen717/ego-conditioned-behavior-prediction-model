import torch
from torch import nn
from .context import mlp

class DirectDecoder(nn.Module):
    def __init__(self,c):
        super().__init__();h=c['hidden_dim'];self.output_scale=c['output_scale_m'];self.queries=nn.Parameter(torch.randn(6,h)*.02)
        self.attention=nn.MultiheadAttention(h,c['heads'],dropout=c['dropout'],batch_first=True)
        self.norm=nn.LayerNorm(h);self.output=mlp(h,2*h,160)
    def forward(self,z,plan_tokens,native_tokens,map_tokens,x):
        b,a,h=z.shape
        query=z[:,:,None]+self.queries[None,None]
        memory=torch.cat([plan_tokens,native_tokens,map_tokens],2)
        mask=torch.cat([torch.ones(b,a,80,device=z.device,dtype=torch.bool),x['native_valid'][...,None].expand(-1,-1,6),x['map_mask'][:,None].expand(-1,a,-1)],-1)
        # Memory always includes plan or learned-null tokens, even for isolated agents.
        att=self.attention(query.reshape(b*a,6,h),memory.reshape(b*a,-1,h),memory.reshape(b*a,-1,h),key_padding_mask=~mask.reshape(b*a,-1),need_weights=False)[0]
        # Direct absolute future positions in target-local coordinates. No native/residual addition.
        return self.output(self.norm(query.reshape(b*a,6,h)+att)).reshape(b,a,6,80,2)*self.output_scale*x['context_mask'][...,None,None,None]
