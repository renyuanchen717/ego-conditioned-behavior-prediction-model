import torch
from torch import nn
from .context import FrozenContextAdapter
from .ego_plan import EgoPlanEncoder
from .interaction import AgentInteraction
from .decoder import DirectDecoder

INPUT_KEYS=('native','native_prob','native_valid','agent_features','types','origin','yaw','velocity','history_mask','ego_role','agent_present_mask','context_mask','target_mask','map_features','map_position','map_yaw','map_mask','nearest_map','ego_plan','ego_plan_valid_mask','ego_current')
DEFAULT=dict(hidden_dim=128,heads=4,dropout=.1,feature_dim=128,use_plan=True,post_interaction=True,interaction_layers=2,output_scale_m=30.)
class FrozenBackboneGuard(nn.Module):
    """Use this for online extraction; cached training does not load a backbone at all."""
    def __init__(self,backbone):
        super().__init__();self.backbone=backbone.eval().requires_grad_(False)
    def train(self,mode=True):super().train(False);return self
    @torch.no_grad()
    def forward(self,*args,**kwargs):
        self.backbone.eval();return self.backbone(*args,**kwargs)

class InteractionK12(nn.Module):
    def __init__(self,config=None):
        super().__init__();self.config={**DEFAULT,**(config or {})}
        self.context=FrozenContextAdapter(self.config);self.ego=EgoPlanEncoder(self.config)
        self.interaction=nn.ModuleList([AgentInteraction(self.config) for _ in range(self.config['interaction_layers'])])
        self.decoder=DirectDecoder(self.config)
    def forward(self,inputs):
        if set(inputs)!=set(INPUT_KEYS):raise ValueError('Inference requires only documented inputs, never GT/future_valid')
        x=dict(inputs);native=x['native']
        if native.shape[-3:]!=(6,80,2):raise ValueError('Expected native [B,A,6,80,2]')
        if self.config['use_plan'] and not x['ego_plan_valid_mask'].all():raise ValueError('Incomplete supplied ego plan')
        active=x['context_mask']&x['agent_present_mask'];x['context_mask']=active
        # Ignore invalid/padded nodes before ANY encoder/geometry; padding may contain NaNs.
        for key in ['native','native_prob','agent_features','origin','yaw','velocity','history_mask','types','ego_role']:
            value=x[key];mask=active.reshape(*active.shape,*([1]*(value.ndim-2)))
            x[key]=torch.where(mask,value,torch.zeros_like(value))
        x['native_valid']=x['native_valid']&active&~x['ego_role']
        x['native']=torch.where(x['native_valid'][...,None,None,None],x['native'],torch.zeros_like(x['native']))
        x['native_prob']=torch.where(x['native_valid'][...,None],x['native_prob'],torch.zeros_like(x['native_prob']))
        for key in ['map_features','map_position','map_yaw']:
            value=x[key];mask=x['map_mask'].reshape(*x['map_mask'].shape,*([1]*(value.ndim-2)))
            x[key]=torch.where(mask,value,torch.zeros_like(value))
        z,modes,maps=self.context(x);z,plans=self.ego(x,z)
        z=z*active[...,None]
        # History-only ablation keeps static interactions; direct-plan bypasses these layers.
        if self.config['post_interaction']:
            for layer in self.interaction:z=layer(z,x)
        new=self.decoder(z,plans,modes,maps,x)
        return {'pool':torch.cat([native,new],2),'new':new}
