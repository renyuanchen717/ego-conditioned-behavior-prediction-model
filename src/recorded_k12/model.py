"""Locked K=12 trajectory pool. No impact router or learned 12-mode probabilities."""
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .components import Head, Response, features, cues, geometry5, descriptors

class RecordedK12(nn.Module):
    def __init__(self, checkpoints, verify=True):
        super().__init__()
        folder=Path(checkpoints)
        if verify:
            manifest=json.loads((folder/'manifest.json').read_text())
            for name, expected in manifest['weights'].items():
                if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=expected:
                    raise ValueError(f'Checkpoint checksum mismatch: {name}')
        self.anchor=Head()
        self.anchor.load_state_dict(torch.load(folder/'anchor.pt',map_location='cpu',weights_only=True)['anchor'])
        self.responses=nn.ModuleList([Response(),Response()])
        scales=[]
        for net,name in zip(self.responses,['five.pt','eight.pt']):
            ck=torch.load(folder/name,map_location='cpu',weights_only=True)
            net.load_state_dict(ck['state']);scales.append(float(ck['scale']))
        self.register_buffer('scales',torch.tensor(scales))
        self.eval().requires_grad_(False)

    def forward(self, native, probabilities, object_type, ego_cv, ego_plan):
        """Inputs: [N,6,16,2], [N,6], [N], [N,16,2], [N,16,2].
        All trajectories in each target's local frame, t=0.5,...,8s.
        Returns [N,12,16,2], native modes first and untouched.
        """
        n=len(native)
        expected=[(native,(n,6,16,2)),(probabilities,(n,6)),(object_type,(n,)),
                  (ego_cv,(n,16,2)),(ego_plan,(n,16,2))]
        if any(tuple(x.shape)!=shape for x,shape in expected):
            raise ValueError('Invalid tensor shapes; see forward docstring')
        if any(not torch.isfinite(x).all() for x,_ in expected):
            raise ValueError('Non-finite input; exclude targets with unreliable inputs')
        if not ((object_type>=1)&(object_type<=3)).all():
            raise ValueError('WOMD types must be 1 vehicle, 2 pedestrian, 3 cyclist')
        if (probabilities<0).any() or not torch.allclose(probabilities.sum(-1),torch.ones(n,device=native.device),atol=1e-4):
            raise ValueError('Expected normalized probabilities, not logits')
        context=features(native,probabilities,object_type,ego_cv,ego_cv)
        reference=native+self.anchor(context)
        axes=cues(native,ego_cv,ego_cv)[1]
        d5=self.scales[0]*self.responses[0](context,geometry5(native,ego_plan)-geometry5(native,ego_cv),axes)
        d8=self.scales[1]*self.responses[1](context,descriptors(native,ego_plan)-descriptors(native,ego_cv),axes)
        times=torch.arange(1,17,device=native.device,dtype=native.dtype)*.5
        def smooth(x):return (x*x*(3-2*x))[None,None,:,None]
        h5=smooth((times/5).clamp(max=1));h8=smooth(((times-5)/3).clamp(min=0))
        proposals=reference+h5*(1-h8)*d5[:,:,None]+h8*d8[:,:,None]
        return torch.cat([native,proposals],1)

    def predict_10hz(self,native,probabilities,object_type,ego_cv,ego_plan):
        """80 steps at 10 Hz; preserve native between-knot geometry, interpolate residual only."""
        if native.shape[1:]!=(6,80,2) or ego_cv.shape[1:]!=(80,2) or ego_plan.shape[1:]!=(80,2):
            raise ValueError('Expected native [N,6,80,2] and plans [N,80,2]')
        x=native[:,:,4::5]
        pool=self(x,probabilities,object_type,ego_cv[:,4::5],ego_plan[:,4::5])
        residual=pool[:,6:]-x
        knots=F.pad(residual.transpose(-1,-2),(1,0)).reshape(-1,2,17)
        dense=F.interpolate(knots,size=81,mode='linear',align_corners=True)[:,:,1:]
        proposals=native+dense.reshape(len(native),6,2,80).transpose(-1,-2)
        return torch.cat([native,proposals],1)
