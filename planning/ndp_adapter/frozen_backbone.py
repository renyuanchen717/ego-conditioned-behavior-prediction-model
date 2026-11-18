import hashlib
from pathlib import Path
import torch
from torch import nn
from .nuplan_converter import REPO,EXPECTED
from predictors import NDPNet
from .prior_density import normalized_scores

def state_hash(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def graph_hash(data,ids):
    h=hashlib.sha256(repr(ids).encode())
    for store in data.stores:
        h.update(str(store._key).encode())
        for k,v in sorted(store.items()):
            h.update(k.encode())
            if torch.is_tensor(v):h.update(str((v.dtype,tuple(v.shape))).encode());h.update(v.cpu().contiguous().numpy().tobytes())
            else:h.update(repr(v).encode())
    return h.hexdigest()

class FrozenBackbone(nn.Module):
    def __init__(self,checkpoint,device='cpu',temperature=1.):
        super().__init__();self.checksum=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        if self.checksum!=EXPECTED:raise ValueError('Checkpoint checksum differs from audited official release')
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False);self.hparams=dict(ck['hyper_parameters'])
        if (self.hparams['output_dim'],self.hparams['num_historical_steps'],self.hparams['num_future_steps'])!=(2,11,80):raise ValueError('Incompatible checkpoint')
        self.model=NDPNet(**self.hparams);self.model.load_state_dict(ck['state_dict'],strict=True)
        self.model.requires_grad_(False);self.model.to(device);self.temperature=temperature;self.train(False)
    def train(self,mode=True):
        super().train(False);return self
    @torch.no_grad()
    def forward_history_only(self,data):
        self.model.eval();d=data.clone()
        for key in ['position','velocity','heading']:d['agent'][key][:,11:]=0
        # Inference queries are defined by observed CURRENT presence, not GT validity.
        current=d['agent'].get('current_observed')
        if current is None:
            # Converter preserves current validity separately from vector validity.
            # Legacy sanitized contexts already contain a fixed history-only query.
            current=d['agent'].predict_mask[:,11:].any(-1).clone()
        d['agent'].valid_mask[:,11:]=False;d['agent'].predict_mask[:,:11]=False
        d['agent'].predict_mask[:,11:]=current[:,None]
        enc=self.model.encoder(d);out=self.model.decoder(d,enc)
        if out['scale_refine_pos'].min()<=.05:raise ValueError('Prior scale violates adapter floor')
        return enc,dict(mu=out['loc_refine_pos'],scale=out['scale_refine_pos'],log_w=normalized_scores(out['pi'],self.temperature),official_logits=out['pi'])
