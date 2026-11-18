import hashlib,json
from pathlib import Path
import torch
from ndp_adapter.frozen_backbone import FrozenBackbone,graph_hash,state_hash

SOURCE_SHA='814544e98171f2bd1e27e63f99aa4ba6794f05ab'
CACHE_VERSION='mode-v4-1'
class FrozenFeatures(FrozenBackbone):
    @torch.no_grad()
    def extract(self,context,return_mode_features=True,max_targets=10,target_types=(1,)):
        data=context['data'];meta=context['metadata'];ids=meta['ordered_track_ids']
        # Single-root graph export avoids any ambiguous PyG av_index offsets.
        if getattr(data,'num_graphs',1)!=1:raise ValueError('Export individual root graphs; padded adapter batching is separate')
        ego=ids.index(meta['ego_track_id'])
        if int(data['agent'].av_index)!=ego:raise ValueError('Ego identity/index mismatch')
        current=data['agent'].current_observed.bool();pos=data['agent'].position[:,10,:2]
        indices=[i for i in range(len(ids)) if current[i] and int(data['agent'].type[i]) in target_types and i!=ego]
        indices.sort(key=lambda i:(float(((pos[i]-pos[ego])**2).sum()),ids[i]));indices=indices[:max_targets]
        if not indices:raise ValueError('No history-selected targets')
        original=graph_hash(data,ids);captured=[]
        hook=self.model.decoder.to_pi.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone())) if return_mode_features else None
        try:enc,pred=self.forward_history_only(data.clone().to(next(self.model.parameters()).device))
        finally:
            if hook is not None:hook.remove()
        # .to on PyG can mutate tensors/device: hash a CPU clone, values must agree.
        if graph_hash(data.clone().cpu(),ids)!=original:raise AssertionError('Input graph mutated')
        if not return_mode_features:return enc,pred
        if len(captured)!=1:raise ValueError('Expected exactly one final native mode-head invocation')
        hp=self.hparams;S=int(hp['num_proximal_steps']);T=int(hp['num_future_steps'])
        if not 0<S<T:raise ValueError('Invalid checkpoint temporal split')
        ctx=current.nonzero().flatten().tolist();mapping={j:i for i,j in enumerate(ctx)}
        cpu=lambda x:x.detach().cpu()
        result={k:cpu(pred[k][indices]) for k in ['mu','scale','log_w']}
        result.update(Ea=cpu(enc['x_a'][ctx,-1]),Em=cpu(enc['x_pl'][:,-1]),U=cpu(captured[0][indices]),Ee=cpu(enc['x_a'][ego,-1]),target_index=torch.tensor([mapping[i] for i in indices]),ego_role=torch.tensor([i==ego for i in ctx]),ego_origin=pos[ego].cpu().clone(),ego_yaw=data['agent'].heading[ego,10].cpu().clone(),origin=pos[indices].cpu().clone(),yaw=data['agent'].heading[indices,10].cpu().clone(),actor_valid=torch.ones(len(indices),dtype=torch.bool),context_valid=torch.ones(len(ctx),dtype=torch.bool),map_valid=torch.ones(enc['x_pl'].shape[0],dtype=torch.bool))
        result['metadata']=dict(context_hash=original,track_ids=[ids[i] for i in indices],context_ids=[ids[i] for i in ctx],ego_track_id=ids[ego],checkpoint_hash=self.checksum,source_sha=SOURCE_SHA,version=CACHE_VERSION,precision='float32',frame=meta['frame'],native_S=S,T=T,d0=hp['hidden_dim'],K=hp['num_modes'],history_steps=hp['num_historical_steps'])
        result['metadata']['cache_key']=hashlib.sha256(json.dumps(result['metadata'],sort_keys=True).encode()).hexdigest()
        return result
