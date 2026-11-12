import torch
def displacement_metrics(pool,gt,valid,hz=2,horizons=(3,5,8)):
    """Per-target oracle displacement errors. FDE requires a valid endpoint.
    ADE and FDE independently select their best mode. Not official WOMD mAP/MR.
    """
    if pool.ndim!=4 or gt.shape!=(pool.shape[0],pool.shape[2],2) or valid.shape!=gt.shape[:2]:
        raise ValueError('Invalid pool, GT or mask shape')
    valid=valid.bool()
    if not torch.isfinite(pool).all() or not torch.isfinite(gt[valid]).all():
        raise ValueError('Nonfinite prediction or valid GT')
    dist=(pool-torch.where(valid[...,None],gt,torch.zeros_like(gt))[:,None]).norm(dim=-1)
    out={}
    for h in horizons:
        end=round(h*hz)
        if end>pool.shape[2]:raise ValueError('Horizon exceeds available future')
        mask=valid[:,:end]
        ade=dist[:,:,:end].masked_fill(~mask[:,None],0).sum(-1)/mask.sum(-1).clamp_min(1)[:,None]
        out[f'{h}s_minADE']=ade.amin(1).masked_fill(~mask.any(1),torch.nan)
        out[f'{h}s_minFDE']=dist[:,:,end-1].amin(1).masked_fill(~mask[:,-1],torch.nan)
    return out
