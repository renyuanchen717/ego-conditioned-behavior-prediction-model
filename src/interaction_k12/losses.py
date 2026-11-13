import torch
def grouped_mean(value,mask):
    counts=mask.sum(-1);groups=counts>0
    per=(value.masked_fill(~mask,0).sum(-1)/counts.clamp_min(1))
    return per.masked_fill(~groups,0).sum()/groups.sum().clamp_min(1)

def trajectory_errors(pred,gt,valid):
    safe=torch.where(valid[...,None],gt,torch.zeros_like(gt))
    error=(pred-safe[:,:,None]).norm(dim=-1)
    return error.masked_fill(~valid[:,:,None],0).sum(-1)/valid.sum(-1).clamp_min(1)[:,:,None]

def objective(output,native,gt,valid,target_mask,current_valid,lambda_disp=1.,lambda_endpoint=0.):
    valid=valid.bool()&target_mask[...,None];supervised=valid.any(-1)&target_mask
    if not torch.isfinite(gt[valid]).all():raise ValueError('Nonfinite GT at valid timestamp')
    new=output['new'];en=trajectory_errors(new,gt,valid)
    ex=trajectory_errors(native.detach(),gt,valid).amin(-1)
    winner=en.detach().argmin(-1);ep=en.gather(-1,winner[...,None]).squeeze(-1)
    selected=new.gather(2,winner[...,None,None,None].expand(-1,-1,1,80,2)).squeeze(2)
    safe=torch.where(valid[...,None],gt,torch.zeros_like(gt))
    dp=torch.diff(selected,dim=2,prepend=torch.zeros_like(selected[:,:,:1]))
    dy=torch.diff(safe,dim=2,prepend=torch.zeros_like(safe[:,:,:1]))
    prev=torch.cat([current_valid[...,None],valid[:,:,:-1]],-1)
    pairs=valid&prev
    de=(dp-dy).norm(dim=-1).masked_fill(~pairs,0).sum(-1)/pairs.sum(-1).clamp_min(1)
    pool=grouped_mean(torch.minimum(ex,ep),supervised)
    fresh=grouped_mean(ep,supervised)
    disp=grouped_mean(de,supervised&pairs.any(-1))
    endpoint=new.sum()*0
    if lambda_endpoint:
        vals=(selected[:,:, [49,79]]-safe[:,:, [49,79]]).norm(dim=-1);em=valid[:,:, [49,79]]
        endpoint=grouped_mean(vals.masked_fill(~em,0).sum(-1)/em.sum(-1).clamp_min(1),supervised&em.any(-1))
    return dict(loss=pool+fresh+lambda_disp*disp+lambda_endpoint*endpoint,pool=pool,new=fresh,disp=disp,endpoint=endpoint,winner=winner,supervised=supervised,groups=int(supervised.any(-1).sum()))
