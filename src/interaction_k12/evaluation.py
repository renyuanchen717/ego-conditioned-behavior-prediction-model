import numpy as np
import torch
from .data import model_inputs,labels
from .losses import trajectory_errors

@torch.inference_mode()
def evaluate(model,loader,device):
    model.eval();records=[];group_ade=[];group_losses=[]
    diagnostics={'new_wins':[],'new_gain_m':[],'utilization':[0]*6,'new_new_mean_distance_m':[],'new_native_min_distance_m':[],'duplicate_new_fraction':[],'speed_over_50mps_fraction':[],'acceleration_over_15mps2_fraction':[],'best_new_displacement_error_m':[]}
    for batch in loader:
        x=model_inputs(batch,device);l=labels(batch,device);out=model(x);n=out['new'];native=x['native']
        valid=l['future_valid']&l['target_mask'][...,None];gt=torch.where(valid[...,None],l['gt'],torch.zeros_like(l['gt']))
        target=valid.any(-1);row={'types':x['types'][target].cpu().numpy()}
        for name,pred in [('frozen6',native),('new6',n),('pool12',out['pool'])]:
            distance=(pred-gt[:,:,None]).norm(dim=-1)
            for h in (3,5,8):
                end=h*10;m=valid[:,:,:end]
                ade=distance[:,:,:,:end].masked_fill(~m[:,:,None],0).sum(-1)/m.sum(-1).clamp_min(1)[:,:,None]
                best=ade.amin(-1).masked_fill(~m.any(-1),torch.nan)
                fde=distance[:,:,:,end-1].amin(-1).masked_fill(~m[:,:,-1],torch.nan)
                row[f'{name}_{h}s_minADE']=best[target].cpu().numpy();row[f'{name}_{h}s_minFDE']=fde[target].cpu().numpy()
                if name=='pool12' and h==8:
                    for b in range(len(best)):
                        ok=torch.isfinite(best[b]);
                        if ok.any():group_ade.append(float(best[b,ok].mean()))
        records.append(row)
        ep=trajectory_errors(n,gt,valid);ex=trajectory_errors(native,gt,valid).amin(-1)
        idx=ep.argmin(-1);best=ep.amin(-1)
        diagnostics['new_wins'].extend((best[target]<ex[target]).cpu().tolist());diagnostics['new_gain_m'].extend((ex-best)[target].cpu().tolist())
        winners=idx[target].cpu();hist=torch.bincount(winners,minlength=6).tolist()
        diagnostics['utilization']=[a+b for a,b in zip(diagnostics['utilization'],hist)]
        pred=n[target]
        if len(pred):
            nn=(pred[:,:,None]-pred[:,None]).norm(dim=-1).mean(-1);tri=torch.triu(torch.ones(6,6,device=device,dtype=torch.bool),diagonal=1)
            diagnostics['new_new_mean_distance_m'].extend(nn[:,tri].mean(-1).cpu().tolist())
            diagnostics['duplicate_new_fraction'].extend((nn[:,tri]<.5).float().mean(-1).cpu().tolist())
            nx=(pred[:,:,None]-native[target][:,None]).norm(dim=-1).mean(-1).amin(-1).mean(-1)
            diagnostics['new_native_min_distance_m'].extend(nx.cpu().tolist())
            speed=torch.diff(pred,dim=2,prepend=torch.zeros_like(pred[:,:,:1]))*10
            acceleration=torch.diff(speed,dim=2)*10
            diagnostics['speed_over_50mps_fraction'].extend((speed.norm(dim=-1)>50).float().mean((1,2)).cpu().tolist())
            diagnostics['acceleration_over_15mps2_fraction'].extend((acceleration.norm(dim=-1)>15).float().mean((1,2)).cpu().tolist())
            sel=n.gather(2,idx[...,None,None,None].expand(-1,-1,1,80,2)).squeeze(2)
            pairs=valid&torch.cat([l['current_valid'][...,None],valid[:,:,:-1]],-1)
            dp=torch.diff(sel,dim=2,prepend=torch.zeros_like(sel[:,:,:1]));dy=torch.diff(gt,dim=2,prepend=torch.zeros_like(gt[:,:,:1]))
            err=(dp-dy).norm(dim=-1).masked_fill(~pairs,0).sum(-1)/pairs.sum(-1).clamp_min(1)
            diagnostics['best_new_displacement_error_m'].extend(err[target&pairs.any(-1)].cpu().tolist())
    if not records:raise ValueError('Empty evaluator dataset')
    allrows={k:np.concatenate([r[k] for r in records]) for k in records[0]}
    report={'targets':len(allrows['types']),'groups':len(group_ade),'selection_pool8_group_minADE':float(np.mean(group_ade)) if group_ade else None,'metrics':{}}
    for group,mask in [('micro',np.ones(len(allrows['types']),bool)),*[(f'class_{c}',allrows['types']==c) for c in (1,2,3)]]:
        report['metrics'][group]={}
        for k,v in allrows.items():
            if k=='types':continue
            ok=mask&np.isfinite(v);report['metrics'][group][k]={'mean':float(v[ok].mean()) if ok.any() else None,'count':int(ok.sum())}
    report['metrics']['class_macro']={}
    for k in report['metrics']['micro']:
        values=[report['metrics'][f'class_{c}'][k]['mean'] for c in (1,2,3)]
        good=[v for v in values if v is not None]
        report['metrics']['class_macro'][k]={'mean':float(np.mean(good)) if good else None,'classes':len(good)}
    report['diagnostics']={k:({'mean':float(np.mean(v)),'p10_p50_p90':np.quantile(np.asarray(v,dtype=float),[.1,.5,.9]).tolist()} if v and k!='utilization' else v) for k,v in diagnostics.items()}
    report['diagnostic_thresholds']={'duplicate_mean_distance_m':.5,'speed_mps':50,'acceleration_mps2':15,'safety_guarantee':False}
    return report
