import json,sys
from pathlib import Path
import numpy as np
import torch
from cvdev_head import Head,load,R,error_modes
from cv_horizon import Correction,quantities,table
O=R/'cv_direct_crossfit'
O.mkdir(exist_ok=True)
def groups(ids,t,v):
    unique,inv=np.unique(ids,return_inverse=True)
    labels=np.zeros(len(unique),int)
    # Balance availability of rare endpoint labels without using trajectory errors.
    for c in [2,3]:np.maximum.at(labels,inv[(t==c)&v],c)
    rng=np.random.default_rng(9013);parts=np.full(len(unique),-1)
    for c in np.unique(labels):
        ix=np.flatnonzero(labels==c);rng.shuffle(ix)
        parts[ix]=np.arange(len(ix))%10
    return parts[inv]
def batches(t,idx,steps):
    freq=torch.bincount(t[idx].long(),minlength=4)
    return idx[torch.multinomial(freq[t[idx].long()].float().rsqrt(),256*steps,replacement=True).view(steps,256)]
def run(rank):
    torch.cuda.set_device(rank);torch.set_num_threads(2)
    d,ids=load(lambda n,q,t:n)
    t,v,y=d['t'],d['v'],d['y'];x,xc=d['x'],d['xc'];off=d['off']
    bins=groups(ids,t.cpu().numpy(),v[:,-1].cpu().numpy())
    outer=bins//2
    for fold in range(rank,5,2):
        directory=O/f'fold_{fold}';directory.mkdir(exist_ok=True)
        test=outer==fold;inner=bins==(2*fold+2)%10;fit=~(inner|test)
        sets=[set(ids[mask]) for mask in [fit,inner,test]]
        assert all(not sets[i]&sets[j] for i,j in [(0,1),(0,2),(1,2)])
        ti,vi,ei=[torch.from_numpy(np.flatnonzero(mask)).cuda() for mask in [fit,inner,test]]
        torch.manual_seed(272+fold)
        anchor=Head().cuda();opt=torch.optim.AdamW(anchor.parameters(),lr=.0006,weight_decay=.002)
        for i in batches(t,ti,750):
            a=anchor(xc[i]);b,m=error_modes(off[i],y[i],v[i]);p,_=error_modes(off[i]+a,y[i],v[i])
            loss=((b.amin(1).minimum(p.amin(1))+.15*p.amin(1))*m).sum(0)/m.sum(0).clamp_min(1)
            loss=(loss*loss.new_tensor([.15/.47,.35/.9,.15/.83,.35/1.76])).sum()+.002*a.square().mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(anchor.parameters(),2);opt.step()
        anchor.eval().requires_grad_(False)
        ep=[9,15];o=off[:,:,ep];gt=y[:,ep];valid=v[:,ep]
        with torch.no_grad():ref=o+torch.cat([anchor(a)[:,:,ep] for a in xc.split(512)])
        def get(model,ii):
            with torch.no_grad():
                vals=[quantities(o[i],ref[i],model(x[i],xc[i]),gt[i]) for i in ii.split(512)]
                return [torch.cat(p) for p in zip(*vals)]
        records=[];selected={}
        for name in ('direct','smooth'):
            torch.manual_seed(804+fold)
            model=Correction(True).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0004,weight_decay=.01)
            best=float('inf')
            for step,i in enumerate(batches(t,ti,800),1):
                extra=model(x[i],xc[i])
                cv,rec=quantities(o[i],ref[i],extra,gt[i])
                delta=(rec-cv)/rec.new_tensor([.9,1.76])
                cost=delta+.4*delta.relu()
                if name=='smooth':
                    # Give gradients to plausible runners-up without pushing the frozen modes.
                    fixed=(o[i]-gt[i,None]).norm(dim=-1)
                    shifted=(ref[i]+extra-gt[i,None]).norm(dim=-1)
                    tau=rec.new_tensor([.15,.25])
                    soft=-tau*torch.logsumexp(-torch.cat([fixed,shifted],1)/tau,dim=1)
                    cost=.6*cost+.4*(soft-cv.detach())/rec.new_tensor([.9,1.76])
                loss=((cost*valid[i]).sum(0)/valid[i].sum(0).clamp_min(1)).mean()+.003*extra.square().mean()
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
                if step%200==0:
                    cv0,p0=get(model,vi);delta=table(p0-cv0,valid[vi],t[vi]).mean(0)
                    key=float((delta/delta.new_tensor([.9,1.76])).max())
                    assert np.isfinite(key)
                    records.append(dict(name=name,step=step,inner_delta_m=delta.tolist(),score=key))
                    if key<best:
                        best=key;selected[name]=(key,step,{k:z.detach().cpu().clone() for k,z in model.state_dict().items()})
        winner=min(selected,key=lambda k:selected[k][0])
        predictions_out=[]
        for name in ('direct','smooth'):
            model.load_state_dict(selected[name][2]);base,pred=get(model,ei);predictions_out.append(pred.cpu().numpy())
        torch.save(dict(anchor=anchor.state_dict(),correction=selected[winner][2],variant=winner,step=selected[winner][1]),directory/'selected.pt')
        np.savez(directory/'oof.npz',target_index=ei.cpu().numpy(),sid=ids[test],typ=t[ei].cpu().numpy(),mask=valid[ei].cpu().numpy(),cv=base.cpu().numpy(),direct=predictions_out[0],smooth=predictions_out[1],selected=predictions_out[int(winner=='smooth')])
        (directory/'audit.json').write_text(json.dumps(dict(fold=fold,winner=winner,step=selected[winner][1],fit_scenes=sorted(sets[0]),inner_scenes=sorted(sets[1]),outer_scenes=sorted(sets[2]),records=records),indent=2))
        print(fold,winner,selected[winner][1],flush=True)
if __name__=='__main__':run(int(sys.argv[1]))
