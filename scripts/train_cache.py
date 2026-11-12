"""Train the three heads from user-prepared, scene-disjoint target caches.

This portable training recipe uses the original losses and head architectures.
It is not a claim to bitwise reproduce the historical fold-0 checkpoints.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from recorded_k12.components import Head, Response, features, cues, geometry5, descriptors


def load(path, device):
    with np.load(path, allow_pickle=False) as z:
        d={k:torch.as_tensor(z[k],device=device) for k in ('n','q','typ','cv','rec','y','v')}
        ids=z['sid'].astype(str)
    for k in ('n','q','cv','rec','y'):d[k]=d[k].float()
    d['typ']=d['typ'].long();d['v']=d['v'].bool()
    n=len(ids)
    shapes={'n':(n,6,16,2),'q':(n,6),'typ':(n,),'cv':(n,16,2),'rec':(n,16,2),'y':(n,16,2),'v':(n,16)}
    if any(tuple(d[k].shape)!=shape for k,shape in shapes.items()):raise ValueError('Invalid cache shapes')
    if not d['v'].any(1).all():raise ValueError('Every training target needs valid future supervision')
    if any(not torch.isfinite(d[k]).all() for k in ('n','q','cv','rec')) or not torch.isfinite(d['y'][d['v']]).all():raise ValueError('Nonfinite valid inputs/labels')
    if not ((d['typ']>=1)&(d['typ']<=3)).all():raise ValueError('Invalid object types')
    if (d['q']<0).any() or not torch.allclose(d['q'].sum(-1),torch.ones(n,device=device),atol=1e-4):raise ValueError('Probabilities must be normalized')
    d['y']=torch.where(d['v'][...,None],d['y'],torch.zeros_like(d['y']))
    with torch.no_grad():
        d['ctx']=torch.cat([features(d['n'][i],d['q'][i],d['typ'][i],d['cv'][i],d['cv'][i]) for i in torch.arange(n,device=device).split(256)])
        d['axes']=cues(d['n'],d['cv'],d['cv'])[1]
    return d,ids


def errors(mu,y,v):
    distance=(mu-y[:,None]).norm(dim=-1);values=[];masks=[]
    for h in (10,16):
        m=v[:,:h]
        values.extend([distance[:,:,:h].masked_fill(~m[:,None],0).sum(-1)/m.sum(-1).clamp_min(1)[:,None],distance[:,:,h-1]])
        masks.extend([m.any(-1),m[:,-1]])
    return torch.stack(values,-1),torch.stack(masks,-1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',required=True);p.add_argument('--val',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--anchor-steps',type=int,default=750)
    p.add_argument('--response-steps',type=int,default=800);p.add_argument('--batch-size',type=int,default=256)
    a=p.parse_args();torch.set_num_threads(2)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if any((out/f).exists() for f in ('anchor.pt','five.pt','eight.pt')):raise ValueError('Use a fresh output directory to preserve prior checkpoints')
    tr,train_ids=load(a.train,a.device);va,val_ids=load(a.val,a.device)
    if set(train_ids)&set(val_ids):raise ValueError('Scene leakage between train and validation')
    torch.manual_seed(272);anchor=Head().to(a.device)
    opt=torch.optim.AdamW(anchor.parameters(),lr=.0006,weight_decay=.002)
    freq=torch.bincount(tr['typ'],minlength=4)
    schedule=torch.multinomial(freq[tr['typ']].float().rsqrt(),a.batch_size*a.anchor_steps,replacement=True).reshape(a.anchor_steps,a.batch_size)
    for step,i in enumerate(schedule,1):
        residual=anchor(tr['ctx'][i]);base,m=errors(tr['n'][i],tr['y'][i],tr['v'][i])
        pred,_=errors(tr['n'][i]+residual,tr['y'][i],tr['v'][i])
        loss=((base.amin(1).minimum(pred.amin(1))+.15*pred.amin(1))*m).sum(0)/m.sum(0).clamp_min(1)
        loss=(loss*loss.new_tensor([.15/.47,.35/.9,.15/.83,.35/1.76])).sum()+.002*residual.square().mean()
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(anchor.parameters(),2);opt.step()
        if step%200==0:print('anchor',step,float(loss),flush=True)
    anchor.eval().requires_grad_(False)
    torch.save({'anchor':{k:v.cpu() for k,v in anchor.state_dict().items()}},out/'anchor.pt')
    for d in (tr,va):
        with torch.no_grad():d['ref']=d['n']+torch.cat([anchor(c) for c in d['ctx'].split(256)])
    audit={'train_targets':len(train_ids),'val_targets':len(val_ids),'scene_overlap':0,'heads':{}}
    for name,endpoint,descriptor,seed in [('five',9,geometry5,590),('eight',15,descriptors,6791)]:
        torch.manual_seed(seed);net=Response().to(a.device)
        opt=torch.optim.AdamW(net.parameters(),lr=.0006,**({'weight_decay':.01} if name=='eight' else {}))
        subsets=[]
        for d in (tr,va):
            keep=d['v'][:,endpoint]
            if name=='five':keep=keep&(d['typ']==1)&d['v'][:,15]
            ix=torch.where(keep)[0]
            if len(ix)==0:raise ValueError(f'No eligible {name} targets')
            item={k:d[k][ix] for k in ('ctx','axes','typ')}
            item['fixed']=(d['n'][ix,:,endpoint]-d['y'][ix,None,endpoint]).norm(dim=-1).amin(1)
            item['reference']=d['ref'][ix,:,endpoint];item['gt']=d['y'][ix,endpoint]
            item['signal']=torch.cat([descriptor(d['n'][j],d['rec'][j])-descriptor(d['n'][j],d['cv'][j]) for j in ix.split(256)])
            subsets.append(item)
        t,v=subsets;best=0.;scale=0.;selected_step=0
        state={k:z.detach().cpu().clone() for k,z in net.state_dict().items()}
        if name=='eight':
            freq=torch.bincount(t['typ'],minlength=4)
            schedule=torch.multinomial(freq[t['typ']].float().rsqrt(),a.response_steps*a.batch_size,replacement=True).reshape(a.response_steps,a.batch_size)
        else:schedule=None
        for step in range(1,a.response_steps+1):
            i=schedule[step-1] if schedule is not None else torch.randint(len(t['typ']),(a.batch_size,),device=a.device)
            delta=net(t['ctx'][i],t['signal'][i],t['axes'][i])
            distances=(t['reference'][i]+delta-t['gt'][i,None]).norm(dim=-1)
            pool=torch.cat([t['fixed'][i,None],distances],1)
            loss=(.75*pool.amin(1)-.25*.25*torch.logsumexp(-pool/.25,1)).mean()+.01*delta.square().mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
            if step%200==0 or step==a.response_steps:
                with torch.no_grad():
                    change=torch.cat([net(v['ctx'][j],v['signal'][j],v['axes'][j]) for j in torch.arange(len(v['typ']),device=a.device).split(256)])
                    baseline=v['fixed'].minimum((v['reference']-v['gt'][:,None]).norm(dim=-1).amin(1))
                    for s in (.25,.5,1.):
                        error=v['fixed'].minimum((v['reference']+s*change-v['gt'][:,None]).norm(dim=-1).amin(1))-baseline
                        score=float(torch.stack([error[v['typ']==c].mean() for c in (1,2,3) if (v['typ']==c).any()]).mean()) if name=='eight' else float(error.mean())
                        if score<best:best,scale,selected_step,state=score,s,step,{k:z.detach().cpu().clone() for k,z in net.state_dict().items()}
                print(name,step,'selected validation delta',best,'scale',scale,flush=True)
        torch.save({'state':state,'scale':scale,'step':selected_step},out/(name+'.pt'))
        audit['heads'][name]={'validation_delta_m':best,'scale':scale,'step':selected_step}
    weights={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ('anchor.pt','five.pt','eight.pt')}
    (out/'manifest.json').write_text(json.dumps({'weights':weights},indent=2))
    (out/'training_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
