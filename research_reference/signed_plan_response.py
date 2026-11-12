"""Odd response to signed ego-plan deviation in the endpoint frame of each target mode."""
import json,sys
import numpy as np
import torch
from torch import nn
from cvdev_head import R,Head,features
from fixed6_path import load
O=R/'signed_plan_response';O.mkdir(exist_ok=True)

def cues(n,cv,rec):
    velocity=n[:,:,-1]-n[:,:,9]
    heading=velocity/velocity.norm(dim=-1,keepdim=True).clamp_min(.1)
    heading=torch.where((velocity.norm(dim=-1)<.1)[:,:,None],n.new_tensor([1.,0.]),heading)
    normal=torch.stack([-heading[...,1],heading[...,0]],-1)
    basis=torch.stack([heading,normal],-2)
    dev=(rec-cv)[:,None].expand_as(n)
    gap=(n-cv[:,None]).norm(dim=-1)
    signed=[]
    for lo,hi in ((0,6),(6,10),(10,16)):
        att=torch.softmax(-gap[:,:,lo:hi]/6,dim=-1)
        action=(att[...,None]*dev[:,:,lo:hi]).sum(2)
        signed.append(torch.einsum('bkij,bkj->bki',basis,action))
    return torch.cat(signed,-1),basis

class Response(nn.Module):
    def __init__(self):
        super().__init__()
        self.coeff=nn.Sequential(nn.Linear(64,48),nn.SiLU(),nn.Linear(48,12))
        nn.init.zeros_(self.coeff[-1].weight);nn.init.zeros_(self.coeff[-1].bias)
    def forward(self,context,signal,basis):
        mat=self.coeff(context).reshape(*signal.shape[:2],2,6)
        local=(mat@(signal/8).clamp(-3,3).unsqueeze(-1)).squeeze(-1)
        return torch.einsum('bkij,bki->bkj',basis,2*torch.tanh(local))

def run(rank, cue_builder=None, destination=O):
    torch.set_num_threads(2);torch.cuda.set_device(rank)
    d=load();valid=d['v'][:,-1].astype(bool);sid=d['sid'][valid]
    n,q,t,y,cv,rec=[torch.from_numpy(d[k][valid]) for k in ('n','q','typ','y','cv','rec')]
    with torch.no_grad():
        ctx=torch.cat([features(a,b,c,p,p) for a,b,c,p in zip(n.split(512),q.split(512),t.split(512),cv.split(512))]).cuda()
        if cue_builder is None:
            signal,basis=[x.cuda() for x in cues(n,cv,rec)]
            mirror=-signal
        else:
            signal,mirror,basis=cue_builder(n,cv,rec)
    typ=t.numpy();t=t.cuda();end=n[:,:,-1].cuda();gt=y[:,-1].cuda()
    fixed=(end-gt[:,None]).norm(dim=-1).amin(1)
    for fold in range(rank,5,2):
        src=R/'cv_direct_crossfit'/f'fold_{fold}';audit=json.loads((src/'audit.json').read_text())
        sets=[set(audit[k]) for k in ('fit_scenes','inner_scenes','outer_scenes')]
        assert all(not sets[i]&sets[j] for i,j in ((0,1),(1,2),(0,2)))
        tr,inner,test=[torch.from_numpy(np.flatnonzero(np.isin(sid,list(s)))).cuda() for s in sets]
        anchor=Head().cuda().eval();anchor.load_state_dict(torch.load(src/'selected.pt',weights_only=True)['anchor'])
        with torch.no_grad(): ref=end+torch.cat([anchor(c)[:,:,-1] for c in ctx.split(512)])
        def pool(ix,change):
            return fixed[ix].minimum((ref[ix]+change-gt[ix,None]).norm(dim=-1).amin(1))
        torch.manual_seed(6791+fold);net=Response().cuda()
        opt=torch.optim.AdamW(net.parameters(),lr=.0006,weight_decay=.01)
        freq=torch.bincount(t[tr].long(),minlength=4)
        schedule=tr[torch.multinomial(freq[t[tr].long()].float().rsqrt(),256*800,replacement=True).reshape(800,256)]
        best=float('inf');choice=None
        for step,ix in enumerate(schedule,1):
            a=net(ctx[ix],signal[ix],basis[ix])
            dist=(ref[ix]+a-gt[ix,None]).norm(dim=-1)
            oracle=dist.amin(1).minimum(fixed[ix])
            soft=-.25*torch.logsumexp(-torch.cat([fixed[ix,None],dist],1)/.25,1)
            loss=(.75*oracle+.25*soft).mean()+.01*a.square().mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
            if step%200==0:
                with torch.no_grad():
                    a=net(ctx[inner],signal[inner],basis[inner])
                    for scale in (.25,.5,1.):
                        delta=pool(inner,scale*a)-pool(inner,0)
                        score=np.mean([float(delta[t[inner]==c].mean()) for c in (1,2,3)])
                        if score<best:
                            best=score;choice=(step,scale,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
        step,scale,state=choice;net.load_state_dict(state)
        with torch.no_grad():
            a=scale*net(ctx[test],signal[test],basis[test]); zero=torch.zeros_like(a)
            reflected=scale*net(ctx[test],mirror[test],basis[test])
            torch.testing.assert_close(scale*net(ctx[test],0*signal[test],basis[test]),zero)
            err=torch.stack([pool(test,z) for z in (zero,a,reflected)],1).cpu().numpy()
        ii=test.cpu().numpy()
        with np.load(R/'cv_follow_weight'/f'fold_{fold}.npz') as old:
            mask=old['valid'][:,1].astype(bool)
            assert np.array_equal(np.flatnonzero(valid)[ii],old['ix'][mask])
            np.testing.assert_allclose(err[:,0],old['err'][mask,1,0],atol=1e-4,rtol=0)
            previous=old['err'][mask,1,2]
        np.savez(destination/f'fold_{fold}.npz',sid=sid[ii],typ=typ[ii],err=np.column_stack([err,previous]))
        torch.save(dict(state=state,scale=scale,step=step,fold=fold),destination/f'fold_{fold}.pt')
        print(fold,step,scale,flush=True)
if __name__=='__main__':run(int(sys.argv[1]))
