import json,sys
import torch
import numpy as np
from conflict_order_response import descriptors
from signed_plan_response import R,Head,Response,features,load,cues

O=R/'separate_horizon12'
O.mkdir(exist_ok=True)

def geometry5(x,plan):
    # Arrival order and synchronous proximity before 5s only.
    d=(x[:,:,:10,None]-plan[:,None,None,:10]).norm(dim=-1)
    time=torch.arange(10,device=x.device)*.5
    lag=time[None,:]-time[:,None]
    result=[]
    for lo,hi in [(0,5),(5,10)]:
        window=d[:,:,lo:hi]
        weight=(-window.flatten(2)/1.5).softmax(-1).reshape_as(window)
        strength=(-window.amin((-1,-2))/8).exp()
        result.extend([strength*(weight*lag[lo:hi]).sum((-1,-2)),
            4*strength*(weight*(lag[lo:hi]/.75).tanh()).sum((-1,-2)),
            8*(-d.diagonal(dim1=-2,dim2=-1)[:,:,lo:hi].amin(2)/8).exp()])
    return torch.stack(result,-1)

def fit(rank):
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    d=load()
    ok=(d['typ']==1)&d['v'][:,9].astype(bool)&d['v'][:,15].astype(bool)
    sid=d['sid'][ok]
    x,q,t,y,b,r=[torch.from_numpy(d[k][ok]).cuda() for k in ['n','q','typ','y','cv','rec']]
    with torch.no_grad():
        ctx=[];sig=[];mirror=[];axes=[]
        for i in torch.arange(len(x),device=x.device).split(256):
            ctx.append(features(x[i],q[i],t[i],b[i],b[i]))
            base=geometry5(x[i],b[i])
            sig.append(geometry5(x[i],r[i])-base)
            mirror.append(geometry5(x[i],2*b[i]-r[i])-base)
            axes.append(cues(x[i],b[i],b[i])[1])
        ctx,sig,mirror,axes=map(torch.cat,[ctx,sig,mirror,axes])
        frozen=(x[:,:,9]-y[:,None,9]).norm(dim=-1).amin(1)
    for f in range(rank,5,2):
        folder=R/'cv_direct_crossfit'/f'fold_{f}'
        split=json.loads((folder/'audit.json').read_text())
        lists=[set(split[k]) for k in ['fit_scenes','inner_scenes','outer_scenes']]
        assert all(not lists[a]&lists[b_] for a,b_ in [(0,1),(1,2),(0,2)])
        tr,vi,te=[torch.from_numpy(np.flatnonzero(np.isin(sid,list(s)))).cuda() for s in lists]
        anchor=Head().cuda().eval()
        anchor.load_state_dict(torch.load(folder/'selected.pt',weights_only=True)['anchor'])
        with torch.no_grad():
            ref=x[:,:,9]+torch.cat([anchor(c)[:,:,9] for c in ctx.split(512)])
        def err(i,delta):
            return (ref[i]+delta-y[i,None,9]).norm(dim=-1)
        def metric(i,delta):
            return frozen[i].minimum(err(i,delta).amin(1))
        torch.manual_seed(590+f)
        net=Response().cuda()
        opt=torch.optim.AdamW(net.parameters(),lr=.0006)
        best=0.;scale=0.;state={k:v.cpu().clone() for k,v in net.state_dict().items()}
        for step in range(1,801):
            i=tr[torch.randint(len(tr),(256,),device=x.device)]
            change=net(ctx[i],sig[i],axes[i])
            ds=err(i,change)
            pool=torch.cat([frozen[i,None],ds],1)
            loss=(.75*pool.amin(1)-.25*.25*torch.logsumexp(-pool/.25,1)).mean()+.01*change.square().mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
            if step%200==0:
                with torch.no_grad():
                    value=net(ctx[vi],sig[vi],axes[vi])
                    for s in [.25,.5,1.]:
                        risk=(metric(vi,s*value)-metric(vi,0)).mean().item()
                        if risk<best:
                            best,scale=risk,s
                            state={k:v.cpu().clone() for k,v in net.state_dict().items()}
        net.load_state_dict(state)
        with torch.no_grad():
            vals=[metric(te,0),metric(te,scale*net(ctx[te],sig[te],axes[te])),metric(te,scale*net(ctx[te],mirror[te],axes[te]))]
        np.savez(O/f'fold_{f}.npz',sid=sid[te.cpu()],e=torch.stack(vals,1).cpu().numpy())
        torch.save(dict(state=state,scale=scale),O/f'fold_{f}.pt')

if __name__=='__main__': fit(int(sys.argv[1]))
