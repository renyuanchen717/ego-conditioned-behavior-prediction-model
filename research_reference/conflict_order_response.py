from pathlib import Path
import sys
import torch
from signed_plan_response import R,cues,run

OUT=R/'conflict_order_response';OUT.mkdir(exist_ok=True)

def descriptors(target,plan):
    # A soft spatial conflict of the two paths. Times are compared at nearby
    # spatial points, without requiring agents to arrive simultaneously.
    sep=torch.linalg.vector_norm(target[:,:,:,None]-plan[:,None,None,:],dim=-1)
    grid=torch.arange(1,17,device=target.device)*.5
    lag=grid[None,:]-grid[:,None]  # ego arrival minus target arrival
    sync=sep.diagonal(dim1=-2,dim2=-1)
    out=[]
    for lo,hi in ((0,8),(8,16)):
        distances=sep[:,:,lo:hi]
        mass=torch.softmax(-distances.flatten(2)/1.5,-1).reshape_as(distances)
        strength=torch.exp(-distances.amin((-1,-2))/8)
        delay=lag[lo:hi]
        out.append(strength*(mass*delay).sum((-1,-2)))
        out.append(4*strength*(mass*torch.tanh(delay/.75)).sum((-1,-2)))
        out.append(8*torch.exp(-sync[:,:,lo:hi].amin(2)/8))
    return torch.stack(out,-1)

def build(n,cv,rec):
    actual=[];control=[];bases=[]
    for x,c,r in zip(n.split(256),cv.split(256),rec.split(256)):
        x,c,r=x.cuda(),c.cuda(),r.cuda()
        reference=descriptors(x,c)
        actual.append(descriptors(x,r)-reference)
        control.append(descriptors(x,2*c-r)-reference)
        bases.append(cues(x,c,r)[1])
    return tuple(torch.cat(v) for v in (actual,control,bases))

if __name__=='__main__':run(int(sys.argv[1]),build,OUT)
